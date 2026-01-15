"""
=====================================================================
MÔ-ĐUN FUSION (KẾT HỢP CÁC MODULES)
=====================================================================
Mô tả:
    Module này kết hợp các anomaly scores và features từ tất cả các
    modules khác để ra quyết định cuối cùng: Ảnh là Real hay Fake.
    
    Các nguồn đầu vào:
    1. Transformer perplexity score (từ VQ-VAE + GPT)
    2. Diffusion anomaly score (từ DDPM)
    3. CNN/ViT features và predictions
    4. GCN structural features
    
    Phương pháp fusion:
    - Weighted sum: Tổng có trọng số các scores
    - MLP fusion: Học cách kết hợp tối ưu
    - Attention fusion: Tự động học trọng số
    
Kiến trúc đề xuất:
    1. Score normalization: Đưa các scores về cùng scale
    2. Feature concatenation: Ghép các features
    3. Fusion network: Học cách kết hợp
    4. Final classification: Ra quyết định
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import config
from config import FusionConfig, CNNViTConfig, GCNConfig


# =====================================================================
# SCORE NORMALIZATION
# =====================================================================

class ScoreNormalizer(nn.Module):
    """
    Normalize các anomaly scores về cùng scale [0, 1]
    
    Các scores từ các modules khác nhau có range rất khác nhau:
    - Transformer perplexity: có thể từ 1 đến hàng nghìn
    - Diffusion residual: thường nhỏ (< 1)
    - CNN probability: từ 0 đến 1
    
    Cần normalize để có thể kết hợp hợp lý.
    
    Args:
        method: Phương pháp normalize
            - 'minmax': Min-max scaling
            - 'sigmoid': Sigmoid transformation
            - 'learned': Học parameters để normalize
    """
    
    def __init__(self, method='sigmoid'):
        super().__init__()
        self.method = method
        
        if method == 'learned':
            # Học scale và shift cho mỗi loại score
            self.scale = nn.Parameter(torch.ones(4))  # 4 modules
            self.shift = nn.Parameter(torch.zeros(4))
    
    def forward(self, scores):
        """
        Normalize scores
        
        Args:
            scores: Dict hoặc tensor chứa các scores
            
        Returns:
            Normalized scores
        """
        if isinstance(scores, dict):
            normalized = {}
            for key, score in scores.items():
                normalized[key] = self._normalize_single(score)
            return normalized
        else:
            return self._normalize_single(scores)
    
    def _normalize_single(self, score):
        """Normalize một score"""
        if self.method == 'sigmoid':
            # Sigmoid transformation - luôn output [0, 1]
            return torch.sigmoid(score)
        elif self.method == 'minmax':
            # Min-max trong batch
            min_val = score.min()
            max_val = score.max()
            if max_val > min_val:
                return (score - min_val) / (max_val - min_val)
            return score
        elif self.method == 'learned':
            # TODO: Implement learned normalization
            return torch.sigmoid(score)
        else:
            return score


# =====================================================================
# WEIGHTED SUM FUSION
# =====================================================================

class WeightedSumFusion(nn.Module):
    """
    Fusion đơn giản bằng tổng có trọng số
    
    final_score = w1*s1 + w2*s2 + w3*s3 + w4*s4
    
    Các trọng số có thể được định nghĩa trước hoặc học.
    
    Args:
        weights: Dict trọng số cho mỗi module
        learnable: Có học trọng số không
    """
    
    def __init__(self, weights=None, learnable=False):
        super().__init__()
        
        if weights is None:
            weights = FusionConfig.WEIGHTS
        
        self.module_names = list(weights.keys())
        
        if learnable:
            # Learnable weights (softmax để tổng = 1)
            init_weights = torch.tensor([weights[k] for k in self.module_names])
            self.raw_weights = nn.Parameter(init_weights)
        else:
            # Fixed weights
            self.register_buffer(
                'fixed_weights',
                torch.tensor([weights[k] for k in self.module_names])
            )
            self.raw_weights = None
    
    @property
    def weights(self):
        """Lấy normalized weights"""
        if self.raw_weights is not None:
            return F.softmax(self.raw_weights, dim=0)
        return self.fixed_weights
    
    def forward(self, scores):
        """
        Compute weighted sum của scores
        
        Args:
            scores: Dict với keys: 'transformer', 'diffusion', 'cnn_vit', 'gcn'
                    Mỗi value là tensor (B,)
                    
        Returns:
            fused_score: (B,) tổng có trọng số
        """
        weights = self.weights
        
        fused = 0
        for i, name in enumerate(self.module_names):
            if name in scores:
                fused = fused + weights[i] * scores[name]
        
        return fused


# =====================================================================
# MLP FUSION
# =====================================================================

class MLPFusion(nn.Module):
    """
    Fusion bằng MLP - học cách kết hợp tối ưu
    
    Concatenate tất cả scores và features, qua MLP để ra final prediction.
    
    Args:
        score_dim: Số scores đầu vào
        feature_dim: Tổng kích thước features đầu vào
        hidden_dims: List các hidden dimensions
        num_classes: Số classes output (2 cho Real/Fake)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        score_dim=4,  # 4 anomaly scores
        feature_dim=None,  # Optional feature vectors
        hidden_dims=None,
        num_classes=2,
        dropout=FusionConfig.MLP_DROPOUT
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = FusionConfig.MLP_HIDDEN
        
        print(f"\n🏗️  Đang xây dựng MLP Fusion...")
        
        # Input dimension
        if feature_dim:
            input_dim = score_dim + feature_dim
        else:
            input_dim = score_dim
        
        # MLP layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        print(f"✅ MLP Fusion khởi tạo thành công!")
        print(f"📊 Input dim: {input_dim}, Output: {num_classes}")
    
    def forward(self, scores, features=None):
        """
        Forward pass
        
        Args:
            scores: Tensor (B, score_dim) hoặc dict các scores
            features: Optional tensor (B, feature_dim)
            
        Returns:
            logits: (B, num_classes)
        """
        # Convert dict to tensor if needed
        if isinstance(scores, dict):
            score_list = [scores.get(k, torch.zeros(1)) for k in 
                         ['transformer', 'diffusion', 'cnn_vit', 'gcn']]
            scores = torch.stack(score_list, dim=-1)
        
        # Concatenate with features if provided
        if features is not None:
            x = torch.cat([scores, features], dim=-1)
        else:
            x = scores
        
        return self.mlp(x)


# =====================================================================
# ATTENTION FUSION
# =====================================================================

class AttentionFusion(nn.Module):
    """
    Fusion bằng self-attention
    
    Mỗi score/feature được coi như một token, dùng attention
    để học cách weighted combination tối ưu.
    
    Args:
        num_sources: Số nguồn input (4 modules)
        embed_dim: Kích thước embedding cho mỗi source
        num_heads: Số attention heads
        num_classes: Số classes output
    """
    
    def __init__(
        self,
        num_sources=4,
        embed_dim=64,
        num_heads=4,
        num_classes=2
    ):
        super().__init__()
        
        print(f"\n🏗️  Đang xây dựng Attention Fusion...")
        
        self.num_sources = num_sources
        self.embed_dim = embed_dim
        
        # Project scores to embeddings
        # Mỗi source có projection riêng
        self.score_projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_sources)
        ])
        
        # Optional feature projections (CNN, GCN features)
        # Sẽ được thêm dynamically nếu có features
        self.feature_projections = nn.ModuleDict()
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        print(f"✅ Attention Fusion khởi tạo thành công!")
    
    def add_feature_projection(self, name, input_dim):
        """Thêm projection cho feature vector"""
        self.feature_projections[name] = nn.Linear(input_dim, self.embed_dim)
    
    def forward(self, scores, features=None):
        """
        Forward pass
        
        Args:
            scores: Dict với keys: 'transformer', 'diffusion', 'cnn_vit', 'gcn'
            features: Optional dict với feature vectors
            
        Returns:
            logits: (B, num_classes)
        """
        # Get batch size from first score
        first_key = list(scores.keys())[0]
        B = scores[first_key].shape[0]
        
        # Project scores to embeddings
        embeddings = []
        score_keys = ['transformer', 'diffusion', 'cnn_vit', 'gcn']
        
        for i, key in enumerate(score_keys):
            if key in scores:
                score = scores[key].unsqueeze(-1)  # (B, 1)
                emb = self.score_projections[i](score)  # (B, embed_dim)
                embeddings.append(emb.unsqueeze(1))  # (B, 1, embed_dim)
        
        # Add feature embeddings if provided
        if features:
            for name, feat in features.items():
                if name in self.feature_projections:
                    emb = self.feature_projections[name](feat)  # (B, embed_dim)
                    embeddings.append(emb.unsqueeze(1))  # (B, 1, embed_dim)
        
        # Concatenate all embeddings
        tokens = torch.cat(embeddings, dim=1)  # (B, num_tokens, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+num_tokens, embed_dim)
        
        # Self-attention
        attn_output, _ = self.attention(tokens, tokens, tokens)  # (B, 1+num_tokens, embed_dim)
        
        # Take class token output
        cls_output = attn_output[:, 0]  # (B, embed_dim)
        
        # Classify
        logits = self.classifier(cls_output)
        
        return logits


# =====================================================================
# COMPLETE FUSION MODULE
# =====================================================================

class DeepfakeFusionModule(nn.Module):
    """
    Complete Fusion Module cho hệ thống nhận diện deepfake
    
    Kết hợp:
    1. Anomaly scores từ các modules
    2. Feature vectors từ CNN và GCN
    3. Multiple fusion strategies
    
    Args:
        fusion_method: 'weighted_sum', 'mlp', hoặc 'attention'
        cnn_feature_dim: Kích thước CNN features
        gcn_feature_dim: Kích thước GCN features
        num_classes: Số classes
        threshold: Ngưỡng phân loại
    """
    
    def __init__(
        self,
        fusion_method=FusionConfig.FUSION_METHOD,
        cnn_feature_dim=CNNViTConfig.FEATURE_DIM,
        gcn_feature_dim=GCNConfig.OUTPUT_DIM,
        num_classes=2,
        threshold=FusionConfig.THRESHOLD
    ):
        super().__init__()
        
        print(f"\n🏗️  Đang xây dựng Deepfake Fusion Module...")
        print(f"   Fusion method: {fusion_method}")
        
        self.fusion_method = fusion_method
        self.threshold = threshold
        
        # Score normalizer
        self.score_normalizer = ScoreNormalizer(method='sigmoid')
        
        # Fusion module based on method
        if fusion_method == 'weighted_sum':
            self.fusion = WeightedSumFusion(learnable=True)
        elif fusion_method == 'mlp':
            total_feature_dim = cnn_feature_dim + gcn_feature_dim
            self.fusion = MLPFusion(
                score_dim=4,
                feature_dim=total_feature_dim,
                num_classes=num_classes
            )
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(
                num_sources=4,
                num_classes=num_classes
            )
            # Add feature projections
            self.fusion.add_feature_projection('cnn', cnn_feature_dim)
            self.fusion.add_feature_projection('gcn', gcn_feature_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        print(f"✅ Deepfake Fusion Module khởi tạo thành công!")
    
    def forward(self, scores, features=None):
        """
        Forward pass - kết hợp scores và features
        
        Args:
            scores: Dict các anomaly scores
                - 'transformer': Perplexity score (B,)
                - 'diffusion': Residual score (B,)
                - 'cnn_vit': CNN logit/prob (B,)
                - 'gcn': GCN score (B,)
            features: Optional dict các feature vectors
                - 'cnn': CNN features (B, cnn_dim)
                - 'gcn': GCN features (B, gcn_dim)
                
        Returns:
            dict: {
                'logits': Classification logits (B, 2),
                'probs': Probabilities (B, 2),
                'predictions': Predicted class (B,),
                'anomaly_score': Final anomaly score (B,)
            }
        """
        # Normalize scores
        normalized_scores = self.score_normalizer(scores)
        
        # Apply fusion
        if self.fusion_method == 'weighted_sum':
            # Weighted sum chỉ cho scores, không có classifier
            anomaly_score = self.fusion(normalized_scores)
            
            # Chuyển anomaly score thành logits (2 classes)
            # Score cao -> Fake, score thấp -> Real
            logits = torch.stack([
                -anomaly_score,  # Real score = negative anomaly
                anomaly_score    # Fake score = positive anomaly
            ], dim=-1)
        
        elif self.fusion_method == 'mlp':
            # Stack scores thành tensor
            score_tensor = torch.stack([
                normalized_scores.get('transformer', torch.zeros_like(next(iter(normalized_scores.values())))),
                normalized_scores.get('diffusion', torch.zeros_like(next(iter(normalized_scores.values())))),
                normalized_scores.get('cnn_vit', torch.zeros_like(next(iter(normalized_scores.values())))),
                normalized_scores.get('gcn', torch.zeros_like(next(iter(normalized_scores.values()))))
            ], dim=-1)
            
            # Concat features
            if features:
                # Lấy batch_size từ scores để tạo fallback tensors đúng kích thước
                batch_size = score_tensor.shape[0]
                device = score_tensor.device
                
                # Tạo fallback tensors với đúng kích thước
                cnn_fallback = torch.zeros(batch_size, CNNViTConfig.FEATURE_DIM, device=device)
                gcn_fallback = torch.zeros(batch_size, GCNConfig.OUTPUT_DIM, device=device)
                
                feat_concat = torch.cat([
                    features.get('cnn', cnn_fallback),
                    features.get('gcn', gcn_fallback)
                ], dim=-1)
            else:
                feat_concat = None
            
            logits = self.fusion(score_tensor, feat_concat)
            
            # Compute anomaly score từ logits
            probs = F.softmax(logits, dim=-1)
            anomaly_score = probs[:, 1]  # Prob of Fake class
        
        elif self.fusion_method == 'attention':
            logits = self.fusion(normalized_scores, features)
            probs = F.softmax(logits, dim=-1)
            anomaly_score = probs[:, 1]
        
        # Compute probabilities và predictions
        probs = F.softmax(logits, dim=-1)
        predictions = (probs[:, 1] > self.threshold).long()  # Fake if prob > threshold
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'anomaly_score': anomaly_score
        }
    
    def predict(self, scores, features=None):
        """
        Dự đoán class
        
        Args:
            scores: Dict các anomaly scores
            features: Optional features
            
        Returns:
            predictions: Tensor (B,) với 0=Real, 1=Fake
        """
        output = self.forward(scores, features)
        return output['predictions']
    
    def get_anomaly_score(self, scores, features=None):
        """
        Lấy anomaly score tổng hợp
        
        Args:
            scores: Dict các anomaly scores
            features: Optional features
            
        Returns:
            anomaly_score: Tensor (B,)
        """
        output = self.forward(scores, features)
        return output['anomaly_score']


# =====================================================================
# COMPLETE PIPELINE
# =====================================================================

class DeepfakeDetectionPipeline(nn.Module):
    """
    Complete pipeline cho deepfake detection
    
    Kết hợp tất cả các modules:
    1. VQ-VAE + Transformer (sequence modeling)
    2. DDPM (diffusion)
    3. CNN/ViT (feature extraction)
    4. GCN (structural analysis)
    5. Fusion (final decision)
    
    Args:
        vqvae: VQ-VAE model
        transformer: ImageGPT model
        ddpm: DDPM model
        cnn_vit: CNN/ViT classifier
        gcn: GCN module
        fusion: Fusion module
    """
    
    def __init__(
        self,
        vqvae=None,
        transformer=None,
        ddpm=None,
        cnn_vit=None,
        gcn=None,
        fusion=None
    ):
        super().__init__()
        
        print("\n🏗️  Đang xây dựng Deepfake Detection Pipeline...")
        
        # Store modules (có thể None nếu chưa train)
        self.vqvae = vqvae
        self.transformer = transformer
        self.ddpm = ddpm
        self.cnn_vit = cnn_vit
        self.gcn = gcn
        
        # Fusion module
        if fusion is None:
            self.fusion = DeepfakeFusionModule()
        else:
            self.fusion = fusion
        
        print("✅ Deepfake Detection Pipeline khởi tạo thành công!")
    
    def compute_all_scores(self, images, landmarks=None):
        """
        Tính tất cả các anomaly scores
        
        Args:
            images: Input images (B, 3, H, W)
            landmarks: Optional facial landmarks (B, N, 2)
            
        Returns:
            dict: Các anomaly scores và features
        """
        scores = {}
        features = {}
        
        # 1. VQ-VAE + Transformer score
        if self.vqvae is not None and self.transformer is not None:
            with torch.no_grad():
                _, token_indices = self.vqvae.encode(images)
                perplexity = self.transformer.compute_anomaly_score(token_indices)
                scores['transformer'] = perplexity
        
        # 2. DDPM anomaly score
        if self.ddpm is not None:
            with torch.no_grad():
                diffusion_score = self.ddpm.compute_anomaly_score(images)
                scores['diffusion'] = diffusion_score
        
        # 3. CNN/ViT score và features
        if self.cnn_vit is not None:
            logits, cnn_features = self.cnn_vit(images, return_features=True)
            probs = F.softmax(logits, dim=-1)
            scores['cnn_vit'] = probs[:, 1]  # Prob of Fake
            features['cnn'] = cnn_features
        
        # 4. GCN score và features
        if self.gcn is not None and landmarks is not None:
            gcn_features = self.gcn(landmarks)
            features['gcn'] = gcn_features
            # Simple anomaly score từ feature norm
            scores['gcn'] = torch.norm(gcn_features, dim=-1)
        
        return scores, features
    
    def forward(self, images, landmarks=None):
        """
        Complete forward pass
        
        Args:
            images: Input images (B, 3, H, W)
            landmarks: Optional facial landmarks (B, N, 2)
            
        Returns:
            dict: Final predictions và scores
        """
        # Compute all scores
        scores, features = self.compute_all_scores(images, landmarks)
        
        # Fusion để ra quyết định cuối
        output = self.fusion(scores, features)
        
        # Thêm individual scores vào output
        output['individual_scores'] = scores
        
        return output
    
    def predict(self, images, landmarks=None):
        """
        Dự đoán Real/Fake
        
        Args:
            images: Input images
            landmarks: Optional landmarks
            
        Returns:
            predictions: Tensor (B,) với 0=Real, 1=Fake
        """
        output = self.forward(images, landmarks)
        return output['predictions']


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test Fusion module khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST FUSION MODULE")
    print("=" * 70)
    
    batch_size = 4
    
    # Tạo dummy scores
    print("\n1. Tạo dummy scores...")
    scores = {
        'transformer': torch.randn(batch_size),
        'diffusion': torch.randn(batch_size) * 0.1,
        'cnn_vit': torch.sigmoid(torch.randn(batch_size)),
        'gcn': torch.randn(batch_size)
    }
    print(f"   Scores shapes: {[f'{k}: {v.shape}' for k, v in scores.items()]}")
    
    # Tạo dummy features
    features = {
        'cnn': torch.randn(batch_size, 512),
        'gcn': torch.randn(batch_size, 256)
    }
    print(f"   Features shapes: {[f'{k}: {v.shape}' for k, v in features.items()]}")
    
    # Test Weighted Sum Fusion
    print("\n2. Test Weighted Sum Fusion...")
    ws_fusion = WeightedSumFusion(learnable=True)
    ws_output = ws_fusion(scores)
    print(f"   Weighted sum output: {ws_output.shape}")
    print(f"   Weights: {ws_fusion.weights}")
    
    # Test MLP Fusion
    print("\n3. Test MLP Fusion...")
    mlp_fusion = MLPFusion(
        score_dim=4,
        feature_dim=512 + 256,
        num_classes=2
    )
    
    score_tensor = torch.stack([scores[k] for k in ['transformer', 'diffusion', 'cnn_vit', 'gcn']], dim=-1)
    feat_concat = torch.cat([features['cnn'], features['gcn']], dim=-1)
    mlp_output = mlp_fusion(score_tensor, feat_concat)
    print(f"   MLP output shape: {mlp_output.shape}")
    
    # Test Attention Fusion
    print("\n4. Test Attention Fusion...")
    attn_fusion = AttentionFusion(num_sources=4, num_classes=2)
    attn_fusion.add_feature_projection('cnn', 512)
    attn_fusion.add_feature_projection('gcn', 256)
    attn_output = attn_fusion(scores, features)
    print(f"   Attention output shape: {attn_output.shape}")
    
    # Test Complete Fusion Module
    print("\n5. Test Complete Fusion Module...")
    fusion_module = DeepfakeFusionModule(
        fusion_method='mlp',
        cnn_feature_dim=512,
        gcn_feature_dim=256
    )
    
    output = fusion_module(scores, features)
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Probs: {output['probs']}")
    print(f"   Predictions: {output['predictions']}")
    print(f"   Anomaly scores: {output['anomaly_score']}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST FUSION MODULE!")
    print("=" * 70)
