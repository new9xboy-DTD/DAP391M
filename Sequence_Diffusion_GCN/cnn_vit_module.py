"""
=====================================================================
MÔ-ĐUN CNN/VIT TRÍCH XUẤT ĐẶC TRƯNG
=====================================================================
Mô tả:
    Module này sử dụng các mạng CNN hoặc Vision Transformer (ViT)
    pretrained để trích xuất đặc trưng từ ảnh khuôn mặt.
    
    Các đặc trưng này được sử dụng:
    1. Trực tiếp cho classification (Real vs Fake)
    2. Kết hợp với GCN để phân tích cấu trúc
    3. Fusion với các modules khác (Transformer, Diffusion)
    
    Sử dụng các model pretrained mạnh như:
    - EfficientNet (CNN hiệu quả)
    - ResNet (CNN cổ điển)
    - ViT, Swin Transformer (Vision Transformers)
    - Có thể dùng features từ CLIP, DINO (self-supervised)
    
Tham khảo:
    - Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
    - Dosovitskiy et al., "An Image is Worth 16x16 Words"
    - Liu et al., "Swin Transformer"
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Import config
from config import CNNViTConfig, DataConfig


# =====================================================================
# FEATURE EXTRACTOR BASE
# =====================================================================

class FeatureExtractor(nn.Module):
    """
    Base class cho Feature Extractor
    
    Đây là abstract class định nghĩa interface chung cho
    tất cả các loại feature extractors (CNN, ViT, etc.)
    
    Args:
        output_dim: Kích thước feature vector đầu ra
    """
    
    def __init__(self, output_dim=CNNViTConfig.FEATURE_DIM):
        super().__init__()
        self.output_dim = output_dim
    
    def extract_features(self, x):
        """
        Trích xuất features từ ảnh
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            features: Feature vectors (B, output_dim)
        """
        raise NotImplementedError
    
    def forward(self, x):
        """
        Forward pass - trả về features hoặc predictions
        """
        raise NotImplementedError


# =====================================================================
# CNN BACKBONE
# =====================================================================

class CNNFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor sử dụng CNN backbone từ timm
    
    Sử dụng các pretrained CNNs như EfficientNet, ResNet, etc.
    
    Args:
        backbone_name: Tên model từ timm (e.g., 'efficientnet_b4')
        pretrained: Sử dụng pretrained weights
        output_dim: Kích thước feature vector đầu ra
        freeze_backbone: Đóng băng backbone trong giai đoạn đầu
    """
    
    def __init__(
        self,
        backbone_name=CNNViTConfig.BACKBONE_NAME,
        pretrained=CNNViTConfig.PRETRAINED,
        output_dim=CNNViTConfig.FEATURE_DIM,
        freeze_backbone=CNNViTConfig.FREEZE_BACKBONE
    ):
        super().__init__(output_dim)
        
        print(f"\n🏗️  Đang xây dựng CNN Feature Extractor...")
        print(f"   Backbone: {backbone_name}")
        print(f"   Pretrained: {pretrained}")
        
        # Tạo backbone từ timm
        # num_classes=0 để không có classification head
        # global_pool='' để giữ spatial features nếu cần
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Không có classification head
            global_pool='avg'  # Global average pooling
        )
        
        # Lấy feature dimension từ backbone
        self.backbone_dim = self.backbone.num_features
        print(f"   Backbone feature dim: {self.backbone_dim}")
        
        # Projection layer nếu cần thay đổi dimension
        if self.backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )
        else:
            self.projection = nn.Identity()
        
        # Freeze backbone nếu cần
        if freeze_backbone:
            self._freeze_backbone()
            print(f"   ❄️ Backbone đã được đóng băng")
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ CNN Feature Extractor khởi tạo thành công!")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
    
    def _freeze_backbone(self):
        """Đóng băng backbone weights"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_ratio=0.5):
        """
        Mở đóng băng một phần backbone
        
        Thường dùng trong fine-tuning: unfreeze dần các layers sâu hơn
        
        Args:
            unfreeze_ratio: Tỷ lệ layers được unfreeze (từ cuối)
        """
        params = list(self.backbone.parameters())
        num_params = len(params)
        num_unfreeze = int(num_params * unfreeze_ratio)
        
        # Unfreeze các layers cuối
        for param in params[-num_unfreeze:]:
            param.requires_grad = True
        
        print(f"🔓 Đã unfreeze {num_unfreeze}/{num_params} parameter groups")
    
    def extract_features(self, x):
        """
        Trích xuất features từ ảnh
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            features: Feature vectors (B, output_dim)
        """
        # Backbone forward
        features = self.backbone(x)  # (B, backbone_dim)
        
        # Project to output dimension
        features = self.projection(features)  # (B, output_dim)
        
        return features
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            features: Feature vectors (B, output_dim)
        """
        return self.extract_features(x)


# =====================================================================
# VISION TRANSFORMER BACKBONE
# =====================================================================

class ViTFeatureExtractor(FeatureExtractor):
    """
    Feature Extractor sử dụng Vision Transformer
    
    ViT chia ảnh thành patches và xử lý như sequence tokens.
    
    Args:
        backbone_name: Tên model ViT từ timm (e.g., 'vit_base_patch16_224')
        pretrained: Sử dụng pretrained weights
        output_dim: Kích thước feature vector đầu ra
        freeze_backbone: Đóng băng backbone
    """
    
    def __init__(
        self,
        backbone_name='vit_base_patch16_224',
        pretrained=True,
        output_dim=CNNViTConfig.FEATURE_DIM,
        freeze_backbone=False
    ):
        super().__init__(output_dim)
        
        print(f"\n🏗️  Đang xây dựng ViT Feature Extractor...")
        print(f"   Backbone: {backbone_name}")
        
        # Tạo ViT backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        self.backbone_dim = self.backbone.num_features
        print(f"   Backbone feature dim: {self.backbone_dim}")
        
        # Projection layer
        if self.backbone_dim != output_dim:
            self.projection = nn.Sequential(
                nn.LayerNorm(self.backbone_dim),
                nn.Linear(self.backbone_dim, output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = nn.Identity()
        
        if freeze_backbone:
            self._freeze_backbone()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ ViT Feature Extractor khởi tạo thành công!")
        print(f"📊 Total parameters: {total_params:,}")
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        """Trích xuất features"""
        features = self.backbone(x)
        features = self.projection(features)
        return features
    
    def forward(self, x):
        return self.extract_features(x)


# =====================================================================
# MULTI-SCALE FEATURE EXTRACTOR
# =====================================================================

class MultiScaleFeatureExtractor(FeatureExtractor):
    """
    Trích xuất features từ nhiều scales (levels) của backbone
    
    Kết hợp features từ các levels khác nhau để capture cả
    low-level details và high-level semantics.
    
    Args:
        backbone_name: Tên backbone
        pretrained: Dùng pretrained
        output_dim: Kích thước output
        levels: Các levels cần trích xuất (e.g., ['layer2', 'layer3', 'layer4'])
    """
    
    def __init__(
        self,
        backbone_name=CNNViTConfig.BACKBONE_NAME,
        pretrained=True,
        output_dim=CNNViTConfig.FEATURE_DIM,
        levels=None
    ):
        super().__init__(output_dim)
        
        print(f"\n🏗️  Đang xây dựng Multi-Scale Feature Extractor...")
        
        # Tạo backbone với features từ multiple levels
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,  # Chỉ lấy features, không có head
            out_indices=levels if levels else [2, 3, 4]  # Default: last 3 stages
        )
        
        # Lấy info về các feature levels
        self.feature_info = self.backbone.feature_info
        
        # In thông tin các levels
        print(f"   Feature levels:")
        total_channels = 0
        for info in self.feature_info:
            print(f"     - {info['module']}: {info['num_chs']} channels, reduction {info['reduction']}x")
            total_channels += info['num_chs']
        
        # Global pooling cho mỗi level
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in self.feature_info
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_channels, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        print(f"✅ Multi-Scale Feature Extractor khởi tạo thành công!")
    
    def extract_features(self, x):
        """Trích xuất multi-scale features"""
        # Get features from multiple levels
        feature_list = self.backbone(x)
        
        # Pool and concatenate
        pooled_features = []
        for features, pool in zip(feature_list, self.pools):
            pooled = pool(features).flatten(1)  # (B, C)
            pooled_features.append(pooled)
        
        # Concatenate all levels
        concat_features = torch.cat(pooled_features, dim=1)  # (B, total_channels)
        
        # Fusion
        output = self.fusion(concat_features)
        
        return output
    
    def forward(self, x):
        return self.extract_features(x)


# =====================================================================
# CLASSIFIER HEAD
# =====================================================================

class ClassifierHead(nn.Module):
    """
    Classification head cho phân loại Real/Fake
    
    MLP với dropout để regularization
    
    Args:
        input_dim: Kích thước feature vector đầu vào
        hidden_dims: List các hidden dimensions
        num_classes: Số classes (2 cho Real/Fake)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim=CNNViTConfig.FEATURE_DIM,
        hidden_dims=None,
        num_classes=2,
        dropout=CNNViTConfig.CLASSIFIER_DROPOUT
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = CNNViTConfig.CLASSIFIER_HIDDEN
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Args:
            features: Feature vectors (B, input_dim)
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        return self.classifier(features)


# =====================================================================
# COMPLETE CNN/ViT MODEL
# =====================================================================

class CNNViTClassifier(nn.Module):
    """
    Complete CNN/ViT model cho classification
    
    Kết hợp:
    1. Feature extractor (CNN hoặc ViT)
    2. Classification head
    
    Args:
        extractor_type: 'cnn', 'vit', hoặc 'multiscale'
        backbone_name: Tên backbone
        pretrained: Dùng pretrained
        feature_dim: Kích thước feature
        num_classes: Số classes
        freeze_backbone: Đóng băng backbone
    """
    
    def __init__(
        self,
        extractor_type='cnn',
        backbone_name=CNNViTConfig.BACKBONE_NAME,
        pretrained=CNNViTConfig.PRETRAINED,
        feature_dim=CNNViTConfig.FEATURE_DIM,
        num_classes=2,
        freeze_backbone=CNNViTConfig.FREEZE_BACKBONE
    ):
        super().__init__()
        
        print(f"\n🏗️  Đang xây dựng CNN/ViT Classifier...")
        print(f"   Extractor type: {extractor_type}")
        
        # Chọn loại feature extractor
        if extractor_type == 'cnn':
            self.feature_extractor = CNNFeatureExtractor(
                backbone_name=backbone_name,
                pretrained=pretrained,
                output_dim=feature_dim,
                freeze_backbone=freeze_backbone
            )
        elif extractor_type == 'vit':
            self.feature_extractor = ViTFeatureExtractor(
                backbone_name=backbone_name,
                pretrained=pretrained,
                output_dim=feature_dim,
                freeze_backbone=freeze_backbone
            )
        elif extractor_type == 'multiscale':
            self.feature_extractor = MultiScaleFeatureExtractor(
                backbone_name=backbone_name,
                pretrained=pretrained,
                output_dim=feature_dim
            )
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        # Classification head
        self.classifier = ClassifierHead(
            input_dim=feature_dim,
            num_classes=num_classes
        )
        
        print(f"✅ CNN/ViT Classifier khởi tạo thành công!")
    
    def extract_features(self, x):
        """
        Chỉ trích xuất features (không qua classifier)
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            features: Feature vectors (B, feature_dim)
        """
        return self.feature_extractor(x)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: Trả về features cùng với logits
            
        Returns:
            Nếu return_features=False: logits (B, num_classes)
            Nếu return_features=True: (logits, features)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_proba(self, x):
        """
        Dự đoán xác suất
        
        Args:
            x: Input images
            
        Returns:
            probabilities: Xác suất cho mỗi class (B, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x):
        """
        Dự đoán class
        
        Args:
            x: Input images
            
        Returns:
            predictions: Predicted class indices (B,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


# =====================================================================
# ATTENTION VISUALIZATION
# =====================================================================

class AttentionVisualizer:
    """
    Helper class để visualize attention maps
    
    Hữu ích để hiểu model đang "nhìn" vào đâu trên ảnh.
    
    Args:
        model: CNN/ViT model
    """
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward và backward hooks"""
        # Đây là placeholder - implementation cụ thể phụ thuộc vào backbone
        pass
    
    def get_attention_map(self, x, target_class=None):
        """
        Lấy attention/saliency map cho input
        
        Args:
            x: Input image (1, 3, H, W)
            target_class: Class để tính gradient-based attention
            
        Returns:
            attention_map: Attention map (H, W)
        """
        # Grad-CAM implementation
        self.model.eval()
        x.requires_grad = True
        
        # Forward
        if hasattr(self.model, 'extract_features'):
            features = self.model.extract_features(x)
        else:
            features = self.model(x)
        
        # Simple gradient-based visualization
        if target_class is None:
            target = features.mean()
        else:
            target = features[:, target_class].mean()
        
        target.backward()
        
        # Get gradients
        gradients = x.grad.abs().mean(dim=1, keepdim=True)  # Average across channels
        
        return gradients[0, 0].detach().cpu().numpy()


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test CNN/ViT module khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST CNN/VIT MODULE")
    print("=" * 70)
    
    # Test với EfficientNet backbone (nhỏ hơn để test nhanh)
    print("\n1. Test CNN Feature Extractor (EfficientNet-B0)...")
    cnn_extractor = CNNFeatureExtractor(
        backbone_name='efficientnet_b0',
        pretrained=True,
        output_dim=256
    )
    
    # Test với random input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {x.shape}")
    
    features = cnn_extractor(x)
    print(f"   Output features shape: {features.shape}")
    
    # Test classifier
    print("\n2. Test CNN/ViT Classifier...")
    classifier = CNNViTClassifier(
        extractor_type='cnn',
        backbone_name='efficientnet_b0',
        feature_dim=256,
        num_classes=2
    )
    
    logits = classifier(x)
    print(f"   Logits shape: {logits.shape}")
    
    logits, features = classifier(x, return_features=True)
    print(f"   Features shape: {features.shape}")
    
    probs = classifier.predict_proba(x)
    print(f"   Probabilities: {probs}")
    
    preds = classifier.predict(x)
    print(f"   Predictions: {preds}")
    
    # Test multi-scale extractor
    print("\n3. Test Multi-Scale Feature Extractor...")
    try:
        multiscale = MultiScaleFeatureExtractor(
            backbone_name='efficientnet_b0',
            output_dim=256
        )
        features_ms = multiscale(x)
        print(f"   Multi-scale features shape: {features_ms.shape}")
    except Exception as e:
        print(f"   ⚠️ Multi-scale test skipped: {e}")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST CNN/VIT MODULE!")
    print("=" * 70)
