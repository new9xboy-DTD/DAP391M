"""
ViXNet: Vision Transformer with Xception Network for Deepfake Detection
Based on the paper published in Expert Systems with Applications (Q1)

Architecture:
- Branch 1: Xception CNN for global spatial features
- Branch 2: Vision Transformer (ViT) for patch-wise self-attention
- Feature Fusion: Combines both branches
- Classification Head: Binary classification (Real/Fake)
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from config import Config


class XceptionBranch(nn.Module):
    """
    Xception CNN branch for extracting global spatial features
    Uses pretrained Xception from ImageNet
    """
    
    def __init__(self, pretrained=True, feature_dim=2048, num_classes=0):
        super(XceptionBranch, self).__init__()
        
        # Load pretrained Xception from timm
        # Xception outputs 2048-dimensional features
        self.xception = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=num_classes,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            features: Global spatial features (batch_size, feature_dim)
        """
        features = self.xception(x)
        return features


class ViTBranch(nn.Module):
    """
    Vision Transformer branch for patch-wise self-attention
    Captures subtle artifacts in deepfakes
    """
    
    def __init__(self, pretrained=True, feature_dim=Config.VIT_DIM, model_name=Config.VIT_MODEL_NAME):
        super(ViTBranch, self).__init__()
        
        # Load pretrained ViT from timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='token'  # Use [CLS] token
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            features: Patch-wise attention features (batch_size, feature_dim)
        """
        features = self.vit(x)
        return features


class XceptionTokenBranch(nn.Module):
    """
    Xception branch that returns spatial tokens for cross-attention fusion
    """

    def __init__(self, pretrained=True, feature_dim=2048):
        super(XceptionTokenBranch, self).__init__()

        self.xception = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Keep spatial map for tokens
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            tokens: Spatial tokens (batch_size, num_tokens, feature_dim)
        """
        features = self.xception.forward_features(x)
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            tokens = features.flatten(2).transpose(1, 2)
        elif features.dim() == 3:
            tokens = features
        else:
            tokens = features.unsqueeze(1)
        return tokens


class ViTTokenBranch(nn.Module):
    """
    ViT branch that returns token sequence for cross-attention fusion
    """

    def __init__(self, pretrained=True, feature_dim=Config.VIT_DIM, model_name=Config.VIT_MODEL_NAME):
        super(ViTTokenBranch, self).__init__()

        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Keep full token sequence
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            tokens: Token sequence (batch_size, num_tokens, feature_dim)
        """
        features = self.vit.forward_features(x)
        if features.dim() == 3:
            tokens = features
        elif features.dim() == 2:
            tokens = features.unsqueeze(1)
        else:
            tokens = features
        return tokens


class FeatureFusion(nn.Module):
    """
    Feature fusion module to combine Xception and ViT features
    Uses concatenation followed by dimension reduction
    """
    
    def __init__(self, xception_dim=Config.XCEPTION_DIM, vit_dim=Config.VIT_DIM, fusion_dim=Config.FUSION_DIM):
        super(FeatureFusion, self).__init__()
        
        # Concatenated dimension
        concat_dim = xception_dim + vit_dim
        
        # Fusion layers with batch normalization and dropout
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def forward(self, xception_features, vit_features):
        """
        Args:
            xception_features: Features from Xception (batch_size, xception_dim)
            vit_features: Features from ViT (batch_size, vit_dim)
        Returns:
            fused_features: Fused features (batch_size, fusion_dim)
        """
        # Concatenate features from both branches
        concat_features = torch.cat([xception_features, vit_features], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(concat_features)
        
        return fused_features


class CrossAttentionFusion(nn.Module):
    """
    One-way cross-attention:
    Xception attends to ViT (X <- V)
    """

    def __init__(
        self,
        xception_dim=Config.XCEPTION_DIM,
        vit_dim=Config.VIT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_heads=2,
        dropout=0.1
    ):
        super(CrossAttentionFusion, self).__init__()

        self.x_proj = nn.Linear(xception_dim, fusion_dim) #tên này để xác định là cross attention, không thay đổi
        self.v_proj = nn.Linear(vit_dim, fusion_dim)

        self.pre_norm_x = nn.LayerNorm(fusion_dim)
        self.pre_norm_v = nn.LayerNorm(fusion_dim)

        self.attn_x_to_v = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # self.attn_v_to_x = nn.MultiheadAttention(
        #     embed_dim=fusion_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True
        # )

        self.norm_x = nn.LayerNorm(fusion_dim)
        self.norm_v = nn.LayerNorm(fusion_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_tokens, v_tokens):
        """
        Args:
            x_tokens: Xception tokens (batch_size, num_x_tokens, xception_dim)
            v_tokens: ViT tokens (batch_size, num_v_tokens, vit_dim)
        Returns:
            fused_features: Fused features (batch_size, fusion_dim)
        """
        x = self.pre_norm_x(self.x_proj(x_tokens))
        v = self.pre_norm_v(self.v_proj(v_tokens))

        # Compute attention in fp32 for stability under AMP
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_attn, _ = self.attn_x_to_v(query=x.float(), key=v.float(), value=v.float())
            # v_attn, _ = self.attn_v_to_x(query=v.float(), key=x.float(), value=x.float())

        x = self.norm_x(x + x_attn.to(x.dtype))
        # v = self.norm_v(v + v_attn.to(v.dtype))

        x_pooled = x.mean(dim=1)
        v_pooled = v.mean(dim=1)
        fused = torch.cat([x_pooled, v_pooled], dim=1)
        fused_features = self.fusion(fused)
        
        return fused_features


class ClassificationHead(nn.Module):
    """
    Classification head for binary classification (Real/Fake)
    """
    
    def __init__(self, fusion_dim=Config.FUSION_DIM, num_classes=Config.NUM_CLASSES, dropout=0.5):
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Fused features (batch_size, fusion_dim)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        logits = self.classifier(x)
        return logits


class ViXNet(nn.Module):
    """
    ViXNet: Complete model combining Xception and ViT with feature fusion
    
    Architecture:
    1. Xception branch (CNN) - Global spatial features
    2. ViT branch (Transformer) - Patch-wise self-attention
    3. Feature fusion - Combines both branches
    4. Classification head - Binary classification
    """
    
    def __init__(
        self,
        pretrained=True,
        xception_dim=Config.XCEPTION_DIM,
        vit_dim=Config.VIT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=2,
        vit_model_name=Config.VIT_MODEL_NAME
    ):
        super(ViXNet, self).__init__()
        
        print("🏗️  Initializing ViXNet model...")
        
        # Branch 1: Xception CNN
        print("   📌 Loading Xception branch...")
        self.xception_branch = XceptionBranch(
            pretrained=pretrained,
            feature_dim=xception_dim
        )
        
        # Branch 2: Vision Transformer
        print("   📌 Loading ViT branch...")
        self.vit_branch = ViTBranch(
            pretrained=pretrained,
            feature_dim=vit_dim,
            model_name=vit_model_name
        )
        
        # Feature fusion module
        print("   📌 Initializing feature fusion...")
        self.fusion = FeatureFusion(
            xception_dim=xception_dim,
            vit_dim=vit_dim,
            fusion_dim=fusion_dim
        )
        
        # Classification head
        print("   📌 Initializing classification head...")
        self.classifier = ClassificationHead(
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )
        
        # Store dimensions for reference
        self.xception_dim = xception_dim
        self.vit_dim = vit_dim
        self.fusion_dim = fusion_dim
        
        print(f"✅ ViXNet initialized successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def forward(self, x):
        """
        Forward pass through ViXNet
        
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Extract features from both branches
        xception_features = self.xception_branch(x)
        vit_features = self.vit_branch(x)
        
        # Fuse features
        fused_features = self.fusion(xception_features, vit_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def freeze_feature_extractors(self):
        """
        Freeze Xception and ViT branches for Stage 1 training
        Only fusion and classification layers remain trainable
        """
        print("🔒 Freezing feature extractors (Stage 1)...")
        
        # Freeze Xception
        for param in self.xception_branch.parameters():
            param.requires_grad = False
            
        # Freeze ViT
        for param in self.vit_branch.parameters():
            param.requires_grad = False
            
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_xception_layers(self, unfreeze_ratio=0.15):
        """
        Unfreeze Xception layers for Stage 2 fine-tuning (3-stage training)
        - Xception: last `unfreeze_ratio` layers
        - ViT: remains frozen
        """
        print("🔓 Unfreezing Xception layers (Stage 2)...")

        xception_modules = list(self.xception_branch.xception.named_children())
        num_to_unfreeze = max(1, int(unfreeze_ratio * len(xception_modules)))
        
        for name, module in xception_modules[-num_to_unfreeze:]:
            for param in module.parameters():
                param.requires_grad = True
        
        print(f"   Unfrozen {num_to_unfreeze} Xception modules")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_vit_layers(self, num_blocks=2):
        """
        Unfreeze ViT layers for Stage 3 fine-tuning (3-stage training)
        - ViT: last `num_blocks` transformer encoder blocks
        """
        print("🔓 Unfreezing ViT layers (Stage 3)...")

        if hasattr(self.vit_branch.vit, 'blocks'):
            vit_blocks = self.vit_branch.vit.blocks
            num_blocks_to_unfreeze = min(num_blocks, len(vit_blocks))
            for block in vit_blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"   Unfrozen last {num_blocks_to_unfreeze} ViT blocks")
        else:
            print("   Warning: ViT blocks not found")

        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def unfreeze_high_level_layers(self):
        """
        Unfreeze high-level layers for Stage 2 fine-tuning (legacy 2-stage training)
        - Xception: 15% last layers
        - ViT: last 2 transformer encoder blocks
        """
        print("🔓 Unfreezing high-level layers (Stage 2)...")
        
        # Unfreeze last blocks of Xception
        # Xception has blocks numbered, we'll unfreeze the last few
        xception_modules = list(self.xception_branch.xception.named_children())
        
        # Typically unfreeze the last 15% layers for fine-tuning
        num_to_unfreeze = int(0.15 * len(xception_modules))
        
        for name, module in xception_modules[-num_to_unfreeze:]:
            for param in module.parameters():
                param.requires_grad = True
                
        # Unfreeze last transformer blocks of ViT
        if hasattr(self.vit_branch.vit, 'blocks'):
            vit_blocks = self.vit_branch.vit.blocks
            # Unfreeze last 2 blocks
            num_blocks_to_unfreeze = min(2, len(vit_blocks))
            for block in vit_blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_param_groups(self, lr_head=1e-4, lr_cnn=1e-5, lr_vit=1e-6):
        """
        Get parameter groups with different learning rates for 3-stage training
        
        Args:
            lr_head: Learning rate for fusion + classifier head
            lr_cnn: Learning rate for Xception CNN
            lr_vit: Learning rate for ViT transformer
            
        Returns:
            List of parameter groups for optimizer
        """
        # Head parameters (fusion + classifier)
        head_params = list(self.fusion.parameters()) + list(self.classifier.parameters())
        
        # CNN parameters (Xception)
        cnn_params = [p for p in self.xception_branch.parameters() if p.requires_grad]
        
        # ViT parameters
        vit_params = [p for p in self.vit_branch.parameters() if p.requires_grad]
        
        param_groups = []
        
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head, 'name': 'head'})
        if cnn_params:
            param_groups.append({'params': cnn_params, 'lr': lr_cnn, 'name': 'cnn'})
        if vit_params:
            param_groups.append({'params': vit_params, 'lr': lr_vit, 'name': 'vit'})
        
        print(f"   Parameter groups:")
        for pg in param_groups:
            num_params = sum(p.numel() for p in pg['params'])
            print(f"     - {pg['name']}: {num_params:,} params, lr={pg['lr']:.2e}")
        
        return param_groups


class ViXNetCrossAttention(nn.Module):
    """
    ViXNet variant using cross-attention fusion between Xception and ViT tokens
    """

    def __init__(
        self,
        pretrained=True,
        xception_dim=Config.XCEPTION_DIM,
        vit_dim=Config.VIT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=2,
        vit_model_name=Config.VIT_MODEL_NAME,
        num_heads=2,
        dropout=0.1
    ):
        super(ViXNetCrossAttention, self).__init__()

        print("🏗️  Initializing ViXNet (Cross-Attention Fusion) model...")

        self.xception_branch = XceptionTokenBranch(
            pretrained=pretrained,
            feature_dim=xception_dim
        )

        self.vit_branch = ViTTokenBranch(
            pretrained=pretrained,
            feature_dim=vit_dim,
            model_name=vit_model_name
        )

        self.fusion = CrossAttentionFusion(
            xception_dim=xception_dim,
            vit_dim=vit_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.classifier = ClassificationHead(
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )

        self.xception_dim = xception_dim
        self.vit_dim = vit_dim
        self.fusion_dim = fusion_dim

        print("✅ ViXNet Cross-Attention initialized successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, x):
        """
        Forward pass through ViXNetCrossAttention

        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        x_tokens = self.xception_branch(x)
        v_tokens = self.vit_branch(x)

        fused_features = self.fusion(x_tokens, v_tokens)
        logits = self.classifier(fused_features)
        return logits

    def freeze_feature_extractors(self):
        """
        Freeze Xception and ViT branches for Stage 1 training
        Only fusion and classification layers remain trainable
        """
        print("🔒 Freezing feature extractors (Stage 1)...")

        for param in self.xception_branch.parameters():
            param.requires_grad = False

        for param in self.vit_branch.parameters():
            param.requires_grad = False

        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_xception_layers(self, unfreeze_ratio=0.15):
        """
        Unfreeze Xception layers for Stage 2 fine-tuning (3-stage training)
        - Xception: last `unfreeze_ratio` layers
        - ViT: remains frozen
        """
        print("🔓 Unfreezing Xception layers (Stage 2)...")

        xception_modules = list(self.xception_branch.xception.named_children())
        num_to_unfreeze = max(1, int(unfreeze_ratio * len(xception_modules)))
        
        for name, module in xception_modules[-num_to_unfreeze:]:
            for param in module.parameters():
                param.requires_grad = True
        
        print(f"   Unfrozen {num_to_unfreeze} Xception modules")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_vit_layers(self, num_blocks=2):
        """
        Unfreeze ViT layers for Stage 3 fine-tuning (3-stage training)
        - ViT: last `num_blocks` transformer encoder blocks
        """
        print("🔓 Unfreezing ViT layers (Stage 3)...")

        if hasattr(self.vit_branch.vit, 'blocks'):
            vit_blocks = self.vit_branch.vit.blocks
            num_blocks_to_unfreeze = min(num_blocks, len(vit_blocks))
            for block in vit_blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"   Unfrozen last {num_blocks_to_unfreeze} ViT blocks")
        else:
            print("   Warning: ViT blocks not found")

        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def unfreeze_high_level_layers(self):
        """
        Unfreeze high-level layers for Stage 2 fine-tuning (legacy 2-stage training)
        - Xception: last 15% layers
        - ViT: last 2 transformer encoder blocks
        """
        print("🔓 Unfreezing high-level layers (Stage 2)...")

        xception_modules = list(self.xception_branch.xception.named_children())
        num_to_unfreeze = int(0.15 * len(xception_modules))
        for name, module in xception_modules[-num_to_unfreeze:]:
            for param in module.parameters():
                param.requires_grad = True

        if hasattr(self.vit_branch.vit, 'blocks'):
            vit_blocks = self.vit_branch.vit.blocks
            num_blocks_to_unfreeze = min(2, len(vit_blocks))
            for block in vit_blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer

        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_param_groups(self, lr_head=1e-4, lr_cnn=1e-5, lr_vit=1e-6):
        """
        Get parameter groups with different learning rates for 3-stage training
        
        Args:
            lr_head: Learning rate for fusion + classifier head
            lr_cnn: Learning rate for Xception CNN
            lr_vit: Learning rate for ViT transformer
            
        Returns:
            List of parameter groups for optimizer
        """
        # Head parameters (fusion + classifier)
        head_params = list(self.fusion.parameters()) + list(self.classifier.parameters())
        
        # CNN parameters (Xception)
        cnn_params = [p for p in self.xception_branch.parameters() if p.requires_grad]
        
        # ViT parameters
        vit_params = [p for p in self.vit_branch.parameters() if p.requires_grad]
        
        param_groups = []
        
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head, 'name': 'head'})
        if cnn_params:
            param_groups.append({'params': cnn_params, 'lr': lr_cnn, 'name': 'cnn'})
        if vit_params:
            param_groups.append({'params': vit_params, 'lr': lr_vit, 'name': 'vit'})
        
        print(f"   Parameter groups:")
        for pg in param_groups:
            num_params = sum(p.numel() for p in pg['params'])
            print(f"     - {pg['name']}: {num_params:,} params, lr={pg['lr']:.2e}")
        
        return param_groups
    
class XceptionOnly(nn.Module):
    """
    Xception-only model for ablation studies
    """
    
    def __init__(self, pretrained=True, num_classes=2):
        super(XceptionOnly, self).__init__()
        
        print("🏗️  Initializing Xception-only model...")
        
        # Xception branch
        self.xception_branch = XceptionBranch(
            pretrained=pretrained,
            feature_dim=Config.XCEPTION_DIM
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            fusion_dim=Config.XCEPTION_DIM,
            num_classes=num_classes
        )
        
        print(f"✅ Xception-only model initialized successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, x):
        """
        Forward pass through XceptionOnly
        
        :param x: Input Images (batch_size, 3, H, W)
        :return logits: Classification logits (batch size, num_classes)
        """
        #Extract features from Xception
        xception_features = self.xception_branch(x)
        
        #Classify
        logits = self.classifier(xception_features)
        
        return logits
    
    def freeze_feature_extractors(self):
        """
        Freeze Xception for stage 1 training
        Only classification head remain trainable
        """
        print("Freezing feature extractors (Stage 1)...")
        
        #Freezing Xception
        for param in self.xception_branch.parameters():
            param.requires_grad = False
    
    def unfreeze_high_level_layers(self):
        """
        Unfreeze Xception for stage 2 training
        
        Args:
            num_layers_to_unfreeze: Number of last layers to unfreeze (default: 30)
        """
        layers = list(self.xception_branch.named_children())
        num_layers_to_unfreeze = int(0.2 * len(layers))
        
        for name, layer in layers[-num_layers_to_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
                
    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
class ViTOnly(nn.Module):
    """
    ViT-only model for ablation studies
    """
    
    def __init__(self, pretrained=True, num_classes=2, model_name='vit_base_patch16_224'):
        super(ViTOnly, self).__init__()
        
        print("🏗️  Initializing ViT-only model...")
        
        # ViT branch
        self.vit_branch = ViTBranch(
            pretrained=pretrained,
            feature_dim=Config.VIT_DIM,
            model_name=model_name
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            fusion_dim=Config.VIT_DIM,
            num_classes=num_classes
        )
        
        print(f"✅ ViT-only model initialized successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def forward(self, x):
        """
        Forward pass through ViTOnly
        
        Args:
            x: Input images (batch_size, 3, H, W)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Extract features from ViT
        vit_features = self.vit_branch(x)
        
        # Classify
        logits = self.classifier(vit_features)
        
        return logits
    
    def freeze_feature_extractors(self):
        """
        Freeze ViT for Stage 1 training
        Only classification head remain trainable
        """
        print("🔒 Freezing feature extractors (Stage 1)...")
        
        # Freeze ViT
        for param in self.vit_branch.parameters():
            param.requires_grad = False
            
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def unfreeze_high_level_layers(self):
        """
        Unfreeze high-level layers for Stage 2 fine-tuning
        - ViT: last 3 transformer encoder blocks
        """
        print("🔓 Unfreezing high-level layers (Stage 2)...")
        
        # Unfreeze last transformer blocks of ViT
        if hasattr(self.vit_branch.vit, 'blocks'):
            vit_blocks = self.vit_branch.vit.blocks
            # Unfreeze last 3 blocks
            num_blocks_to_unfreeze = min(len(vit_blocks), 3)
            for block in vit_blocks[-num_blocks_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
                    
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
def create_vit_only(pretrained=True, num_classes=2, model_name='vit_base_patch16_224'):
    """
    Factory function to create ViT-only model
    
    Args:
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (default: 2)
        model_name: Name of the ViT model architecture
        
    Returns:
        ViT-only model
    """
    model = ViTOnly(
        pretrained=pretrained,
        num_classes=num_classes,
        model_name=model_name
    )
    return model
    
def create_xception_only(pretrained=True, num_classes=2):
    """
    Factory function to create Xception model
    
    Args:
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (default: 2)
    
    Returns:
        Xception model
    """
    model = XceptionOnly(
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def create_vixnet(pretrained=True, num_classes=2):
    """
    Factory function to create ViXNet model
    
    Args:
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (default: 2 for binary classification)
        
    Returns:
        ViXNet model
    """
    model = ViXNet(
        pretrained=pretrained,
        xception_dim=Config.XCEPTION_DIM,
        vit_dim=Config.VIT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=num_classes,
        vit_model_name=Config.VIT_MODEL_NAME
    )
    return model


def create_vixnet_cross_attention(
    pretrained=True,
    num_classes=2,
    num_heads=8,
    dropout=0.1
):
    """
    Factory function to create ViXNet with cross-attention fusion

    Args:
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (default: 2 for binary classification)
        num_heads: Number of attention heads
        dropout: Dropout rate inside cross-attention fusion

    Returns:
        ViXNetCrossAttention model
    """
    model = ViXNetCrossAttention(
        pretrained=pretrained,
        xception_dim=Config.XCEPTION_DIM,
        vit_dim=Config.VIT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=num_classes,
        vit_model_name=Config.VIT_MODEL_NAME,
        num_heads=num_heads,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    """
    Test model creation and forward pass
    """
    print("="*70)
    print("Testing ViXNet Model")
    print("="*70)
    
    # Create model
    model = create_vixnet(pretrained=False)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n🧪 Testing forward pass with input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"✅ Output shape: {output.shape}")
    
    # Test freezing
    print("\n" + "="*70)
    model.freeze_feature_extractors()
    
    print("\n" + "="*70)
    model.unfreeze_high_level_layers()
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
