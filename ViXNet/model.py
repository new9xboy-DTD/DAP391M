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
        
    def unfreeze_high_level_layers(self):
        """
        Unfreeze high-level layers for Stage 2 fine-tuning
        - Xception: last 2-3 blocks
        - ViT: last 1-2 transformer encoder blocks
        """
        print("🔓 Unfreezing high-level layers (Stage 2)...")
        
        # Unfreeze last blocks of Xception
        # Xception has blocks numbered, we'll unfreeze the last few
        xception_modules = list(self.xception_branch.xception.named_children())
        
        # Typically unfreeze the last 20% of layers for fine-tuning
        num_to_unfreeze = max(1, len(xception_modules) // 5)
        
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
    
    def unfreeze_last_layers(self, num_layers_to_unfreeze=30):
        """
        Unfreeze Xception for stage 2 training
        
        Args:
            num_layers_to_unfreeze: Number of last layers to unfreeze (default: 30)
        """
        layers = list(self.xception_branch.children())
        
        for layer in layers[-num_layers_to_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
                
    def get_trainable_params(self):
        """
        Get list of trainable parameters for optimizer
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
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
