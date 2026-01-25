"""
Model factory for creating different deepfake detection models
Supports: ViXNet, Xception Only, ViT Only
"""

from venv import create
import torch
import torch.nn as nn
import timm
from config import Config
from model import create_vit_only, create_vixnet_cross_attention, create_xception_only, create_vixnet


def detect_model_type(checkpoint):
    """
    Detect model type from checkpoint state dict
    
    Args:
        checkpoint: Model checkpoint dictionary
        
    Returns:
        str: Model type ('vixnet', 'xception', 'vit', or 'unknown')
    """
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check for ViXNet-specific keys
    has_xception_branch = any('xception_branch' in key for key in state_dict.keys())
    has_vit_branch = any('vit_branch' in key for key in state_dict.keys())
    has_fusion = any('fusion' in key for key in state_dict.keys())
    has_cross_attention = any('fusion.x_proj' in key for key in state_dict.keys())

    if has_cross_attention and has_xception_branch and has_vit_branch and has_fusion:
        return 'vixnet_cross_attention'
    elif has_xception_branch and has_vit_branch and has_fusion:
        return 'vixnet'
    elif has_xception_branch or any('xception' in key for key in state_dict.keys()):
        return 'xception'
    elif has_vit_branch or any('vit' in key for key in state_dict.keys()):
        return 'vit'
    else:
        # Try to infer from layer structure
        if any('blocks' in key or 'patch_embed' in key for key in state_dict.keys()):
            return 'vit'
        elif any('conv' in key or 'bn' in key for key in state_dict.keys()):
            return 'xception'
        
    return 'unknown'


def create_model(model_type='vixnet', pretrained=True, num_classes=2):
    """
    Factory function to create different model architectures
    
    Args:
        model_type: Type of model ('vixnet', 'xception', 'vit')
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'vixnet_cross_attention':
        model = create_vixnet_cross_attention(pretrained=pretrained, num_classes=num_classes)
        arch_info = {
            'name': 'ViXNet Cross-Attention',
            'description': 'Vision Transformer + Xception Network with Cross-Attention Fusion',
            'xception_dim': Config.XCEPTION_DIM,
            'vit_dim': Config.VIT_DIM,
            'fusion_dim': Config.FUSION_DIM,
            'num_classes': num_classes
        }
    elif model_type == 'vixnet':
        model = create_vixnet(pretrained=pretrained, num_classes=num_classes)
        arch_info = {
            'name': 'ViXNet',
            'description': 'Vision Transformer + Xception Network',
            'xception_dim': Config.XCEPTION_DIM,
            'vit_dim': Config.VIT_DIM,
            'fusion_dim': Config.FUSION_DIM,
            'num_classes': num_classes
        }
    elif model_type == 'xception':
        model = create_xception_only(pretrained=pretrained, num_classes=num_classes)
        arch_info = {
            'name': 'Xception Only',
            'description': 'Xception CNN for spatial features',
            'feature_dim': 2048,
            'num_classes': num_classes
        }
    elif model_type == 'vit':
        model = create_vit_only(pretrained=pretrained, num_classes=num_classes)
        arch_info = {
            'name': 'ViT Only',
            'description': 'Vision Transformer for patch-wise attention',
            'feature_dim': 768,
            'num_classes': num_classes
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'vixnet', 'xception', or 'vit'")
    
    print(f"✅ Created {arch_info['name']} model")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, arch_info


def load_model_from_checkpoint(checkpoint_path, device='cpu', strict=False):
    """
    Load model from checkpoint and automatically detect model type
    Handles state_dict key mismatches by loading with strict=False
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce matching state dict keys
        
    Returns:
        tuple: (model, model_info)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Detect model type
    model_type = detect_model_type(checkpoint)
    
    if model_type == 'unknown':
        # Try ViXNet as default
        print("⚠️  Could not detect model type, trying ViXNet...")
        model_type = 'vixnet'
    
    # Create model with pretrained weights to match checkpoint structure
    num_classes = Config.NUM_CLASSES
    model, arch_info = create_model(model_type=model_type, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    # Extract state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try to load state dict with flexible matching
    try:
        # First try strict loading
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded state dict with strict matching")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {str(e)[:100]}...")
        print(f"   Attempting flexible loading with strict=False...")
        
        try:
            # Try non-strict loading
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            
            if incompatible_keys.missing_keys:
                print(f"⚠️  Missing keys ({len(incompatible_keys.missing_keys)}):")
                for key in incompatible_keys.missing_keys[:5]:
                    print(f"     - {key}")
                if len(incompatible_keys.missing_keys) > 5:
                    print(f"     ... and {len(incompatible_keys.missing_keys) - 5} more")
            
            if incompatible_keys.unexpected_keys:
                print(f"⚠️  Unexpected keys in checkpoint ({len(incompatible_keys.unexpected_keys)}):")
                for key in incompatible_keys.unexpected_keys[:5]:
                    print(f"     - {key}")
                if len(incompatible_keys.unexpected_keys) > 5:
                    print(f"     ... and {len(incompatible_keys.unexpected_keys) - 5} more")
            
            print(f"✅ Loaded state dict with flexible matching")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {str(e)[:200]}")
            raise RuntimeError(f"Could not load model from checkpoint: {str(e)}")
    
    model.eval()
    
    # Build model info
    model_info = {
        'loaded': True,
        'model_type': model_type,
        'checkpoint_path': checkpoint_path,
        'metrics': checkpoint.get('metrics', {}),
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'architecture': arch_info
    }
    
    print(f"✅ Loaded {arch_info['name']} from checkpoint")
    
    return model, model_info


if __name__ == "__main__":
    """Test model factory"""
    print("="*70)
    print("Testing Model Factory")
    print("="*70)
    
    # Test creating different models
    for model_type in ['vixnet', 'xception', 'vit']:
        print(f"\n📦 Creating {model_type} model...")
        model, arch_info = create_model(model_type=model_type, pretrained=False)
        print(f"   Architecture: {arch_info}")
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
