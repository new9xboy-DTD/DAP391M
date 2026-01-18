"""
Test script to verify ViXNet implementation
Runs all components to ensure they work correctly
"""

import sys
import torch


def test_imports():
    """Test all imports work correctly"""
    print("\n" + "="*70)
    print("🔍 Testing Imports")
    print("="*70)
    
    try:
        from model import create_vixnet, ViXNet
        print("✅ model.py imports successful")
    except Exception as e:
        print(f"❌ model.py import failed: {str(e)}")
        return False
    
    try:
        from config import Config
        print("✅ config.py imports successful")
    except Exception as e:
        print(f"❌ config.py import failed: {str(e)}")
        return False
    
    try:
        from dataset import create_data_loaders, check_dataset_availability
        print("✅ dataset.py imports successful")
    except Exception as e:
        print(f"❌ dataset.py import failed: {str(e)}")
        return False
    
    try:
        from utils import train_one_epoch, validate, save_checkpoint
        print("✅ utils.py imports successful")
    except Exception as e:
        print(f"❌ utils.py import failed: {str(e)}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and initialization"""
    print("\n" + "="*70)
    print("🏗️  Testing Model Creation")
    print("="*70)
    
    try:
        from model import create_vixnet
        model = create_vixnet(pretrained=False, num_classes=2)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Verify parameter count is reasonable for ViT-Tiny model
        # ViT-Tiny has significantly fewer parameters than ViT-Base
        assert total_params > 25_000_000, "Model too small (should be ~27-28M)"
        assert total_params < 35_000_000, "Model too large (should be ~27-28M)"
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass works correctly"""
    print("\n" + "="*70)
    print("🧪 Testing Forward Pass")
    print("="*70)
    
    try:
        from model import create_vixnet
        model = create_vixnet(pretrained=False, num_classes=2)
        model.eval()
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        print(f"   Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Output shape: {output.shape}")
        
        # Verify output shape
        assert output.shape == (batch_size, 2), f"Wrong output shape: {output.shape}"
        
        print("✅ Forward pass successful")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_freezing_mechanism():
    """Test freezing and unfreezing mechanism"""
    print("\n" + "="*70)
    print("🔒 Testing Freezing Mechanism")
    print("="*70)
    
    try:
        from model import create_vixnet
        model = create_vixnet(pretrained=False, num_classes=2)
        
        # Test initial state
        initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Initial trainable parameters: {initial_trainable:,}")
        
        # Test freezing
        model.freeze_feature_extractors()
        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   After freezing: {frozen_trainable:,}")
        
        assert frozen_trainable < initial_trainable * 0.2, "Too many parameters still trainable"
        
        # Test unfreezing
        model.unfreeze_high_level_layers()
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   After unfreezing: {unfrozen_trainable:,}")
        
        assert unfrozen_trainable > frozen_trainable, "Unfreezing didn't work"
        assert unfrozen_trainable < initial_trainable * 0.5, "Too many parameters unfrozen"
        
        print("✅ Freezing mechanism works correctly")
        return True
    except Exception as e:
        print(f"❌ Freezing mechanism failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\n" + "="*70)
    print("⚙️  Testing Configuration")
    print("="*70)
    
    try:
        from config import Config
        
        # Test stage configs
        stage1_config = Config.get_stage_config(1)
        stage2_config = Config.get_stage_config(2)
        
        print(f"   Stage 1 epochs: {stage1_config['epochs']}")
        print(f"   Stage 1 LR: {stage1_config['lr']}")
        print(f"   Stage 2 epochs: {stage2_config['epochs']}")
        print(f"   Stage 2 LR: {stage2_config['lr']}")
        
        # Verify stage 2 LR is much lower
        assert stage2_config['lr'] < stage1_config['lr'] / 50, "Stage 2 LR should be much lower"
        
        print("✅ Configuration is correct")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_check():
    """Test dataset availability check"""
    print("\n" + "="*70)
    print("📂 Testing Dataset Check")
    print("="*70)
    
    try:
        from dataset import check_dataset_availability
        
        dataset_available = check_dataset_availability()
        
        if dataset_available:
            print("✅ Dataset is available")
        else:
            print("⚠️  Dataset not available (this is OK for testing)")
        
        return True
    except Exception as e:
        print(f"❌ Dataset check failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_example():
    """Test inference example"""
    print("\n" + "="*70)
    print("🎯 Testing Inference Example")
    print("="*70)
    
    try:
        from inference_example import load_model
        
        # Load model (will use random weights since no checkpoint)
        model = load_model('checkpoints/best_model.pth')
        
        print("✅ Inference example loads correctly")
        return True
    except Exception as e:
        print(f"❌ Inference example failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*70)
    print("🧪 VIXNET IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Freezing Mechanism", test_freezing_mechanism),
        ("Configuration", test_config),
        ("Dataset Check", test_dataset_check),
        ("Inference Example", test_inference_example),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
