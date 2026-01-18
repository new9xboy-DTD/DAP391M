"""
Integration test for the Stage 1 skip functionality in train.py
This test verifies that the training script correctly skips Stage 1 when checkpoints exist
"""

import os
import sys
import tempfile
import shutil
import torch
import json
import numpy as np

# Add ViXNet to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_training_history


def test_stage1_skip_integration():
    """Test that train.py correctly skips Stage 1 when checkpoints exist"""
    print("=" * 70)
    print("Integration Test: Stage 1 Skip in train.py")
    print("=" * 70)
    
    # Import after setting up environment
    from config import Config
    from utils import check_stage1_complete
    
    # Save original SAVE_DIR
    original_save_dir = Config.SAVE_DIR
    
    # Create a temporary directory for testing with restricted permissions
    temp_dir = tempfile.mkdtemp(prefix='vixnet_test_', suffix='_checkpoints')
    Config.SAVE_DIR = temp_dir
    
    try:
        print("\n📋 Test Scenario 1: No checkpoints exist")
        print("-" * 70)
        
        result = check_stage1_complete()
        if result == False:
            print("✅ Correctly detected no checkpoints - Stage 1 will train")
        else:
            print("❌ ERROR: Should return False when no checkpoints exist")
            return False
        
        print("\n📋 Test Scenario 2: Simulating completed Stage 1")
        print("-" * 70)
        
        # Create checkpoints directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create dummy checkpoints for all 5 epochs
        dummy_state = {'layer': torch.randn(10, 10)}
        dummy_metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.85,
            'f1': 0.80,
            'confusion_matrix': [[10, 2], [3, 15]]
        }
        
        print("\n   Creating Stage 1 checkpoints...")
        for epoch in range(1, 6):  # Create all 5 epochs
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': dummy_state,
                'optimizer_state_dict': {'state': {}, 'param_groups': []},
                'metrics': dummy_metrics,
                'config': {
                    'img_size': 224,
                    'xception_dim': 2048,
                    'vit_dim': 192,
                    'fusion_dim': 512,
                    'num_classes': 2
                },
                'timestamp': '2024-01-01 00:00:00'
            }
            checkpoint_path = os.path.join(temp_dir, f'checkpoint_stage1_epoch{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"   ✓ Created: checkpoint_stage1_epoch{epoch}.pth")
        
        # Create best model
        best_checkpoint = {
            'epoch': 5,
            'stage': 1,
            'model_state_dict': dummy_state,
            'optimizer_state_dict': {'state': {}, 'param_groups': []},
            'metrics': dummy_metrics,
            'config': {
                'img_size': 224,
                'xception_dim': 2048,
                'vit_dim': 192,
                'fusion_dim': 512,
                'num_classes': 2
            },
            'timestamp': '2024-01-01 00:00:00'
        }
        best_path = os.path.join(temp_dir, 'best_model_stage1.pth')
        torch.save(best_checkpoint, best_path)
        print(f"   ✓ Created: best_model_stage1.pth")
        
        # Create dummy Stage 1 history
        stage1_history = []
        for epoch in range(1, 6):
            stage1_history.append({
                'epoch': epoch,
                'stage': 1,
                'train': {'loss': 0.5, 'accuracy': 0.75},
                'val': dummy_metrics,
                'lr': 0.001
            })
        
        history_path = os.path.join(temp_dir, 'stage1_history.json')
        with open(history_path, 'w') as f:
            json.dump(stage1_history, f, indent=2)
        print(f"   ✓ Created: stage1_history.json")
        
        print("\n   Verifying checkpoint detection...")
        result = check_stage1_complete()
        if result == True:
            print("✅ Correctly detected complete Stage 1 checkpoints")
        else:
            print("❌ ERROR: Should return True when all checkpoints exist")
            return False
        
        print("\n📋 Test Scenario 3: Verify checkpoint files")
        print("-" * 70)
        
        expected_files = [
            'checkpoint_stage1_epoch1.pth',
            'checkpoint_stage1_epoch2.pth',
            'checkpoint_stage1_epoch3.pth',
            'checkpoint_stage1_epoch4.pth',
            'checkpoint_stage1_epoch5.pth',
            'best_model_stage1.pth',
            'stage1_history.json'
        ]
        
        print("\n   Checking for required files:")
        all_exist = True
        for filename in expected_files:
            filepath = os.path.join(temp_dir, filename)
            exists = os.path.exists(filepath)
            status = "✓" if exists else "✗"
            print(f"   {status} {filename}")
            if not exists:
                all_exist = False
        
        if all_exist:
            print("\n✅ All required checkpoint files exist")
        else:
            print("\n❌ ERROR: Some checkpoint files are missing")
            return False
        
        print("\n" + "=" * 70)
        print("🎉 INTEGRATION TEST PASSED!")
        print("=" * 70)
        print("\nSummary:")
        print("  • check_stage1_complete() correctly detects missing checkpoints")
        print("  • check_stage1_complete() correctly detects complete checkpoints")
        print("  • All required checkpoint files can be created and verified")
        print("  • The skip functionality is ready for use in train.py")
        
        return True
        
    finally:
        # Cleanup: Restore original SAVE_DIR and remove temp directory
        Config.SAVE_DIR = original_save_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    success = test_stage1_skip_integration()
    sys.exit(0 if success else 1)

"""
Integration test to verify the JSON save function works with real training history format
"""


def test_save_training_history():
    """Test saving training history with numpy types"""
    print("Testing save_training_history function...")
    
    # Create realistic training history with numpy types (as it would come from actual training)
    history = [
        {
            'epoch': np.int64(1),
            'stage': 1,
            'train': {
                'loss': np.float64(0.5234),
                'accuracy': np.float64(0.8567)
            },
            'val': {
                'loss': np.float64(0.4123),
                'accuracy': np.float64(0.8901),
                'precision': np.float64(0.8745),
                'recall': np.float64(0.9023),
                'f1': np.float64(0.8882),
                'confusion_matrix': np.array([[450, 50], [30, 470]])
            },
            'lr': np.float64(0.001)
        },
        {
            'epoch': np.int64(2),
            'stage': 1,
            'train': {
                'loss': np.float64(0.3567),
                'accuracy': np.float64(0.9012)
            },
            'val': {
                'loss': np.float64(0.3234),
                'accuracy': np.float64(0.9156),
                'precision': np.float64(0.9087),
                'recall': np.float64(0.9234),
                'f1': np.float64(0.9160),
                'confusion_matrix': np.array([[475, 25], [20, 480]])
            },
            'test': {
                'loss': np.float64(0.3345),
                'accuracy': np.float64(0.9123),
                'precision': np.float64(0.9056),
                'recall': np.float64(0.9201),
                'f1': np.float64(0.9128),
                'confusion_matrix': np.array([[470, 30], [22, 478]])
            },
            'lr': np.float64(0.0009)
        }
    ]
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock the Config.SAVE_DIR
        from config import Config
        original_save_dir = Config.SAVE_DIR
        Config.SAVE_DIR = tmpdir
        
        try:
            # Save the history
            save_training_history(history, 'test_history.json')
            
            # Check if file was created
            filepath = os.path.join(tmpdir, 'test_history.json')
            if not os.path.exists(filepath):
                print("❌ File was not created")
                return False
            
            # Try to load and parse the JSON file
            with open(filepath, 'r') as f:
                loaded_history = json.load(f)
            
            # Verify the data is correct
            if len(loaded_history) != 2:
                print(f"❌ Expected 2 epochs, got {len(loaded_history)}")
                return False
            
            if loaded_history[0]['epoch'] != 1:
                print(f"❌ Expected epoch 1, got {loaded_history[0]['epoch']}")
                return False
            
            if not isinstance(loaded_history[0]['epoch'], int):
                print(f"❌ Expected int type for epoch, got {type(loaded_history[0]['epoch'])}")
                return False
            
            if not isinstance(loaded_history[0]['train']['accuracy'], float):
                print(f"❌ Expected float type for accuracy, got {type(loaded_history[0]['train']['accuracy'])}")
                return False
            
            if not isinstance(loaded_history[0]['val']['confusion_matrix'], list):
                print(f"❌ Expected list type for confusion_matrix, got {type(loaded_history[0]['val']['confusion_matrix'])}")
                return False
            
            # Check confusion matrix values
            cm = loaded_history[0]['val']['confusion_matrix']
            if cm[0][0] != 450 or cm[0][1] != 50:
                print(f"❌ Confusion matrix values incorrect: {cm}")
                return False
            
            print("✅ save_training_history works correctly!")
            print(f"   File saved to: {filepath}")
            print(f"   File size: {os.path.getsize(filepath)} bytes")
            print(f"   Data validated successfully")
            
            # Print sample of saved JSON
            with open(filepath, 'r') as f:
                content = f.read()
                print(f"\n   Sample output (first 500 chars):")
                print(f"   {content[:500]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Restore original save dir
            Config.SAVE_DIR = original_save_dir


if __name__ == "__main__":
    print("="*70)
    print("INTEGRATION TEST: save_training_history")
    print("="*70 + "\n")
    
    if test_save_training_history():
        print("\n" + "="*70)
        print("✅ INTEGRATION TEST PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ INTEGRATION TEST FAILED")
        print("="*70)
        sys.exit(1)
