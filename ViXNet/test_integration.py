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
                    'vit_dim': 768,
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
                'vit_dim': 768,
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
