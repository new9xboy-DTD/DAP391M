"""
Test script to verify the Stage 1 checkpoint skip functionality
"""

import os
import sys
import tempfile
import shutil
from config import Config
from utils import check_stage1_complete, save_checkpoint
import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def state_dict(self):
        return {'fc.weight': torch.randn(2, 10), 'fc.bias': torch.randn(2)}
    
    def load_state_dict(self, state_dict):
        pass


def test_check_stage1_complete():
    """Test the check_stage1_complete function"""
    print("=" * 70)
    print("Testing check_stage1_complete function")
    print("=" * 70)
    
    # Save original SAVE_DIR
    original_save_dir = Config.SAVE_DIR
    
    # Create a temporary directory for testing with restricted permissions
    temp_dir = tempfile.mkdtemp(prefix='vixnet_test_', suffix='_checkpoints')
    Config.SAVE_DIR = temp_dir
    
    try:
        # Test 1: No checkpoints directory
        print("\nTest 1: No checkpoints directory")
        result = check_stage1_complete()
        print(f"Result: {result}")
        assert result == False, "Expected False when no checkpoints directory exists"
        print("✅ PASSED: Returns False when no checkpoints directory")
        
        # Create the checkpoints directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Test 2: Empty checkpoints directory
        print("\nTest 2: Empty checkpoints directory")
        result = check_stage1_complete()
        print(f"Result: {result}")
        assert result == False, "Expected False when checkpoints directory is empty"
        print("✅ PASSED: Returns False when checkpoints directory is empty")
        
        # Test 3: Create incomplete checkpoints (only epochs 1-3)
        print("\nTest 3: Incomplete checkpoints (epochs 1-3 out of 5)")
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        dummy_metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.85,
            'f1': 0.80,
            'confusion_matrix': [[10, 2], [3, 15]]
        }
        
        for epoch in range(1, 4):  # Only create epochs 1-3
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': dummy_metrics,
                'config': {},
                'timestamp': '2024-01-01 00:00:00'
            }
            checkpoint_path = os.path.join(temp_dir, f'checkpoint_stage1_epoch{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"   Created: checkpoint_stage1_epoch{epoch}.pth")
        
        result = check_stage1_complete()
        print(f"Result: {result}")
        assert result == False, "Expected False when only 3 out of 5 checkpoints exist"
        print("✅ PASSED: Returns False when checkpoints are incomplete")
        
        # Test 4: Create all 5 epoch checkpoints but no best model
        print("\nTest 4: All 5 epoch checkpoints but no best model")
        for epoch in range(4, 6):  # Create epochs 4-5
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': dummy_metrics,
                'config': {},
                'timestamp': '2024-01-01 00:00:00'
            }
            checkpoint_path = os.path.join(temp_dir, f'checkpoint_stage1_epoch{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"   Created: checkpoint_stage1_epoch{epoch}.pth")
        
        result = check_stage1_complete()
        print(f"Result: {result}")
        assert result == False, "Expected False when best model doesn't exist"
        print("✅ PASSED: Returns False when best model is missing")
        
        # Test 5: Create best model - now it should be complete
        print("\nTest 5: All 5 epoch checkpoints + best model")
        best_checkpoint = {
            'epoch': 5,
            'stage': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': dummy_metrics,
            'config': {},
            'timestamp': '2024-01-01 00:00:00'
        }
        best_path = os.path.join(temp_dir, 'best_model_stage1.pth')
        torch.save(best_checkpoint, best_path)
        print(f"   Created: best_model_stage1.pth")
        
        result = check_stage1_complete()
        print(f"Result: {result}")
        assert result == True, "Expected True when all checkpoints and best model exist"
        print("✅ PASSED: Returns True when Stage 1 is complete")
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
    finally:
        # Cleanup: Restore original SAVE_DIR and remove temp directory
        Config.SAVE_DIR = original_save_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_check_stage1_complete()
