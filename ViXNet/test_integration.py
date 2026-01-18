"""
Integration test to verify the JSON save function works with real training history format
"""

import numpy as np
import json
import os
import sys
import tempfile

# Add ViXNet to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_training_history


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
