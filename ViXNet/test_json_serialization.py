"""
Test script to verify JSON serialization works with numpy types
"""

import numpy as np
import json
import sys
import os

# Add ViXNet to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import convert_to_json_serializable


def test_numpy_types():
    """Test conversion of various numpy types"""
    print("Testing numpy type conversion...")
    
    # Create test data with various numpy types
    test_data = {
        'int64_value': np.int64(42),
        'int32_value': np.int32(100),
        'float64_value': np.float64(3.14159),
        'float32_value': np.float32(2.718),
        'bool_value': np.bool_(True),
        'array': np.array([1, 2, 3, 4, 5]),
        'nested_dict': {
            'metric': np.float64(0.95),
            'count': np.int64(1000)
        },
        'list_with_numpy': [np.int64(1), np.float64(2.5), np.array([3, 4])],
        'confusion_matrix': np.array([[10, 2], [3, 15]]),
        'regular_int': 123,
        'regular_float': 4.56,
        'regular_string': 'test'
    }
    
    try:
        # Convert to JSON-serializable format
        converted = convert_to_json_serializable(test_data)
        
        # Try to serialize to JSON
        json_string = json.dumps(converted, indent=2)
        
        print("✅ JSON serialization successful!")
        print("\nOriginal types:")
        print(f"  int64_value: {type(test_data['int64_value'])}")
        print(f"  float64_value: {type(test_data['float64_value'])}")
        print(f"  array: {type(test_data['array'])}")
        
        print("\nConverted types:")
        print(f"  int64_value: {type(converted['int64_value'])}")
        print(f"  float64_value: {type(converted['float64_value'])}")
        print(f"  array: {type(converted['array'])}")
        
        print("\nSample JSON output:")
        print(json_string[:300] + "...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_training_history_format():
    """Test with realistic training history format"""
    print("\n\nTesting realistic training history format...")
    
    # Create mock training history similar to actual usage
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
            'lr': np.float64(0.0009)
        }
    ]
    
    try:
        # Convert and serialize
        converted = convert_to_json_serializable(history)
        json_string = json.dumps(converted, indent=2)
        
        # Verify we can parse it back
        parsed = json.loads(json_string)
        
        print("✅ Training history serialization successful!")
        print(f"\nNumber of epochs: {len(parsed)}")
        print(f"Epoch 1 validation accuracy: {parsed[0]['val']['accuracy']}")
        print(f"Confusion matrix shape preserved: {len(parsed[0]['val']['confusion_matrix'])}x{len(parsed[0]['val']['confusion_matrix'][0])}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("JSON SERIALIZATION TEST")
    print("="*70)
    
    test1_passed = test_numpy_types()
    test2_passed = test_training_history_format()
    
    print("\n" + "="*70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)
