"""
Demo script showing the JSON serialization fix
This demonstrates how the fix handles numpy types that previously caused errors
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import convert_to_json_serializable

print("="*70)
print("DEMONSTRATION: JSON Serialization Fix for NumPy Types")
print("="*70)

# Create sample data similar to what's produced during training
sample_training_data = {
    'epoch': np.int64(10),
    'stage': 1,
    'train': {
        'loss': np.float64(0.2345),
        'accuracy': np.float64(0.9234)
    },
    'val': {
        'loss': np.float64(0.2567),
        'accuracy': np.float64(0.9123),
        'precision': np.float64(0.9087),
        'recall': np.float64(0.9156),
        'f1': np.float64(0.9121),
        'confusion_matrix': np.array([[480, 20], [15, 485]])
    }
}

print("\n1. Original data contains NumPy types:")
print(f"   - epoch type: {type(sample_training_data['epoch'])}")
print(f"   - accuracy type: {type(sample_training_data['train']['accuracy'])}")
print(f"   - confusion_matrix type: {type(sample_training_data['val']['confusion_matrix'])}")

print("\n2. Attempting to serialize WITHOUT conversion:")
try:
    json.dumps(sample_training_data)
    print("   ✅ Serialization successful (unexpected!)")
except TypeError as e:
    print(f"   ❌ Error: {e}")

print("\n3. Using convert_to_json_serializable function:")
converted_data = convert_to_json_serializable(sample_training_data)
print(f"   - epoch type: {type(converted_data['epoch'])}")
print(f"   - accuracy type: {type(converted_data['train']['accuracy'])}")
print(f"   - confusion_matrix type: {type(converted_data['val']['confusion_matrix'])}")

print("\n4. Attempting to serialize WITH conversion:")
try:
    json_output = json.dumps(converted_data, indent=2)
    print("   ✅ Serialization successful!")
    print("\n5. Sample JSON output:")
    print(json_output[:400] + "...")
except TypeError as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("The convert_to_json_serializable function successfully converts:")
print("  • np.int64, np.int32, etc. → int")
print("  • np.float64, np.float32, etc. → float")
print("  • np.ndarray → list")
print("  • np.bool_ → bool")
print("  • Handles nested dictionaries, lists, and tuples recursively")
print("="*70)
