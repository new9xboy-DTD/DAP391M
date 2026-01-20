"""
Simple test script to verify backend components work correctly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from flask import Flask
        print("  ✓ Flask imported")
    except ImportError as e:
        print(f"  ✗ Flask import failed: {e}")
        return False
    
    try:
        from flask_cors import CORS
        print("  ✓ flask-cors imported")
    except ImportError as e:
        print(f"  ✗ flask-cors import failed: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ PyTorch imported (version {torch.__version__})")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        print("    Note: PyTorch is required for model operations")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ NumPy imported (version {np.__version__})")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        from sklearn.metrics import roc_auc_score
        print("  ✓ scikit-learn imported")
    except ImportError as e:
        print(f"  ✗ scikit-learn import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("  ✓ Pillow imported")
    except ImportError as e:
        print(f"  ✗ Pillow import failed: {e}")
        return False
    
    return True

def test_model_module():
    """Test that model module can be imported"""
    print("\nTesting ViXNet model module...")
    
    try:
        from model import create_vixnet
        print("  ✓ Model module imported")
        return True
    except ImportError as e:
        print(f"  ✗ Model import failed: {e}")
        print("    Make sure you're running from ViXNet/web_app/backend/")
        return False

def test_config_module():
    """Test that config module can be imported"""
    print("\nTesting config module...")
    
    try:
        from config import Config
        print("  ✓ Config module imported")
        print(f"    Device: {Config.DEVICE}")
        print(f"    Image size: {Config.IMG_SIZE}")
        return True
    except ImportError as e:
        print(f"  ✗ Config import failed: {e}")
        return False

def test_flask_app():
    """Test that Flask app can be created"""
    print("\nTesting Flask app creation...")
    
    try:
        from flask import Flask
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/test')
        def test():
            return {'status': 'ok'}
        
        print("  ✓ Flask app created successfully")
        print("  ✓ CORS configured")
        print("  ✓ Test route added")
        return True
    except Exception as e:
        print(f"  ✗ Flask app creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("ViXNet Backend Component Tests")
    print("="*70)
    
    results = []
    
    results.append(("Import Test", test_imports()))
    results.append(("Model Module Test", test_model_module()))
    results.append(("Config Module Test", test_config_module()))
    results.append(("Flask App Test", test_flask_app()))
    
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed!")
        print("="*70)
        return 0
    else:
        print("⚠️  Some tests failed. Please install missing dependencies.")
        print("   Run: pip install -r requirements.txt")
        print("="*70)
        return 1

if __name__ == "__main__":
    exit(main())
