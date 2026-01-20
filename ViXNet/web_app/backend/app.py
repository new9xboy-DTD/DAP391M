"""
Flask backend API for ViXNet model visualization and inference
Provides endpoints for:
- Image inference (drag-and-drop)
- Model analysis with AUC calculation
- Model information and architecture
- Multiple model architecture support (ViXNet, Xception Only, ViT Only)
- Dataset selection for model evaluation
"""

import os
import sys
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import io
import base64

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ViXNet.model import create_data_loaders
from ViXNet.model_factory import create_model, load_model_from_checkpoint, detect_model_type
from ViXNet.config import Config
from torchvision import datasets

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model variable - No default model loaded
current_model = None
model_info = {
    'loaded': False,
    'model_type': None,
    'checkpoint_path': None,
    'metrics': None
}

# Uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def convert_to_serializable(obj):
    """
    Convert numpy and torch types to Python native types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return obj


def preprocess_image(image):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)


def create_test_loader(dataset_key='default', batch_size=32, num_workers=2):
    """
    Create test data loader for a specific dataset
    
    Args:
        dataset_key: Key for dataset in Config.DATASETS
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader or None if dataset not available
    """
    try:
        dataset_config = Config.get_dataset_config(dataset_key)
        test_dir = dataset_config['test']
        
        if not os.path.exists(test_dir):
            print(f"⚠️  Test directory not found: {test_dir}")
            return None
        
        # Define transforms
        test_transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Create dataset
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        
        # Create data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY
        )
        
        print(f"✅ Test loader created: {len(test_dataset)} images")
        return test_loader
        
    except Exception as e:
        print(f"❌ Error creating test loader: {str(e)}")
        return None


def calculate_auc_on_test_set(model, dataset_key='default'):
    """
    Calculate AUC on the test dataset
    
    Args:
        model: Model to evaluate
        dataset_key: Key for dataset in Config.DATASETS
        
    Returns:
        tuple: (results dict, error message)
    """
    try:
        print(f"📊 Calculating AUC on test set (dataset: {dataset_key})...")
        
        # Load test dataset
        test_loader = create_data_loaders(batch_size=Config.STAGE1_BATCH_SIZE, num_workers=Config.NUM_WORKERS).get('test_loader')
        
        if test_loader is None:
            return None, f"Test dataset '{dataset_key}' not available"
        
        model.eval()
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "Real" class
                all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        auc = roc_auc_score(all_labels, all_probs)
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        
        results = {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            },
            'num_samples': len(all_labels),
            'dataset_key': dataset_key
        }
        
        print(f"✅ AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        return results, None
        
    except Exception as e:
        print(f"❌ Error calculating AUC: {str(e)}")
        return None, str(e)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info['loaded']
    })


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets"""
    try:
        datasets = Config.list_available_datasets()
        return jsonify({
            'datasets': datasets,
            'count': len(datasets)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    return jsonify(convert_to_serializable(model_info))


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if an image is real or fake"""
    if not model_info['loaded']:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
        
        # Preprocess
        image_tensor = preprocess_image(image).to(Config.DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = current_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ['Fake', 'Real']
        result = {
            'prediction': class_names[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'Fake': float(probabilities[0][0].item()),
                'Real': float(probabilities[0][1].item())
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-model', methods=['POST'])
def analyze_model():
    """
    Analyze an uploaded model and calculate AUC
    
    Security Note: Only upload models from trusted sources. The weights_only=False
    flag is used for checkpoint compatibility but can execute arbitrary code.
    """
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    try:
        global current_model, model_info
        
        # Get dataset selection (default to 'default')
        dataset_key = request.form.get('dataset', 'default')
        
        # Save uploaded model
        model_file = request.files['model']
        model_path = os.path.join(UPLOAD_FOLDER, 'uploaded_model.pth')
        model_file.save(model_path)
        
        # Load the model using model factory
        print(f"🔄 Loading uploaded model from: {model_path}")
        
        try:
            new_model, new_model_info = load_model_from_checkpoint(
                model_path, 
                device=Config.DEVICE,
                strict=False
            )
        except Exception as load_error:
            error_msg = f"Failed to load model: {str(load_error)}"
            print(f"❌ {error_msg}")
            return jsonify({
                'error': error_msg,
                'details': str(load_error)[:500]
            }), 400
        
        # Calculate AUC on selected test set
        auc_results, error = calculate_auc_on_test_set(new_model, dataset_key=dataset_key)
        
        if error:
            return jsonify({
                'warning': 'Model loaded but AUC calculation failed',
                'error': error,
                'model_info': new_model_info
            }), 200
        
        # Update current model
        current_model = new_model
        model_info = new_model_info
        model_info['auc_results'] = auc_results
        
        # Convert all to serializable types
        model_info = convert_to_serializable(model_info)
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        print(f"❌ Error analyzing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/calculate-auc', methods=['POST'])
def calculate_auc():
    """Calculate AUC for the current model on a selected dataset"""
    if not model_info['loaded']:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get dataset selection from request body
        data = request.get_json() or {}
        dataset_key = data.get('dataset', 'default')
        
        auc_results, error = calculate_auc_on_test_set(current_model, dataset_key=dataset_key)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Update model info with AUC results
        model_info['auc_results'] = auc_results
        
        return jsonify(auc_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*70)
    print("🚀 Starting Multi-Model Deepfake Detection Web API")
    print("="*70)
    
    print("\n📦 Supported Model Types:")
    print("   • ViXNet (Vision Transformer + Xception)")
    print("   • Xception Only")
    print("   • ViT Only")
    
    print("\n📁 Available Datasets:")
    available_datasets = Config.list_available_datasets()
    for ds in available_datasets:
        print(f"   • {ds['name']} ({ds['key']})")
    
    print("\n" + "="*70)
    print("📡 API Endpoints:")
    print("="*70)
    print("  GET  /api/health          - Health check")
    print("  GET  /api/datasets        - List available datasets")
    print("  GET  /api/model-info      - Get current model information")
    print("  POST /api/analyze-model   - Upload and analyze model with AUC")
    print("       Form data: model (file), dataset (string, optional)")
    print("  POST /api/calculate-auc   - Calculate AUC for current model")
    print("       JSON body: {\"dataset\": \"default\"}")
    print("  POST /api/predict         - Predict image (Real/Fake)")
    print("\n" + "="*70)
    print("🌐 Server running on http://localhost:5000")
    print("="*70)
    print("\n⚠️  Note: Running in development mode.")
    print("   For production, use a WSGI server like Gunicorn:")
    print("   gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    print("\n💡 Tip: No model is loaded by default. Upload a model to begin.")
    print("")
    
    # Run in development mode - for production use gunicorn or similar
    app.run(host='0.0.0.0', port=5000, debug=True)
