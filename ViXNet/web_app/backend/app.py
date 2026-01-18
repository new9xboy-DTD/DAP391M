"""
Flask backend API for ViXNet model visualization and inference
Provides endpoints for:
- Image inference (drag-and-drop)
- Model analysis with AUC calculation
- Model information and architecture
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
from model import create_vixnet
from config import Config
from dataset import create_dataloaders

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global model variable
current_model = None
model_info = {
    'loaded': False,
    'checkpoint_path': None,
    'metrics': None
}

# Uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_default_model():
    """
    Load the default trained model
    
    Security Note: Using weights_only=False for torch.load() to maintain
    compatibility with existing checkpoints. In production environments,
    only load models from trusted sources or use weights_only=True with
    appropriate checkpoint format validation.
    """
    global current_model, model_info
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'best_model.pth')
    
    try:
        print(f"🔄 Loading default model from: {checkpoint_path}")
        current_model = create_vixnet(pretrained=False, num_classes=Config.NUM_CLASSES)
        current_model = current_model.to(Config.DEVICE)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
            current_model.load_state_dict(checkpoint['model_state_dict'])
            current_model.eval()
            
            model_info = {
                'loaded': True,
                'checkpoint_path': checkpoint_path,
                'metrics': checkpoint.get('metrics', {}),
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'architecture': {
                    'name': 'ViXNet',
                    'xception_dim': 2048,
                    'vit_dim': 192,
                    'fusion_dim': 512,
                    'num_classes': 2
                }
            }
            print("✅ Default model loaded successfully")
        else:
            print("⚠️  No checkpoint found, using untrained model")
            current_model.eval()
            model_info = {
                'loaded': True,
                'checkpoint_path': None,
                'metrics': {},
                'epoch': 0,
                'architecture': {
                    'name': 'ViXNet',
                    'xception_dim': 2048,
                    'vit_dim': 192,
                    'fusion_dim': 512,
                    'num_classes': 2
                }
            }
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model_info['loaded'] = False


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


def calculate_auc_on_test_set(model):
    """Calculate AUC on the test dataset"""
    try:
        print("📊 Calculating AUC on test set...")
        
        # Load test dataset
        _, _, test_loader = create_dataloaders(
            batch_size=32,
            num_workers=2
        )
        
        if test_loader is None:
            return None, "Test dataset not available"
        
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
            'num_samples': len(all_labels)
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


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    return jsonify(model_info)


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
        
        # Save uploaded model
        model_file = request.files['model']
        model_path = os.path.join(UPLOAD_FOLDER, 'uploaded_model.pth')
        model_file.save(model_path)
        
        # Load the model
        print(f"🔄 Loading uploaded model from: {model_path}")
        new_model = create_vixnet(pretrained=False, num_classes=Config.NUM_CLASSES)
        new_model = new_model.to(Config.DEVICE)
        
        checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_model.eval()
        
        # Calculate AUC on test set
        auc_results, error = calculate_auc_on_test_set(new_model)
        
        if error:
            return jsonify({
                'warning': 'Model loaded but AUC calculation failed',
                'error': error,
                'model_info': {
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'metrics': checkpoint.get('metrics', {})
                }
            }), 200
        
        # Update current model
        current_model = new_model
        model_info = {
            'loaded': True,
            'checkpoint_path': model_path,
            'metrics': checkpoint.get('metrics', {}),
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'auc_results': auc_results,
            'architecture': {
                'name': 'ViXNet',
                'xception_dim': 2048,
                'vit_dim': 192,
                'fusion_dim': 512,
                'num_classes': 2
            }
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calculate-auc', methods=['POST'])
def calculate_auc():
    """Calculate AUC for the current model"""
    if not model_info['loaded']:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        auc_results, error = calculate_auc_on_test_set(current_model)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Update model info with AUC results
        model_info['auc_results'] = auc_results
        
        return jsonify(auc_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*70)
    print("🚀 Starting ViXNet Web API")
    print("="*70)
    
    # Load default model
    load_default_model()
    
    print("\n" + "="*70)
    print("📡 API Endpoints:")
    print("="*70)
    print("  GET  /api/health          - Health check")
    print("  GET  /api/model-info      - Get model information")
    print("  POST /api/predict         - Predict image (Real/Fake)")
    print("  POST /api/analyze-model   - Upload and analyze model with AUC")
    print("  POST /api/calculate-auc   - Calculate AUC for current model")
    print("\n" + "="*70)
    print("🌐 Server running on http://localhost:5000")
    print("="*70)
    print("\n⚠️  Note: Running in development mode.")
    print("   For production, use a WSGI server like Gunicorn:")
    print("   gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    print("")
    
    # Run in development mode - for production use gunicorn or similar
    app.run(host='0.0.0.0', port=5000, debug=True)
