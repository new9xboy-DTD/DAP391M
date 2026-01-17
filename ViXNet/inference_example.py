"""
Example script demonstrating how to use ViXNet for inference
"""

import torch
import sys
import os
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_vixnet
from config import Config


def load_model(checkpoint_path='checkpoints/best_model.pth'):
    """
    Load trained ViXNet model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model in evaluation mode
    """
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Create model
    model = create_vixnet(pretrained=False, num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        print(f"   Trained epoch: {checkpoint['epoch']}")
        print(f"   Validation accuracy: {checkpoint['metrics']['accuracy']:.4f}")
    else:
        print(f"⚠️  Checkpoint not found. Using randomly initialized model.")
    
    model.eval()
    return model


def predict_image(model, image_path, class_names=['Fake', 'Real']):
    """
    Predict whether an image is real or fake
    
    Args:
        model: Trained ViXNet model
        image_path: Path to image file
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    result = {
        'class': class_names[predicted_class],
        'class_id': predicted_class,
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[0][i].item()
            for i in range(len(class_names))
        }
    }
    
    return result


def predict_batch(model, image_paths, class_names=['Fake', 'Real']):
    """
    Predict multiple images at once
    
    Args:
        model: Trained ViXNet model
        image_paths: List of image file paths
        class_names: List of class names
        
    Returns:
        List of prediction results
    """
    results = []
    
    for image_path in image_paths:
        print(f"\n🖼️  Processing: {image_path}")
        try:
            result = predict_image(model, image_path, class_names)
            results.append(result)
            
            print(f"   Prediction: {result['class']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"      {cls}: {prob:.2%}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append(None)
    
    return results


def main():
    """
    Main function demonstrating inference
    """
    print("="*70)
    print("🧪 VIXNET INFERENCE EXAMPLE")
    print("="*70)
    
    # Load model
    model = load_model()
    
    print("\n" + "="*70)
    print("📋 USAGE INSTRUCTIONS")
    print("="*70)
    print("\n1. To predict a single image:")
    print("   >>> from inference_example import predict_image, load_model")
    print("   >>> model = load_model('checkpoints/best_model.pth')")
    print("   >>> result = predict_image(model, 'path/to/image.jpg')")
    print("   >>> print(result)")
    
    print("\n2. To predict multiple images:")
    print("   >>> from inference_example import predict_batch, load_model")
    print("   >>> model = load_model('checkpoints/best_model.pth')")
    print("   >>> results = predict_batch(model, ['img1.jpg', 'img2.jpg'])")
    
    print("\n3. In Python script:")
    print("   ```python")
    print("   import torch")
    print("   from model import create_vixnet")
    print("   ")
    print("   # Load model")
    print("   model = create_vixnet()")
    print("   checkpoint = torch.load('checkpoints/best_model.pth')")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("   model.eval()")
    print("   ")
    print("   # Predict")
    print("   # ... (preprocess image, run inference)")
    print("   ```")
    
    print("\n" + "="*70)
    print("✅ Example script ready!")
    print("="*70)
    
    # Test with dummy prediction
    print("\n🧪 Testing with random input...")
    dummy_input = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(dummy_input)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    
    print(f"   Prediction: {'Real' if pred == 1 else 'Fake'}")
    print(f"   Confidence: {conf:.2%}")
    print("\n✅ Model is ready for inference!")


if __name__ == "__main__":
    main()
