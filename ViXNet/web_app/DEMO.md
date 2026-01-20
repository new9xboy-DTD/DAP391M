# ViXNet Web Application Demo

## Overview

This document provides a step-by-step guide to demonstrate the ViXNet web application functionality.

## Setup

### Prerequisites
1. Python 3.8+ with PyTorch, Flask, and dependencies installed
2. Node.js 16+ and npm installed
3. A trained ViXNet model checkpoint (optional - untrained model works for demo)
4. Test dataset at `CNN + Transformer/Dataset/Test/` (optional - for AUC calculation)

### Installation

1. **Install Backend Dependencies:**
   ```bash
   cd ViXNet/web_app/backend
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies:**
   ```bash
   cd ViXNet/web_app/frontend
   npm install
   ```

## Running the Application

### Option 1: Using Startup Scripts (Recommended)

**Linux/Mac:**
```bash
cd ViXNet/web_app
./start.sh
```

**Windows:**
```bash
cd ViXNet\web_app
start.bat
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd ViXNet/web_app/backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd ViXNet/web_app/frontend
npm start
```

The application will open automatically at `http://localhost:3000`

## Features Demo

### 1. Model Architecture Visualization

Upon opening the app, you'll see:
- Interactive visualization of the ViXNet 2-branch architecture
- Feature dimensions at each stage
- Current model information (if loaded)

**What to observe:**
- Xception branch (CNN) - extracts global spatial features (2048D)
- ViT branch (Transformer) - learns patch-wise attention (192D)
- Feature fusion layer (512D)
- Classification head (Real/Fake output)

### 2. Image Inference

**Steps:**
1. Locate the "Image Inference" section
2. Drag and drop an image file (JPG, PNG, JPEG) or click to browse
3. Wait for processing (~1-2 seconds)
4. View results:
   - Prediction: Real or Fake
   - Confidence score
   - Probability breakdown for both classes
   - Visual confidence meter

**Test Cases:**
- Upload a real face image → Should predict "Real" with high confidence
- Upload a deepfake image → Should predict "Fake" with high confidence
- Upload random objects → May give unpredictable results (model trained on faces)

**Expected Output:**
```
Prediction: Real ✓
Confidence: 98.76%
Probabilities:
  Fake: 1.24%
  Real: 98.76%
```

### 3. Model Analysis with AUC

**Steps:**
1. Locate the "Model Analysis" section
2. Drag and drop a `.pth` model checkpoint file
3. Wait for analysis (may take 30-60 seconds for AUC calculation)
4. View comprehensive results:
   - AUC score (Area Under ROC Curve)
   - Accuracy on test set
   - Confusion matrix
   - ROC curve visualization
   - Model metadata (epoch, architecture)

**What gets calculated:**
- **AUC Score**: Measures model's ability to distinguish between Real and Fake
  - 1.0 = Perfect classifier
  - 0.5 = Random classifier
  - Typical good models: 0.95 - 0.99+
- **Confusion Matrix**: Shows True Positives, False Positives, True Negatives, False Negatives
- **ROC Curve**: True Positive Rate vs False Positive Rate

**Expected Output:**
```
AUC Score: 0.9945
Accuracy: 98.23%
Test Samples: 1000

Confusion Matrix:
            Predicted
           Fake  Real
Actual Fake  450    12
       Real    8   530
```

### 4. Comparing Multiple Models

**Steps:**
1. Upload first model → Note the AUC score
2. Click "Upload Another Model"
3. Upload second model → Compare AUC scores
4. The model with higher AUC performs better

**Use Cases:**
- Compare different training epochs
- Compare Stage 1 vs Stage 2 models
- Compare different hyperparameter configurations

## UI Features

### Drag-and-Drop Zones
- **Active State**: Zone highlights when dragging a file over it
- **Processing State**: Shows spinner and status text
- **Success State**: Displays results immediately
- **Error State**: Shows error message with details

### Responsive Design
- Works on desktop, tablet, and mobile devices
- Adaptive layout for different screen sizes
- Touch-friendly interface

### Visual Feedback
- Color-coded predictions (Green for Real, Red for Fake)
- Animated progress bars
- Interactive charts (ROC curve)
- Hover effects on interactive elements

## Troubleshooting

### Backend Issues

**"Backend API Not Available" Error**
- Ensure Flask server is running on port 5000
- Check backend terminal for error messages
- Verify Python dependencies are installed

**"Model not loaded" Error**
- Backend starts with untrained model if no checkpoint exists
- Upload a trained model via the Model Analysis section
- Or place a checkpoint at `ViXNet/checkpoints/best_model.pth`

**"Dataset not available" Warning**
- AUC calculation requires test dataset
- Ensure `CNN + Transformer/Dataset/Test/` exists with Fake/ and Real/ subdirectories
- Image inference still works without test dataset

### Frontend Issues

**Page doesn't load**
- Check if React dev server started on port 3000
- Look for errors in browser console (F12)
- Try clearing browser cache

**Slow inference**
- First inference may be slower (model initialization)
- Subsequent inferences should be fast (~1-2 seconds)
- Large images may take longer to upload

### Performance Tips

**For faster inference:**
- Use GPU-enabled PyTorch if available
- Reduce image size before upload (model resizes to 224x224 anyway)
- Close other applications to free up resources

**For faster AUC calculation:**
- Reduce test dataset size (if very large)
- Use GPU for faster computation
- AUC calculation is one-time per model upload

## API Testing

You can also test the API directly using curl or Postman:

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Model Info
```bash
curl http://localhost:5000/api/model-info
```

### Predict Image
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/api/predict
```

### Analyze Model
```bash
curl -X POST -F "model=@path/to/model.pth" http://localhost:5000/api/analyze-model
```

## Screenshots

When testing, you can take screenshots of:
1. Model architecture visualization
2. Image inference with results
3. Model analysis with AUC metrics
4. ROC curve chart
5. Confusion matrix

## Expected Performance

**With a trained model:**
- AUC: 0.95 - 0.99+
- Accuracy: 95% - 99%+
- Inference time: 30-100ms per image (GPU)
- AUC calculation: 30-120 seconds (depends on test set size)

**With untrained model:**
- Random predictions (~50% accuracy)
- AUC ~0.5 (random classifier)
- Still useful for UI/functionality demo

## Next Steps

After demonstrating the web app:
1. Train a full model using `ViXNet/train.py`
2. Upload the trained checkpoint via the web interface
3. Test with real images from the dataset
4. Compare different model versions
5. Share the web interface with stakeholders

## Notes

- The web app is designed for demonstration and analysis purposes
- For production deployment, consider:
  - Adding authentication
  - Using production WSGI server (Gunicorn)
  - Serving React build via Flask or nginx
  - Adding rate limiting
  - Implementing proper error handling
  - Adding model versioning

## Support

For issues or questions:
1. Check the main README: `ViXNet/web_app/README.md`
2. Review API documentation in the README
3. Check Flask backend logs: `backend/backend.log`
4. Check browser console for frontend errors (F12)
