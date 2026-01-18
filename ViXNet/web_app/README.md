# ViXNet Web Application

A ReactJS web interface for visualizing and interacting with the ViXNet deepfake detection model.

## Features

✅ **Model Architecture Visualization** - Interactive display of ViXNet's two-branch architecture (Xception + ViT)  
✅ **Image Inference** - Drag-and-drop image upload for real-time deepfake detection  
✅ **Model Analysis** - Upload custom model checkpoints and analyze their performance  
✅ **AUC Calculation** - Automatic calculation of Area Under the ROC Curve on test dataset  
✅ **ROC Curve Visualization** - Interactive charts showing model performance  
✅ **Confusion Matrix** - Visual representation of classification results  
✅ **Real-time Results** - Instant feedback with confidence scores and probabilities

## Architecture

### Backend (Flask API)
- **Framework**: Flask with CORS support
- **Endpoints**:
  - `GET /api/health` - Health check
  - `GET /api/model-info` - Get current model information
  - `POST /api/predict` - Predict if an image is real or fake
  - `POST /api/analyze-model` - Upload and analyze a model file
  - `POST /api/calculate-auc` - Calculate AUC for the current model

### Frontend (React)
- **Framework**: React 18
- **Components**:
  - `App.js` - Main application component
  - `ImageDropzone.js` - Drag-and-drop image upload
  - `ModelDropzone.js` - Drag-and-drop model upload
  - `ModelArchitecture.js` - Architecture visualization
  - `ResultsDisplay.js` - Results with charts and metrics
- **Libraries**:
  - `axios` - HTTP requests
  - `recharts` - Data visualization

## Prerequisites

### Backend Requirements
- Python 3.8+
- PyTorch 2.1.2+
- Flask 3.0.0+
- Other dependencies (see `backend/requirements.txt`)

### Frontend Requirements
- Node.js 16+ and npm

### Dataset
- Test dataset must be available at `CNN + Transformer/Dataset/Test/` for AUC calculation
- Dataset structure:
  ```
  CNN + Transformer/Dataset/
  └── Test/
      ├── Fake/
      └── Real/
  ```

## Installation

### 1. Install Backend Dependencies

```bash
cd ViXNet/web_app/backend
pip install -r requirements.txt
```

Or use the main repository's requirements:
```bash
cd /path/to/DAP391M
pip install -r requirements.txt
pip install Flask flask-cors
```

### 2. Install Frontend Dependencies

```bash
cd ViXNet/web_app/frontend
npm install
```

## Usage

### Step 1: Start the Backend Server

```bash
cd ViXNet/web_app/backend
python app.py
```

The backend will start on `http://localhost:5000`

**Note**: The backend will automatically try to load the default model from `ViXNet/checkpoints/best_model.pth`. If no checkpoint exists, it will use an untrained model.

### Step 2: Start the Frontend Development Server

In a new terminal:

```bash
cd ViXNet/web_app/frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

### Step 3: Use the Application

1. **View Model Architecture**: The architecture visualization is displayed automatically
2. **Test Image Inference**: 
   - Drag and drop an image (or click to browse)
   - View the prediction (Real/Fake) with confidence scores
3. **Analyze Model**:
   - Drag and drop a `.pth` model file
   - Wait for analysis (includes AUC calculation on test set)
   - View detailed metrics, ROC curve, and confusion matrix

## Features in Detail

### Image Inference
- Supports JPG, PNG, JPEG formats
- Automatic preprocessing (resize to 224x224, normalization)
- Real-time prediction with confidence scores
- Visual probability breakdown for both classes

### Model Analysis
- Upload custom ViXNet checkpoints (.pth files)
- Automatic model validation
- AUC calculation on test dataset
- ROC curve visualization
- Confusion matrix display
- Model metadata (epoch, accuracy, architecture details)

### Model Visualization
- Two-branch architecture diagram
- Feature dimensions at each stage
- Current model information and metrics

## API Documentation

### GET /api/health
Check if the backend is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET /api/model-info
Get information about the currently loaded model.

**Response:**
```json
{
  "loaded": true,
  "epoch": 10,
  "metrics": {
    "accuracy": 0.9823,
    "precision": 0.9810,
    "recall": 0.9835,
    "f1": 0.9823
  },
  "architecture": {
    "name": "ViXNet",
    "xception_dim": 2048,
    "vit_dim": 192,
    "fusion_dim": 512,
    "num_classes": 2
  }
}
```

### POST /api/predict
Predict if an uploaded image is real or fake.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` file

**Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.9876,
  "probabilities": {
    "Fake": 0.0124,
    "Real": 0.9876
  }
}
```

### POST /api/analyze-model
Upload and analyze a model checkpoint.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `model` file (.pth)

**Response:**
```json
{
  "success": true,
  "model_info": {
    "loaded": true,
    "epoch": 15,
    "metrics": {...},
    "auc_results": {
      "auc": 0.9945,
      "accuracy": 0.9823,
      "confusion_matrix": [[450, 12], [8, 530]],
      "roc_curve": {
        "fpr": [...],
        "tpr": [...],
        "thresholds": [...]
      },
      "num_samples": 1000
    }
  }
}
```

### POST /api/calculate-auc
Calculate AUC for the currently loaded model.

**Response:**
```json
{
  "auc": 0.9945,
  "accuracy": 0.9823,
  "confusion_matrix": [[450, 12], [8, 530]],
  "roc_curve": {...},
  "num_samples": 1000
}
```

## Building for Production

### Security Considerations

⚠️ **Important Security Notes:**

1. **Model Upload Security**: The application uses `torch.load()` with `weights_only=False` for checkpoint compatibility. **Only upload model files from trusted sources** as malicious .pth files could execute arbitrary code.

2. **Production Deployment**: For production environments:
   - Add authentication and authorization
   - Implement rate limiting
   - Use HTTPS for all communications
   - Validate and sanitize all file uploads
   - Consider using `weights_only=True` with appropriate checkpoint validation
   - Run backend with restricted permissions
   - Implement proper CORS policies (not wildcard)

3. **File Storage**: Uploaded models are stored temporarily. Implement proper file cleanup and storage management for production.

### Backend
The Flask backend can be deployed using production WSGI servers like Gunicorn:

```bash
cd ViXNet/web_app/backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend
Build the React app for production:

```bash
cd ViXNet/web_app/frontend
npm run build
```

The production build will be in the `build/` directory. Serve it with any static file server or integrate with the Flask backend.

## Troubleshooting

### Backend Issues

**Error: Model not loaded**
- Ensure a trained model checkpoint exists at `ViXNet/checkpoints/best_model.pth`
- Or upload a model through the web interface

**Error: Dataset not found**
- Ensure test dataset is at `../CNN + Transformer/Dataset/Test/`
- AUC calculation requires the test dataset

**Error: CUDA out of memory**
- The backend automatically uses CPU if GPU is unavailable
- Reduce batch size in `config.py` if needed

### Frontend Issues

**Error: Cannot connect to backend**
- Ensure Flask backend is running on port 5000
- Check CORS settings in `backend/app.py`

**Error: npm install fails**
- Try deleting `node_modules` and `package-lock.json`
- Run `npm install` again

## Development

### Running Tests
```bash
# Backend tests (if available)
cd ViXNet/web_app/backend
python -m pytest

# Frontend tests
cd ViXNet/web_app/frontend
npm test
```

### File Structure
```
web_app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # Temporary upload storage
├── frontend/
│   ├── public/
│   │   └── index.html     # HTML template
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── utils/         # API utilities
│   │   ├── App.js         # Main app component
│   │   ├── App.css        # App styles
│   │   ├── index.js       # Entry point
│   │   └── index.css      # Global styles
│   └── package.json       # Node dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please ensure:
- Backend code follows PEP 8 style guidelines
- Frontend code uses consistent React patterns
- All new features include documentation
- Test your changes before submitting

## License

This web application is part of the ViXNet project for educational and research purposes.

## Credits

Based on the ViXNet paper published in Expert Systems with Applications (Q1 Journal).
