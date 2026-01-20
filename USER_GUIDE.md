# Multi-Model Web Application - User Guide

## 🎯 Main Features

### 1. **No Default Model Loading**
When you first open the web application, no model is loaded. This saves memory and allows you to choose which model to use.

### 2. **Multi-Model Support**
The application now supports three types of deepfake detection models:
- **ViXNet**: Vision Transformer + Xception (hybrid approach)
- **Xception Only**: CNN-based approach using Xception
- **ViT Only**: Transformer-based approach using Vision Transformer

### 3. **Dataset Selection**
You can select which dataset to use for evaluating your model's performance. This allows testing the same model on different datasets to compare results.

## 📖 Step-by-Step Usage

### Step 1: Start the Application

1. **Start Backend Server**:
   ```bash
   cd ViXNet/web_app/backend
   python app.py
   ```
   
   You should see:
   ```
   ======================================================================
   🚀 Starting Multi-Model Deepfake Detection Web API
   ======================================================================
   
   📦 Supported Model Types:
      • ViXNet (Vision Transformer + Xception)
      • Xception Only
      • ViT Only
   
   📁 Available Datasets:
      • Default Dataset (default)
   
   🌐 Server running on http://localhost:5000
   ```

2. **Start Frontend** (in a new terminal):
   ```bash
   cd ViXNet/web_app/frontend
   npm start
   ```
   
   Your browser will open to http://localhost:3000

### Step 2: Upload a Model

1. **Locate the "Model Upload & Analysis" section**
   - This is the first section you'll see when no model is loaded

2. **Upload your model file**:
   - Drag and drop a `.pth` or `.pt` file into the dropzone
   - OR click the dropzone to browse for a file
   
3. **Select a dataset**:
   - After dropping the file, a dropdown will appear
   - Select which dataset to use for evaluation
   - The default dataset is pre-selected

4. **Click "Analyze Model"**:
   - The analysis will start automatically
   - A loading spinner will show progress
   - This may take 1-5 minutes depending on dataset size

5. **View Results**:
   - AUC (Area Under Curve) score
   - Accuracy percentage
   - Confusion matrix
   - ROC curve data
   - Model architecture information

### Step 3: Test with Images

After your model is loaded and analyzed:

1. **Find the "Image Inference" section**
   - This section appears only after a model is loaded

2. **Upload an image**:
   - Drag and drop an image (JPG, PNG, etc.)
   - OR click to browse for an image

3. **View Prediction**:
   - **Prediction**: "Real" or "Fake"
   - **Confidence**: Percentage confidence of prediction
   - **Probabilities**: Breakdown of Real vs Fake probabilities

### Step 4: Upload Another Model (Optional)

1. Click "Upload Another Model" button
2. Repeat steps from Step 2
3. The new model will replace the current one

## 🎨 UI Components

### Welcome Screen (No Model Loaded)
```
┌─────────────────────────────────────────┐
│  🧠 Multi-Model Deepfake Detection      │
│  Support for ViXNet, Xception, ViT      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  🔧 Model Upload & Analysis             │
│  ┌───────────────────────────────────┐  │
│  │  🔧                               │  │
│  │  Drag & drop a model file        │  │
│  │  Supports: .pth, .pt             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  👋 Welcome!                            │
│  To get started:                        │
│  1. Upload a trained model file         │
│  2. Select a dataset for evaluation     │
│  3. Wait for analysis                   │
│  4. Upload images for detection         │
└─────────────────────────────────────────┘
```

### After File Upload (Before Analysis)
```
┌─────────────────────────────────────────┐
│  🔧 Model Upload & Analysis             │
│  ┌───────────────────────────────────┐  │
│  │  📦                               │  │
│  │  model_vixnet.pth                 │  │
│  │  Ready to analyze                 │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Select Dataset for Evaluation:         │
│  ┌───────────────────────────────────┐  │
│  │  Default Dataset            ▼     │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      Analyze Model                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Model Loaded - Full Interface
```
┌─────────────────────────────────────────┐
│  📊 Current Model Architecture          │
│  • Name: ViXNet                         │
│  • Type: vixnet                         │
│  • Status: Loaded ✅                    │
│  • AUC: 0.9543                         │
│  • Accuracy: 92.35%                    │
└─────────────────────────────────────────┘

┌──────────────────────┬──────────────────┐
│  🔧 Model Upload     │  🖼️ Image        │
│  [Loaded: ✅]        │  Inference        │
│  ┌────────────────┐  │  ┌──────────────┐│
│  │ Upload Another │  │  │ 📸           ││
│  │ Model          │  │  │ Drag & drop  ││
│  └────────────────┘  │  │ an image     ││
│                      │  └──────────────┘│
└──────────────────────┴──────────────────┘

┌─────────────────────────────────────────┐
│  📈 Results                             │
│  Prediction: Real ✓                     │
│  Confidence: 95.43%                     │
│  • Fake: 4.57%                         │
│  • Real: 95.43%                        │
└─────────────────────────────────────────┘
```

## 🔧 Configuration

### Adding New Datasets

Edit `ViXNet/config.py`:

```python
DATASETS = {
    'default': {
        'name': 'Default Dataset',
        'path': '/path/to/default/dataset',
        'train': '/path/to/default/dataset/Train',
        'val': '/path/to/default/dataset/Validation',
        'test': '/path/to/default/dataset/Test'
    },
    'celeb_df': {
        'name': 'Celeb-DF Dataset',
        'path': '/path/to/celeb-df',
        'train': '/path/to/celeb-df/Train',
        'val': '/path/to/celeb-df/Validation',
        'test': '/path/to/celeb-df/Test'
    },
    'faceforensics': {
        'name': 'FaceForensics++ Dataset',
        'path': '/path/to/faceforensics',
        'train': '/path/to/faceforensics/Train',
        'val': '/path/to/faceforensics/Validation',
        'test': '/path/to/faceforensics/Test'
    }
}
```

After adding datasets:
1. Restart the backend server
2. The new datasets will appear in the dropdown automatically

### Dataset Folder Structure

Each dataset must follow this structure:
```
Dataset/
├── Train/
│   ├── Fake/
│   │   ├── fake_image1.jpg
│   │   ├── fake_image2.jpg
│   │   └── ...
│   └── Real/
│       ├── real_image1.jpg
│       ├── real_image2.jpg
│       └── ...
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## ⚡ Tips & Tricks

### Performance
- **Model analysis**: Takes 1-5 minutes depending on test dataset size
- **Image inference**: Almost instant (< 1 second)
- **Memory**: No model loaded = minimal memory usage

### Best Practices
1. **Test on multiple datasets**: Upload the same model and test on different datasets to compare generalization
2. **Save model info**: Note down AUC scores for different models/datasets
3. **Clear model**: Use "Upload Another Model" to free up memory before loading a new one

### Troubleshooting
- **"Model not loaded" error**: You need to upload and analyze a model first
- **"Dataset not available" error**: Check dataset paths in config.py
- **Slow analysis**: Large test datasets take longer; this is normal
- **Backend not responding**: Ensure Flask server is running on port 5000

## 🚀 Workflow Example

**Scenario**: Testing multiple models on the same dataset

1. **Load Model 1** (ViXNet)
   - Upload `vixnet_model.pth`
   - Select "Default Dataset"
   - Analyze → Get AUC: 0.954, Accuracy: 92.3%

2. **Test with images**
   - Upload test images
   - Note predictions and confidence

3. **Load Model 2** (Xception Only)
   - Click "Upload Another Model"
   - Upload `xception_model.pth`
   - Select "Default Dataset"
   - Analyze → Get AUC: 0.931, Accuracy: 89.7%

4. **Compare results**
   - ViXNet performs better on this dataset
   - Both models can be tested with the same images for comparison

## 📊 Understanding Results

### AUC (Area Under Curve)
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 0.9 - 1.0: Excellent
  - 0.8 - 0.9: Good
  - 0.7 - 0.8: Fair
  - Below 0.7: Poor

### Accuracy
- Percentage of correct predictions
- Higher is better
- Compare with AUC for balanced view

### Confidence
- How certain the model is about its prediction
- 95%+ = Very confident
- 70-95% = Confident
- Below 70% = Uncertain

## 🛠️ Advanced Features

### API Direct Access

You can also interact with the API directly:

```bash
# List datasets
curl http://localhost:5000/api/datasets

# Upload model with dataset selection
curl -X POST \
  -F "model=@model.pth" \
  -F "dataset=default" \
  http://localhost:5000/api/analyze-model

# Predict image
curl -X POST \
  -F "image=@test_image.jpg" \
  http://localhost:5000/api/predict
```

### Integration

The API can be integrated into other applications:
- Mobile apps
- Desktop applications
- Batch processing scripts
- Automated testing pipelines

---

**Note**: This application is designed for research and testing purposes. Always validate results and consider ethical implications when working with deepfake detection systems.
