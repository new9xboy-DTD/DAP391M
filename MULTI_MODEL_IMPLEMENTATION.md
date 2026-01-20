# Multi-Model Support and Dataset Selection - Implementation Guide

## Overview
This implementation adds support for multiple model architectures and dataset selection in the web application, allowing users to:
- Load different model types (ViXNet, Xception Only, ViT Only)
- Select datasets for model evaluation
- Upload models without requiring a default model on startup

## Changes Made

### 1. Backend Changes

#### New File: `ViXNet/model_factory.py`
- **Purpose**: Factory pattern for creating and loading different model architectures
- **Classes**:
  - `XceptionOnly`: Xception-only CNN model for deepfake detection
  - `ViTOnly`: Vision Transformer-only model for deepfake detection
- **Functions**:
  - `detect_model_type(checkpoint)`: Automatically detect model type from checkpoint
  - `create_model(model_type, pretrained, num_classes)`: Create model instance by type
  - `load_model_from_checkpoint(checkpoint_path, device)`: Load model with automatic type detection

#### Updated: `ViXNet/config.py`
- **New Configuration**: `DATASETS` dictionary for multiple dataset configurations
  ```python
  DATASETS = {
      'default': {
          'name': 'Default Dataset',
          'path': '...',
          'train': '...',
          'val': '...',
          'test': '...'
      },
      # Add more datasets here
  }
  ```
- **New Methods**:
  - `get_dataset_config(dataset_key)`: Get dataset configuration by key
  - `list_available_datasets()`: List all available datasets

#### Updated: `ViXNet/web_app/backend/app.py`
- **Removed**: Default model loading on startup
- **Updated**: Model loading to use `model_factory`
- **New Endpoints**:
  - `GET /api/datasets`: List available datasets
  - Updated `POST /api/analyze-model`: Now accepts `dataset` parameter in form data
  - Updated `POST /api/calculate-auc`: Now accepts `dataset` parameter in JSON body
- **Updated Functions**:
  - `calculate_auc_on_test_set(model, dataset_key)`: Support dataset selection
  - `create_test_loader(dataset_key, batch_size, num_workers)`: Create test loader for specific dataset

### 2. Frontend Changes

#### Updated: `ViXNet/web_app/frontend/src/utils/api.js`
- **New Function**: `getDatasets()`: Fetch available datasets from API
- **Updated Functions**:
  - `analyzeModel(modelFile, dataset)`: Accept dataset parameter
  - `calculateAUC(dataset)`: Accept dataset parameter

#### Updated: `ViXNet/web_app/frontend/src/components/ModelDropzone.js`
- **New Features**:
  - Dataset selector dropdown
  - Two-step upload process: select file → choose dataset → analyze
  - Load available datasets on component mount
- **New State**:
  - `datasets`: List of available datasets
  - `selectedDataset`: Currently selected dataset
  - `pendingFile`: File waiting for dataset selection

#### Updated: `ViXNet/web_app/frontend/src/components/ModelDropzone.css`
- **New Styles**:
  - `.dataset-selector`: Container for dataset selection UI
  - `.dataset-dropdown`: Dropdown for dataset selection
  - `.analyze-button`: Button to trigger analysis after dataset selection

#### Updated: `ViXNet/web_app/frontend/src/App.js`
- **Removed**: Initial model loading requirement
- **Updated**: Show image upload only after model is loaded
- **New**: Welcome instructions when no model is loaded
- **Updated**: Header and footer text to reflect multi-model support

#### Updated: `ViXNet/web_app/frontend/src/App.css`
- **New Styles**: `.info-section` for welcome/instruction display

## Usage Guide

### Starting the Application

1. **Start Backend**:
   ```bash
   cd ViXNet/web_app/backend
   python app.py
   ```

2. **Start Frontend** (in separate terminal):
   ```bash
   cd ViXNet/web_app/frontend
   npm start
   ```

### Using the Application

1. **Upload a Model**:
   - Navigate to the "Model Upload & Analysis" section
   - Drag and drop a `.pth` or `.pt` model file
   - Select a dataset from the dropdown (default is pre-selected)
   - Click "Analyze Model" button
   - Wait for the analysis to complete (shows AUC and accuracy)

2. **Test with Images**:
   - After model is loaded, the "Image Inference" section becomes available
   - Drag and drop an image to detect if it's real or fake
   - View prediction results with confidence scores

### Adding New Datasets

Edit `ViXNet/config.py` and add to the `DATASETS` dictionary:

```python
DATASETS = {
    'default': {...},
    'my_new_dataset': {
        'name': 'My New Dataset',
        'path': '/path/to/dataset',
        'train': '/path/to/dataset/Train',
        'val': '/path/to/dataset/Validation',
        'test': '/path/to/dataset/Test'
    }
}
```

The new dataset will automatically appear in the dropdown after restarting the backend.

## Model Architecture Support

### ViXNet (Vision Transformer + Xception)
- Combined architecture with CNN and Transformer branches
- Best for comprehensive feature extraction
- Checkpoint should contain: `xception_branch`, `vit_branch`, `fusion`, `classifier`

### Xception Only
- CNN-only architecture using Xception
- Faster inference, focuses on spatial features
- Checkpoint should contain: `xception`, `classifier`

### ViT Only
- Transformer-only architecture using Vision Transformer
- Focuses on patch-wise attention patterns
- Checkpoint should contain: `vit`, `classifier`

## API Endpoints

### New Endpoints
- `GET /api/datasets` - List available datasets
  ```json
  {
    "datasets": [
      {"key": "default", "name": "Default Dataset", "path": "..."}
    ],
    "count": 1
  }
  ```

### Updated Endpoints
- `POST /api/analyze-model` - Upload model with dataset selection
  - Form data: `model` (file), `dataset` (string, optional, default: "default")
  
- `POST /api/calculate-auc` - Calculate AUC with dataset selection
  - JSON body: `{"dataset": "default"}`

## Technical Details

### Model Type Detection
The `detect_model_type()` function automatically identifies model architecture by checking for:
- ViXNet: presence of `xception_branch`, `vit_branch`, and `fusion` layers
- Xception: presence of `xception` layers or CNN-specific layers
- ViT: presence of `vit` layers or transformer-specific layers (e.g., `blocks`, `patch_embed`)

### State Management
- Backend: No default model is loaded on startup (reduces memory usage)
- Frontend: Model info is fetched only if a model is already loaded
- State is cleared when uploading a new model

## Testing

Run the structural test:
```bash
python test_model_factory.py
```

This verifies:
- Config changes are correct
- Model factory has all required classes and functions
- Backend API has new endpoints and imports
- Default model loading is removed

## Future Enhancements

1. **Add More Model Architectures**:
   - Create new model classes in `model_factory.py`
   - Add detection logic to `detect_model_type()`
   - Update `create_model()` factory function

2. **Add More Datasets**:
   - Simply add entries to `Config.DATASETS` dictionary
   - No frontend changes needed

3. **Model Comparison**:
   - Allow loading multiple models simultaneously
   - Compare performance across models on same dataset

4. **Dataset Upload**:
   - Allow users to upload custom datasets
   - Store in temporary directory for evaluation

## Security Considerations

⚠️ **Important**: The current implementation uses `torch.load(..., weights_only=False)` for checkpoint loading. This is necessary for compatibility with existing checkpoints but can execute arbitrary code. 

**Recommendations**:
- Only load models from trusted sources
- In production, consider using `weights_only=True` with proper checkpoint format validation
- Implement user authentication if deploying publicly
- Add file size limits for uploads
- Sanitize file paths to prevent directory traversal

## Troubleshooting

### Model fails to load
- Check that the checkpoint format is correct (has `model_state_dict` key)
- Verify the model architecture matches the checkpoint
- Ensure PyTorch version compatibility

### Dataset not found
- Verify the dataset path in `Config.DATASETS`
- Check that the directory exists and contains Test/Fake and Test/Real subdirectories
- Ensure proper folder structure: `Test/{Fake,Real}/`

### AUC calculation fails
- Dataset must have at least some images in both Fake and Real classes
- Check that images are in supported formats (JPG, PNG, etc.)
- Verify proper folder structure with ImageFolder-compatible layout

## Contact & Support

For issues or questions:
- Check the implementation files for inline comments
- Review the API endpoint documentation in backend code
- Test with the provided test script first
