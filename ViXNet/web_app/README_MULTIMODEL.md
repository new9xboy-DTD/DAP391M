# Multi-Model Web Application - Quick Start

## 🆕 What's New

This web application now supports:
- **Multiple model architectures**: ViXNet, Xception Only, ViT Only
- **Dataset selection**: Choose which dataset to evaluate your model on
- **On-demand loading**: No model loaded by default (saves memory)
- **Automatic model detection**: Uploads automatically detect model type

## 🚀 Quick Start

### 1. Start Backend
```bash
cd backend
python app.py
```

Backend will run on http://localhost:5000

### 2. Start Frontend
```bash
cd frontend
npm install  # Only needed first time
npm start
```

Frontend will open on http://localhost:3000

### 3. Use the Application

1. **Upload a model**:
   - Drag & drop a `.pth` file
   - Select a dataset from dropdown
   - Click "Analyze Model"
   - Wait for AUC calculation

2. **Test with images**:
   - After model is loaded, upload images
   - Get instant Real/Fake predictions

## 📋 Requirements

### Backend
- Python 3.8+
- PyTorch
- Flask
- See `requirements.txt` in project root

### Frontend
- Node.js 14+
- React
- See `package.json` for dependencies

## 🔧 Configuration

### Adding Datasets

Edit `../../config.py` and add to `DATASETS`:

```python
DATASETS = {
    'default': {
        'name': 'Default Dataset',
        'path': '/path/to/dataset',
        'train': '/path/to/dataset/Train',
        'val': '/path/to/dataset/Validation',
        'test': '/path/to/dataset/Test'
    },
    'my_dataset': {
        'name': 'My Custom Dataset',
        'path': '/path/to/my/dataset',
        'train': '/path/to/my/dataset/Train',
        'val': '/path/to/my/dataset/Validation',
        'test': '/path/to/my/dataset/Test'
    }
}
```

Restart backend after changes.

## 📡 API Endpoints

### New/Updated Endpoints

- `GET /api/datasets` - List available datasets
- `POST /api/analyze-model` - Upload model with dataset selection
  - Form data: `model` (file), `dataset` (string)
- `POST /api/calculate-auc` - Calculate AUC with dataset selection
  - JSON body: `{"dataset": "default"}`

### Existing Endpoints

- `GET /api/health` - Health check
- `GET /api/model-info` - Get current model info
- `POST /api/predict` - Predict image (Real/Fake)

## 📚 Documentation

For detailed information, see:
- [`../../MULTI_MODEL_IMPLEMENTATION.md`](../../MULTI_MODEL_IMPLEMENTATION.md) - Technical details
- [`../../USER_GUIDE.md`](../../USER_GUIDE.md) - User guide (English)
- [`../../HUONG_DAN_TIENG_VIET.md`](../../HUONG_DAN_TIENG_VIET.md) - User guide (Vietnamese)

## 🐛 Troubleshooting

### Backend won't start
- Check Python dependencies: `pip install -r ../../requirements.txt`
- Verify PyTorch is installed correctly
- Check port 5000 is available

### Frontend won't connect
- Ensure backend is running on port 5000
- Check CORS is enabled in backend
- Clear browser cache

### Model upload fails
- Verify model file is valid PyTorch checkpoint
- Check model has `model_state_dict` key
- Ensure dataset paths are correct in config

### Dataset not found
- Check dataset path in `config.py`
- Verify folder structure: `Test/{Fake,Real}/`
- Ensure images are in supported formats

## 🔍 Testing Without Models

You can test the UI without models/datasets:
1. Start backend and frontend
2. Open http://localhost:3000
3. See welcome screen with instructions
4. Try uploading any `.pth` file (will fail gracefully)

## 🎯 Key Changes from Previous Version

| Before | After |
|--------|-------|
| Auto-loads ViXNet on startup | No model loaded by default |
| Only supports ViXNet | Supports ViXNet, Xception, ViT |
| Fixed dataset | Selectable datasets |
| Model required to enter | Can enter without model |
| Manual model type specification | Automatic model detection |

## 📝 Notes

- Model analysis can take 1-5 minutes depending on dataset size
- Image inference is near-instant once model is loaded
- Models are automatically detected from checkpoint structure
- Dataset selection happens before model analysis
- Multiple models can be tested sequentially (one at a time)

## 🔐 Security Note

⚠️ Only upload models from trusted sources. Model files can contain executable code.

## 💡 Tips

1. **Test multiple models**: Upload different models on same dataset to compare
2. **Try different datasets**: Test same model on different datasets
3. **Save results**: Note down AUC scores for each combination
4. **Clear memory**: Use "Upload Another Model" to replace current model

## 🆘 Support

For issues or questions:
1. Check documentation files in project root
2. Review backend console logs for errors
3. Check browser console for frontend errors
4. Verify paths and configurations in `config.py`

---

**Version**: 2.0 (Multi-Model Support)  
**Last Updated**: 2026-01-20
