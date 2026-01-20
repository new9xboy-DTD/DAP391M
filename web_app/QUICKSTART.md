# Quick Start Guide - ViXNet Web Application

## 🎯 What You're Building

A web-based interface to:
- Visualize the ViXNet deepfake detection model architecture
- Upload images and detect if they're real or fake
- Upload and analyze model checkpoints with AUC metrics
- View ROC curves and confusion matrices

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies

```bash
# Install backend dependencies
cd ViXNet/web_app/backend
pip install Flask flask-cors

# Note: PyTorch and other ML dependencies should already be installed
# from the main requirements.txt. If not:
pip install torch torchvision timm scikit-learn pillow numpy

# Install frontend dependencies
cd ../frontend
npm install
```

### Step 2: Start the Application

**Option A - Using startup script (Linux/Mac):**
```bash
cd ViXNet/web_app
./start.sh
```

**Option B - Using startup script (Windows):**
```bash
cd ViXNet\web_app
start.bat
```

**Option C - Manual start (any OS):**

Terminal 1:
```bash
cd ViXNet/web_app/backend
python app.py
```

Terminal 2:
```bash
cd ViXNet/web_app/frontend
npm start
```

### Step 3: Open Browser

The app will automatically open at: **http://localhost:3000**

## 🎨 First Usage

### Try Image Inference
1. Find any image file (JPG, PNG)
2. Drag it into the "Image Inference" box
3. See the prediction (Real/Fake) with confidence

### Try Model Analysis (Optional)
1. If you have a trained `.pth` model file
2. Drag it into the "Model Analysis" box
3. View comprehensive metrics including AUC score

## 📋 Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend running on port 5000
- [ ] Frontend running on port 3000
- [ ] Browser opened to localhost:3000

## ⚠️ Troubleshooting

**Backend won't start:**
- Check Python version: `python --version`
- Install missing packages: `pip install Flask flask-cors`

**Frontend won't start:**
- Check Node version: `node --version`
- Delete `node_modules` and run `npm install` again

**"Backend API Not Available":**
- Make sure backend is running on port 5000
- Check backend terminal for errors

**"Dataset not available" when analyzing model:**
- This is normal if you don't have test dataset
- Image inference still works
- To fix: Place test images at `CNN + Transformer/Dataset/Test/`

## 🚀 Next Steps

1. **Train a Model:**
   ```bash
   cd ViXNet
   python train.py
   ```

2. **Upload Your Trained Model:**
   - After training, upload `ViXNet/checkpoints/best_model.pth` via the web UI

3. **Test with Real Images:**
   - Use images from your dataset to test predictions

4. **Compare Models:**
   - Train multiple versions with different hyperparameters
   - Upload and compare their AUC scores

## 📚 More Information

- Full documentation: `ViXNet/web_app/README.md`
- Demo guide: `ViXNet/web_app/DEMO.md`
- Visual guide: `ViXNet/web_app/VISUAL_GUIDE.md`
- API docs: In the main README

## 💡 Tips

- First inference may be slow (model initialization)
- Subsequent inferences are fast (~1-2 seconds)
- AUC calculation takes 30-60 seconds (one-time per model)
- Works best with face images (model is trained on faces)
- Supports JPG, PNG, JPEG formats

## ✅ Success!

If you see the web interface with the model architecture visualization, you're all set! 🎉

Start by dragging an image into the inference zone to see it in action.
