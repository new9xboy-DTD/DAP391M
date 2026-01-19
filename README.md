# DAP391M

## Dataset

- Kaggle: [https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)
- This dataset contains manipulated images and real images. The manipulated images are the faces which are created by various means. The source for this dataset was [https://zenodo.org/record/5528418#.YpdlS2hBzDd](https://zenodo.org/record/5528418#.YpdlS2hBzDd)
  this dataset was processed as our will to get maximum outcome out of these images. Each image is a 256 X 256 jpg image of human face either real or fake
- Kaggle: https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set

## Project Structure

This repository contains multiple deepfake detection implementations:

### 1. ViXNet (Recommended)

Vision Transformer with Xception Network - State-of-the-art deepfake detection with web interface.

**Location:** `ViXNet/`

**Features:**

- 2-branch fusion architecture (Xception + ViT)
- 2-stage training strategy
- Web application for model visualization and inference
- Comprehensive documentation

**See:** [ViXNet/README.md](ViXNet/README.md)

### 2. CNN + Transformer

Basic CNN and Transformer implementation for deepfake detection.

**Location:** `CNN + Transformer/`

### 3. Sequence Diffusion GCN

Advanced implementation using diffusion models and graph convolutional networks.

**Location:** `Sequence_Diffusion_GCN/`

### 4. VGG16 Fine-tuning

Transfer learning approach using VGG16.

**Location:** `sky-787770/`

## Quick Start

### For ViXNet Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Start web application
cd ViXNet/web_app
./start.sh  # or start.bat on Windows
```

Access at: http://localhost:3000

**Full documentation:** [ViXNet/web_app/README.md](ViXNet/web_app/README.md)

## Requirements

- Python 3.8+
- **PyTorch 2.6.0+** (updated for security)
- Node.js 16+ (for web app)
- See `requirements.txt` for complete dependencies

## Security

This project has been updated to address critical PyTorch vulnerabilities:

- Heap buffer overflow (fixed in PyTorch 2.2.0)
- Use-after-free vulnerability (fixed in PyTorch 2.2.0)
- Remote code execution via torch.load (fixed in PyTorch 2.6.0)

**See [SECURITY.md](SECURITY.md) for complete security information and migration guide.**

## Documentation

- **ViXNet Documentation:** [ViXNet/README.md](ViXNet/README.md)
- **Web App Guide:** [ViXNet/web_app/README.md](ViXNet/web_app/README.md)
- **Quick Start:** [ViXNet/web_app/QUICKSTART.md](ViXNet/web_app/QUICKSTART.md)

## License

This project is for educational and research purposes.
