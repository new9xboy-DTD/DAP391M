# ViXNet: Vision Transformer with Xception Network

Implementation of ViXNet for deepfake detection based on the paper published in Expert Systems with Applications (Q1 journal).

## Architecture Overview

ViXNet uses a **2-branch fusion architecture**:

1. **Xception Branch (CNN)**: Extracts global spatial features using pretrained Xception network
2. **Vision Transformer Branch (ViT)**: Learns patch-wise self-attention to capture subtle deepfake artifacts
3. **Feature Fusion**: Combines features from both branches
4. **Classification Head**: Binary classification (Real/Fake)

```
Input Image (224x224x3)
        |
        ├─→ [Xception Branch] ─→ Global Features (2048-d)
        |                              |
        └─→ [ViT Branch] ──────→ Attention Features (768-d)
                                       |
                                       ├─→ [Feature Fusion] → (512-d)
                                       |
                                       └─→ [Classifier] → Real/Fake
```

## 2-Stage Training Strategy

Following the paper's training strategy:

### Stage 1: Feature Extractor Training (Fast Training)
- **Duration**: 5 epochs
- **Frozen**: Xception backbone + ViT encoder
- **Trainable**: Fusion layers + Classification head
- **Learning Rate**: 1e-3
- **Goal**: Stabilize gradient, avoid overfitting, train quickly (1-2 hours)

### Stage 2: Controlled Fine-tuning
- **Duration**: 10 epochs  
- **Unfrozen**: Last 2-3 blocks of Xception + Last 1-2 ViT transformer blocks
- **Learning Rate**: 1e-5 (very low)
- **Goal**: Learn deepfake-specific artifacts without destroying general features

## File Structure

```
ViXNet/
├── model.py          # ViXNet model architecture
├── config.py         # Configuration and hyperparameters
├── dataset.py        # Data loading and preprocessing
├── utils.py          # Training utilities and metrics
├── train.py          # Main training script
└── README.md         # This file
```

## Requirements

All dependencies are already listed in the main repository's `requirements.txt`:
- torch
- torchvision  
- timm (for pretrained models)
- scikit-learn
- tqdm
- pillow
- numpy

## Dataset Structure

The model expects the following dataset structure:

```
CNN + Transformer/Dataset/
├── Train/
│   ├── Fake/
│   │   ├── fake_0.jpg
│   │   ├── fake_1.jpg
│   │   └── ...
│   └── Real/
│       ├── real_0.jpg
│       ├── real_1.jpg
│       └── ...
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## Usage

### Training

To train the model with the 2-stage strategy:

```bash
cd ViXNet
python train.py
```

The training script will:
1. Check dataset availability
2. Initialize ViXNet with pretrained weights
3. **Check for existing Stage 1 checkpoints** - skip if all 5 epochs exist
4. Run Stage 1 training (frozen feature extractors) - or skip if complete
5. Run Stage 2 training (fine-tune high-level layers)
6. Save best models after each epoch
7. Automatically test on test set after each epoch
8. Save training history and metrics

### Resuming Training

If Stage 1 is already complete (all 5 epoch checkpoints exist), simply run:

```bash
cd ViXNet
python train.py
```

The script will automatically:
- Detect existing Stage 1 checkpoints
- Skip Stage 1 training (saves ~1-2 hours)
- Load the best Stage 1 model
- Continue directly to Stage 2

No special flags or configuration needed!

### Testing the Model Implementation

To test the model architecture without training:

```bash
cd ViXNet
python model.py
```

To test data loading:

```bash
cd ViXNet
python dataset.py
```

To test configuration:

```bash
cd ViXNet
python config.py
```

## Model Checkpoints

Checkpoints are saved in `ViXNet/checkpoints/`:

- `best_model.pth` - Overall best model (highest validation accuracy)
- `best_model_stage1.pth` - Best model from Stage 1
- `best_model_stage2.pth` - Best model from Stage 2
- `checkpoint_stage{N}_epoch{M}.pth` - Epoch checkpoints

Each checkpoint includes:
- Model weights
- Optimizer state
- Training metrics
- Configuration
- Timestamp

## Training History

Training progress is saved in JSON format:

- `stage1_history.json` - Stage 1 training history
- `stage2_history.json` - Stage 2 training history  
- `full_training_history.json` - Complete training history

Each entry includes:
- Epoch number
- Stage number
- Training metrics (loss, accuracy)
- Validation metrics (loss, accuracy, precision, recall, F1)
- Test metrics (if enabled)
- Learning rate
- Confusion matrix

## Configuration

Key hyperparameters in `config.py`:

### Stage 1
- Epochs: 5
- Batch size: 32
- Learning rate: 1e-3
- Weight decay: 0.01

### Stage 2
- Epochs: 10
- Batch size: 32
- Learning rate: 1e-5
- Weight decay: 0.01

### Model
- Image size: 224x224
- Xception output: 2048-d
- ViT output: 768-d (ViT-Base)
- Fusion dimension: 512-d

### Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing (p=0.3)

## Features

✅ **2-stage training strategy** as described in the paper  
✅ **Automatic Stage 1 skip** when all checkpoints exist  
✅ **Automatic best model saving** after each epoch  
✅ **Epoch-wise testing** on test set  
✅ **Mixed precision training** for faster training  
✅ **Gradient clipping** for stable training  
✅ **Early stopping** to prevent overfitting  
✅ **Learning rate scheduling** (cosine annealing)  
✅ **Comprehensive metrics** (accuracy, precision, recall, F1, confusion matrix)  
✅ **Training history logging** in JSON format  

## Expected Performance

Based on similar architectures on deepfake detection:
- **Accuracy**: 95-99% on standard datasets
- **Training time**: 
  - Stage 1: 1-2 hours (GPU)
  - Stage 2: 2-4 hours (GPU)
- **Inference**: ~30ms per image (GPU)

## Paper Reference

**ViXNet: Vision Transformer with Xception Network for deepfakes based video and image forgery detection**

Published in: Expert Systems with Applications (Q1 Journal)

Key contributions:
- 2-branch fusion architecture (CNN + Transformer)
- Transfer learning from ImageNet
- Controlled fine-tuning strategy
- State-of-the-art performance on FaceForensics++ and Celeb-DF

## Notes

- The model uses **pretrained weights** from ImageNet for both Xception and ViT
- **Transfer learning** is applied instead of training from scratch
- The 2-stage training strategy prevents overfitting and stabilizes gradients
- Testing on the test set after each epoch helps monitor generalization

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size in `config.py`
- Disable mixed precision training
- Use smaller ViT model (`vit_small_patch16_224`)

### Dataset Not Found
- Ensure dataset is at `../CNN + Transformer/Dataset/`
- Check folder structure matches expected format
- Verify images are in correct subdirectories

### Slow Training
- Enable mixed precision training
- Increase batch size (if memory allows)
- Use more workers for data loading
- Ensure GPU is being used (check `Config.DEVICE`)

## License

This implementation is for educational and research purposes.
