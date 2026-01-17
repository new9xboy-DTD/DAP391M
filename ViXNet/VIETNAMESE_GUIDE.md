# ViXNet: Hướng Dẫn Sử Dụng Đầy Đủ

## Tổng Quan

ViXNet là một kiến trúc học sâu kết hợp CNN (Xception) và Vision Transformer (ViT) để phát hiện deepfake. Mô hình được thiết kế dựa trên bài báo được xuất bản trên tạp chí Q1 Expert Systems with Applications.

## Kiến Trúc

### 2 Nhánh Chính:

1. **Nhánh Xception (CNN)**
   - Trích xuất đặc trưng không gian toàn cục (global spatial features)
   - Sử dụng pretrained weights từ ImageNet
   - Output: vector 2048 chiều

2. **Nhánh Vision Transformer (ViT)**
   - Học self-attention theo từng patch
   - Bắt các artifact tinh vi trong deepfake
   - Sử dụng pretrained ViT-Base
   - Output: vector 768 chiều

3. **Feature Fusion**
   - Kết hợp đặc trưng từ cả 2 nhánh
   - Giảm chiều xuống 512-d
   - Sử dụng batch normalization và dropout

4. **Classification Head**
   - Phân loại nhị phân: Real/Fake
   - 2 fully connected layers với activation

## Chiến Lược Training 2 Giai Đoạn

### Giai Đoạn 1: Training Fusion + Classifier (5 epochs)

**Mục tiêu:** Ổn định gradient, tránh overfitting, training nhanh

**Cài đặt:**
- Freeze toàn bộ Xception
- Freeze toàn bộ ViT encoder
- Chỉ train: Fusion layers + Classification head
- Learning rate: 1e-3 (cao)
- Batch size: 32
- Thời gian: ~1-2 giờ

**Số parameters:**
- Tổng: 108M
- Trainable: 1.8M (chỉ 1.7%)

**Lợi ích:**
- Training rất nhanh
- Tránh phá hỏng pretrained features
- Gradient ổn định
- Không overfitting

### Giai Đoạn 2: Fine-tuning Có Kiểm Soát (10 epochs)

**Mục tiêu:** Học các artifact deepfake đặc thù

**Cài đặt:**
- Unfreeze last 2-3 blocks của Xception
- Unfreeze last 1-2 transformer blocks của ViT
- Learning rate: 1e-5 (rất thấp)
- Batch size: 32
- Thời gian: ~2-4 giờ

**Số parameters:**
- Tổng: 108M
- Trainable: 19M (17.7%)

**Lợi ích:**
- Học deepfake-specific patterns
- Không phá hỏng general features
- Tăng accuracy thêm 1-3%

## Cấu Trúc Dataset

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

## Cài Đặt & Chạy

### 1. Cài Đặt Dependencies

Tất cả dependencies đã có trong `requirements.txt` của repo:

```bash
pip install torch torchvision timm scikit-learn tqdm pillow numpy
```

### 2. Kiểm Tra Model

Test model có hoạt động không:

```bash
cd ViXNet
python model.py
```

Output mong đợi:
```
✅ ViXNet initialized successfully!
   Total parameters: 108,444,970
   Trainable parameters: 108,444,970
```

### 3. Kiểm Tra Dataset

```bash
cd ViXNet
python dataset.py
```

Sẽ hiển thị thông tin dataset nếu có.

### 4. Kiểm Tra Config

```bash
cd ViXNet
python config.py
```

Xem tất cả hyperparameters.

### 5. Training Model

**Chạy training đầy đủ 2 stages:**

```bash
cd ViXNet
python train.py
```

Script sẽ tự động:
1. Kiểm tra dataset
2. Khởi tạo model với pretrained weights
3. Training Stage 1 (5 epochs)
4. Lưu best model Stage 1
5. Training Stage 2 (10 epochs)
6. Lưu best model Stage 2
7. Test trên test set sau mỗi epoch
8. Lưu training history

### 6. Inference (Dự Đoán)

**Sử dụng trained model:**

```python
from inference_example import load_model, predict_image

# Load model
model = load_model('checkpoints/best_model.pth')

# Predict một ảnh
result = predict_image(model, 'path/to/image.jpg')
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Hoặc predict nhiều ảnh:**

```python
from inference_example import load_model, predict_batch

model = load_model('checkpoints/best_model.pth')
results = predict_batch(model, ['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

## Cấu Trúc File

```
ViXNet/
├── model.py                    # Kiến trúc ViXNet
├── config.py                   # Configuration
├── dataset.py                  # Data loading
├── utils.py                    # Training utilities
├── train.py                    # Main training script
├── inference_example.py        # Inference examples
├── README.md                   # English documentation
├── VIETNAMESE_GUIDE.md         # This file
└── __init__.py                 # Package init
```

## Checkpoints & Logs

### Checkpoints Được Lưu

Trong folder `checkpoints/`:

- `best_model.pth` - Best model tổng thể
- `best_model_stage1.pth` - Best model Stage 1
- `best_model_stage2.pth` - Best model Stage 2
- `checkpoint_stage{N}_epoch{M}.pth` - Checkpoint từng epoch

### Training History

File JSON được lưu:

- `stage1_history.json` - Lịch sử Stage 1
- `stage2_history.json` - Lịch sử Stage 2
- `full_training_history.json` - Lịch sử đầy đủ

Format:
```json
{
  "epoch": 1,
  "stage": 1,
  "train": {
    "loss": 0.5432,
    "accuracy": 0.7234
  },
  "val": {
    "loss": 0.4321,
    "accuracy": 0.8012,
    "precision": 0.7891,
    "recall": 0.8123,
    "f1": 0.8006
  },
  "test": {
    "loss": 0.4456,
    "accuracy": 0.7945,
    ...
  },
  "lr": 0.001
}
```

## Hyperparameters Quan Trọng

### Model Architecture
- Image size: 256×256
- Xception output: 2048-d
- ViT output: 768-d (ViT-Base)
- Fusion dimension: 512-d

### Stage 1
- Epochs: 5
- Learning rate: 1e-3
- Batch size: 32
- Weight decay: 0.01

### Stage 2
- Epochs: 10
- Learning rate: 1e-5
- Batch size: 32
- Weight decay: 0.01

### Optimization
- Optimizer: AdamW
- Scheduler: Cosine Annealing
- Gradient clipping: 1.0
- Mixed precision: Enabled
- Early stopping patience: 5

### Augmentation
- Random horizontal flip: 50%
- Random rotation: ±15°
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing: 30%

## Kết Quả Mong Đợi

Dựa trên các kiến trúc tương tự:

- **Accuracy**: 95-99% trên dataset chuẩn
- **Training time**: 
  - Stage 1: 1-2 giờ (GPU)
  - Stage 2: 2-4 giờ (GPU)
  - Tổng: ~3-6 giờ
- **Inference**: ~30ms/ảnh (GPU), ~200ms/ảnh (CPU)
- **Memory**: ~4GB VRAM (batch size 32)

## Tips & Tricks

### 1. Nếu Gặp Out of Memory (OOM)

**Giảm batch size:**
```python
# Trong config.py
STAGE1_BATCH_SIZE = 16  # thay vì 32
STAGE2_BATCH_SIZE = 16
```

**Tắt mixed precision:**
```python
MIXED_PRECISION = False
```

**Dùng ViT nhỏ hơn:**
```python
VIT_MODEL_NAME = 'vit_small_patch16_224'  # thay vì vit_base
```

### 2. Training Chậm

**Tăng batch size (nếu có RAM):**
```python
STAGE1_BATCH_SIZE = 64
STAGE2_BATCH_SIZE = 64
```

**Tăng workers:**
```python
NUM_WORKERS = 8  # thay vì 4
```

**Bật mixed precision:**
```python
MIXED_PRECISION = True
```

### 3. Model Overfit

**Tăng augmentation:**
```python
USE_RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.5  # thay vì 0.3
```

**Tăng dropout:**
```python
DROPOUT = 0.6  # thay vì 0.5
```

**Tăng weight decay:**
```python
STAGE1_WEIGHT_DECAY = 0.02  # thay vì 0.01
STAGE2_WEIGHT_DECAY = 0.02
```

### 4. Model Underfit

**Training lâu hơn:**
```python
STAGE1_EPOCHS = 10  # thay vì 5
STAGE2_EPOCHS = 20  # thay vì 10
```

**Tăng learning rate (Stage 1 only):**
```python
STAGE1_LR = 2e-3  # thay vì 1e-3
```

**Giảm regularization:**
```python
DROPOUT = 0.3
WEIGHT_DECAY = 0.005
```

## Monitoring Training

### 1. In Console

Training sẽ hiển thị real-time:
```
Stage 1 - Epoch 1 [TRAIN]: 100%|████████| 1000/1000 [05:23<00:00]
loss: 0.5432, acc: 0.7234

Stage 1 - Epoch 1 [VAL]: 100%|████████| 200/200 [01:12<00:00]

📊 METRICS - EPOCH 1
======================================================================

🎯 TRAINING:
   Loss: 0.5432
   Accuracy: 0.7234

✅ VALIDATION:
   Loss: 0.4321
   Accuracy: 0.8012
   Precision: 0.7891
   Recall: 0.8123
   F1-Score: 0.8006
```

### 2. Training History

Load và phân tích:
```python
import json

with open('checkpoints/full_training_history.json', 'r') as f:
    history = json.load(f)

# Xem accuracy qua các epochs
for entry in history:
    epoch = entry['epoch']
    val_acc = entry['val']['accuracy']
    print(f"Epoch {epoch}: {val_acc:.4f}")
```

### 3. Visualize (Optional)

```python
import matplotlib.pyplot as plt
import json

with open('checkpoints/full_training_history.json', 'r') as f:
    history = json.load(f)

epochs = [e['epoch'] for e in history]
train_acc = [e['train']['accuracy'] for e in history]
val_acc = [e['val']['accuracy'] for e in history]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Train')
plt.plot(epochs, val_acc, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_curve.png')
```

## Đánh Giá Model

### Load Best Model và Test

```python
import torch
from model import create_vixnet
from dataset import create_data_loaders
from utils import validate
import torch.nn as nn

# Load model
model = create_vixnet()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
data_loaders = create_data_loaders()
test_loader = data_loaders['test']

# Evaluate
criterion = nn.CrossEntropyLoss()
test_metrics = validate(model, test_loader, criterion)

print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1-Score: {test_metrics['f1']:.4f}")
```

## So Sánh với Baseline

| Model | Parameters | Accuracy | Training Time |
|-------|-----------|----------|---------------|
| Simple CNN | 5M | 85-90% | 30 min |
| ResNet50 | 25M | 90-93% | 1 hour |
| EfficientNet-B0 | 5M | 91-94% | 1 hour |
| **ViXNet** | **108M** | **95-99%** | **3-6 hours** |

## Troubleshooting

### Lỗi: "Dataset not found"

**Giải pháp:** Đảm bảo dataset ở đúng vị trí:
```bash
ls "../CNN + Transformer/Dataset/Train/Fake/"
ls "../CNN + Transformer/Dataset/Train/Real/"
```

### Lỗi: "CUDA out of memory"

**Giải pháp:** Giảm batch size trong `config.py`

### Lỗi: "timm model not found"

**Giải pháp:** 
```bash
pip install --upgrade timm
```

### Training bị NaN loss

**Giải pháp:**
- Giảm learning rate
- Bật gradient clipping
- Kiểm tra data normalization

## Best Practices

1. **Luôn bắt đầu với config mặc định** trước khi tune
2. **Monitor validation metrics** để phát hiện overfitting sớm
3. **Lưu checkpoints thường xuyên** để tránh mất tiến độ
4. **Test trên test set** chỉ một lần khi hoàn thành
5. **Sử dụng early stopping** để tránh training quá lâu
6. **Mixed precision training** để tăng tốc và tiết kiệm RAM

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra file README.md
2. Xem lại VIETNAMESE_GUIDE.md này
3. Chạy test scripts để debug
4. Kiểm tra logs trong checkpoints/

## Tham Khảo

- Paper gốc: Expert Systems with Applications (Q1)
- Xception: https://arxiv.org/abs/1610.02357
- Vision Transformer: https://arxiv.org/abs/2010.11929
- timm library: https://github.com/huggingface/pytorch-image-models
