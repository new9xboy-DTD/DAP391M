# Hệ thống Nhận diện Ảnh Deepfake với CNN + Transformer

## Tổng quan

Dự án này triển khai một mô hình hybrid CNN + Transformer để nhận diện ảnh deepfake. Model kết hợp sức mạnh của CNN trong việc trích xuất features cục bộ và Transformer trong việc học mối quan hệ global giữa các features.

## Kiến trúc Model

### 1. CNN Backbone (EfficientNet-B0)
- Pretrained trên ImageNet
- Trích xuất features từ ảnh 256x256
- Output: features 1280 channels với spatial size 8x8

### 2. Transformer Encoder
- 4 lớp Transformer encoder
- 8 attention heads
- Dimension: 512
- Feedforward dimension: 2048
- Dropout: 0.1

### 3. Classification Head
- LayerNorm + Dropout
- 2-layer MLP với GELU activation
- Output: 2 classes (Fake/Real)

## Cấu trúc Dataset

Dataset phải được tổ chức theo cấu trúc sau:

```
Dataset/
├── Train/
│   ├── Fake/
│   │   ├── fake_1.jpg
│   │   ├── fake_2.jpg
│   │   └── ...
│   └── Real/
│       ├── real_1.jpg
│       ├── real_2.jpg
│       └── ...
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## Yêu cầu hệ thống

### Dependencies
```bash
pip install -r requirements.txt
```

Các thư viện chính:
- PyTorch 2.9.1
- torchvision 0.24.1
- timm 1.0.24
- scikit-learn 1.8.0
- tqdm 4.67.1
- Pillow 12.1.0

### Hardware
- **Khuyến nghị**: GPU với ít nhất 8GB VRAM (CUDA support)
- **Tối thiểu**: CPU (training sẽ chậm hơn nhiều)

## Cách sử dụng

### Training Model

```bash
cd "CNN + Transformer"
python deepfake_detection.py
```

### Tùy chỉnh tham số

Chỉnh sửa class `Config` trong file `deepfake_detection.py`:

```python
class Config:
    # Tham số model
    IMG_SIZE = 256          # Kích thước ảnh
    BATCH_SIZE = 32         # Batch size
    NUM_EPOCHS = 50         # Số epochs
    LEARNING_RATE = 0.0001  # Learning rate
    
    # Tham số CNN
    CNN_MODEL = "efficientnet_b0"  # Có thể thay đổi: efficientnet_b1, efficientnet_b2...
    PRETRAINED = True               # Sử dụng pretrained weights
    
    # Tham số Transformer
    D_MODEL = 512                    # Dimension
    NHEAD = 8                        # Số attention heads
    NUM_TRANSFORMER_LAYERS = 4       # Số lớp transformer
    DIM_FEEDFORWARD = 2048           # FFN dimension
    DROPOUT = 0.1                    # Dropout rate
    
    # Early stopping
    PATIENCE = 10  # Dừng sớm nếu không cải thiện
```

### Output

Sau khi training, các file sau sẽ được tạo trong thư mục `checkpoints/`:

- `checkpoint_epoch_N.pth`: Checkpoint mỗi 5 epochs
- `best_model.pth`: Model tốt nhất (validation accuracy cao nhất)
- `training_history.json`: Lịch sử training (loss, metrics qua các epochs)
- `test_results.json`: Kết quả đánh giá trên test set

## Kết quả Training

Trong quá trình training, bạn sẽ thấy:

```
======================================================================
🚀 BẮT ĐẦU TRAINING MODEL NHẬN DIỆN DEEPFAKE
======================================================================
📱 Device: cuda
🖼️  Image size: 256x256
📦 Batch size: 32
🔄 Number of epochs: 50
📚 Learning rate: 0.0001
======================================================================

📂 Đang load dữ liệu...
📊 Số lượng ảnh training: XXXXX
📊 Số lượng ảnh validation: XXXXX
📊 Số lượng ảnh test: XXXXX

🏗️  Đang khởi tạo model...
📐 CNN feature dimension: 1280
📐 Spatial size: 64 (grid: 8x8)
✅ Model khởi tạo thành công!
📊 Tổng số parameters: X,XXX,XXX
```

## Metrics được tracking

- **Loss**: CrossEntropyLoss
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác của predictions
- **Recall**: Tỷ lệ phát hiện đúng
- **F1-Score**: Harmonic mean của precision và recall
- **Confusion Matrix**: Ma trận nhầm lẫn

## Tips để cải thiện hiệu suất

1. **Tăng batch size**: Nếu có đủ GPU memory, tăng `BATCH_SIZE` lên 64 hoặc 128
2. **Data augmentation**: Thêm các augmentation khác trong `get_data_transforms()`
3. **Model backbone**: Thử các model lớn hơn như `efficientnet_b3`, `efficientnet_b4`
4. **Transformer layers**: Tăng `NUM_TRANSFORMER_LAYERS` lên 6 hoặc 8
5. **Learning rate**: Thử các learning rates khác nhau (0.001, 0.00001)
6. **Mixed precision**: Sử dụng `torch.cuda.amp` để training nhanh hơn

## Xử lý lỗi thường gặp

### Out of Memory (OOM)
- Giảm `BATCH_SIZE` xuống 16 hoặc 8
- Giảm `IMG_SIZE` xuống 224
- Giảm `NUM_TRANSFORMER_LAYERS` xuống 2

### Dataset không tìm thấy
- Kiểm tra cấu trúc thư mục Dataset
- Đảm bảo đường dẫn `DATA_DIR` trong Config đúng

### Training quá chậm
- Sử dụng GPU nếu có
- Giảm `NUM_WORKERS` nếu CPU yếu
- Giảm số epochs hoặc sử dụng early stopping

## Load Model đã train

Để sử dụng model đã train cho inference:

```python
import torch
from deepfake_detection import CNNTransformerModel, Config

# Load model
model = CNNTransformerModel()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(Config.DEVICE)

# Inference trên ảnh mới
# ... (code inference)
```

## Tác giả

Dự án DAP391M - Deepfake Detection

## License

[Thêm license của bạn ở đây]
