# Hệ thống Nhận diện Deepfake - Sequence Modeling + Diffusion + GCN

## Giới thiệu

Đây là hệ thống nhận diện ảnh deepfake sử dụng ba phương pháp chính:

1. **Sequence Modeling (VQ-VAE + Transformer)**: Chuyển ảnh thành chuỗi token và dùng mô hình ngôn ngữ để phát hiện bất thường
2. **Back-in-Time Diffusion (DDPM)**: Sử dụng mô hình khuếch tán để phát hiện dấu vết nhân tạo
3. **CNN/ViT + GCN**: Kết hợp CNN và Graph Convolutional Network để phân tích pixel và cấu trúc khuôn mặt

## Cấu trúc Thư mục

```
Sequence_Diffusion_GCN/
├── config.py              # Cấu hình và hyperparameters
├── data_preprocessing.py  # Xử lý và load dữ liệu
├── vqvae_module.py       # Module VQ-VAE (tokenization)
├── transformer_module.py # Module Transformer (sequence modeling)
├── diffusion_module.py   # Module DDPM (diffusion)
├── cnn_vit_module.py     # Module CNN/ViT (feature extraction)
├── gcn_module.py         # Module GCN (structural analysis)
├── fusion_module.py      # Module Fusion (kết hợp scores)
├── train.py              # Script huấn luyện
├── inference.py          # Script sử dụng model
└── README.md             # Tài liệu này
```

## Yêu cầu Cài đặt

```bash
pip install torch torchvision timm scikit-learn pillow tqdm numpy
```

Các dependencies bổ sung (tùy chọn):
```bash
pip install dlib mediapipe  # Cho facial landmark detection
pip install tensorboard     # Cho visualization
```

## Cấu hình Dataset

Trước khi sử dụng, cần cấu hình đường dẫn dataset trong `config.py`:

```python
class DataConfig:
    DATASET_ROOT = "path/to/your/dataset"  # Thay đổi đường dẫn này
```

Cấu trúc dataset yêu cầu:
```
Dataset/
├── Train/
│   ├── Fake/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Real/
│       ├── image1.jpg
│       └── ...
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

## Kiến trúc Hệ thống

### 1. Mô-đun Sequence Modeling (VQ-VAE + Transformer)

**Ý tưởng**: Chuyển ảnh thành chuỗi token rời rạc và học phân phối xác suất của các token ảnh thật.

```
Ảnh → VQ-VAE Encoder → Token indices → Transformer → Perplexity Score
```

- **VQ-VAE**: Mã hóa ảnh thành các token rời rạc (như từ trong NLP)
- **Transformer (GPT-like)**: Học dự đoán token tiếp theo
- **Perplexity**: Đo độ "bất ngờ" của chuỗi token - cao = khả năng deepfake

### 2. Mô-đun Diffusion (DDPM)

**Ý tưởng**: Huấn luyện mô hình khử nhiễu chỉ trên ảnh thật. Deepfake có "dấu vết" mà model không học được.

```
Ảnh → Thêm nhiễu → Khử nhiễu → So sánh với gốc → Residual Score
```

- **Forward process**: Thêm nhiễu dần vào ảnh
- **Reverse process**: Khử nhiễu để tái tạo ảnh
- **Anomaly score**: Sai số giữa ảnh gốc và ảnh đã khử nhiễu

### 3. Mô-đun CNN/ViT + GCN

**Ý tưởng**: Kết hợp phân tích pixel-level (CNN) và structural-level (GCN).

```
Ảnh → CNN/ViT → Features → Classifier → Prediction
Landmarks → GCN → Structural Features ↗
```

- **CNN/ViT**: Trích xuất đặc trưng pixel
- **GCN**: Phân tích cấu trúc khuôn mặt từ facial landmarks
- **Fusion**: Kết hợp hai nguồn thông tin

### 4. Mô-đun Fusion

**Ý tưởng**: Kết hợp scores từ tất cả các modules để ra quyết định cuối cùng.

```
Transformer Score  ─┐
Diffusion Score    ─├─→ Fusion → Final Prediction
CNN/ViT Score      ─┤
GCN Score          ─┘
```

Các phương pháp fusion:
- **Weighted Sum**: Tổng có trọng số
- **MLP**: Mạng MLP học cách kết hợp
- **Attention**: Self-attention để học trọng số tối ưu

## Huấn luyện

### Phase 1: Unsupervised (chỉ dùng ảnh thật)

```bash
# Train VQ-VAE
python train.py --phase vqvae

# Train Transformer
python train.py --phase transformer

# Train DDPM
python train.py --phase ddpm

# Hoặc train tất cả unsupervised modules
python train.py --phase 1
```

### Phase 2: Supervised (dùng cả Real và Fake)

```bash
# Train CNN/ViT classifier
python train.py --phase cnn_vit

# Hoặc train tất cả supervised modules
python train.py --phase 2
```

### Train tất cả

```bash
python train.py --phase all
```

### Options

```bash
python train.py --phase all --epochs 50 --device cuda
```

## Sử dụng Model

### Dự đoán một ảnh

```bash
python inference.py --image path/to/image.jpg
```

### Dự đoán batch ảnh

```bash
python inference.py --input_dir path/to/images/ --output_dir results/
```

### Options

```bash
python inference.py --image image.jpg \
    --checkpoint_dir checkpoints/ \
    --threshold 0.5 \
    --visualize
```

## API Python

```python
from inference import DeepfakeDetector

# Khởi tạo detector
detector = DeepfakeDetector(
    checkpoint_dir='checkpoints/',
    threshold=0.5
)

# Phát hiện deepfake
result = detector.detect('path/to/image.jpg')

print(f"Prediction: {result['prediction']}")  # 'REAL' hoặc 'FAKE'
print(f"Confidence: {result['confidence']}")
print(f"Anomaly Score: {result['anomaly_score']}")
```

## Cấu hình Hyperparameters

Tất cả cấu hình nằm trong `config.py`:

### DataConfig
- `IMG_SIZE`: Kích thước ảnh đầu vào (default: 256)
- `BATCH_SIZE`: Kích thước batch (default: 16)

### VQVAEConfig
- `EMBEDDING_DIM`: Kích thước embedding (default: 64)
- `NUM_EMBEDDINGS`: Số tokens trong codebook (default: 512)

### TransformerConfig
- `D_MODEL`: Kích thước model (default: 512)
- `NUM_LAYERS`: Số layers (default: 6)
- `NHEAD`: Số attention heads (default: 8)

### DiffusionConfig
- `NUM_TIMESTEPS`: Số bước diffusion (default: 1000)
- `BETA_SCHEDULE`: Loại schedule ('linear' hoặc 'cosine')

### CNNViTConfig
- `BACKBONE_NAME`: Tên backbone ('efficientnet_b4', 'vit_base_patch16_224')
- `PRETRAINED`: Sử dụng pretrained weights

### FusionConfig
- `FUSION_METHOD`: Phương pháp fusion ('weighted_sum', 'mlp', 'attention')
- `THRESHOLD`: Ngưỡng phân loại (default: 0.5)

## Tài liệu Tham khảo

1. Van Den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE)
2. Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM)
3. Grabovski et al. (2024) - Back-in-Time Diffusion for Deepfake Detection
4. Samad & Bandhu (2025) - Deepfake Detection using CNN+GCN
5. Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
6. Kipf & Welling, "Semi-Supervised Classification with GCNs"

## License

MIT License

## Tác giả

DAP391M Team
