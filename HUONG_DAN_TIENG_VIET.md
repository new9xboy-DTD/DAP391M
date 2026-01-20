# Hướng Dẫn Sử Dụng - Hệ Thống Phát Hiện Deepfake Đa Model

## 🎯 Các Tính Năng Chính

### 1. **Không Tự Động Tải Model Mặc Định**
Khi mở web lần đầu, hệ thống sẽ KHÔNG tự động tải model. Điều này giúp tiết kiệm bộ nhớ và cho phép bạn tự chọn model muốn sử dụng.

### 2. **Hỗ Trợ Nhiều Loại Model**
Ứng dụng hiện hỗ trợ ba loại model phát hiện deepfake:
- **ViXNet**: Kết hợp Vision Transformer + Xception
- **Xception Only**: Chỉ sử dụng CNN Xception
- **ViT Only**: Chỉ sử dụng Vision Transformer

### 3. **Chọn Dataset Để Test**
Bạn có thể chọn dataset nào để đánh giá hiệu suất của model. Điều này cho phép test cùng một model trên nhiều dataset khác nhau để so sánh kết quả.

## 📖 Hướng Dẫn Từng Bước

### Bước 1: Khởi Động Ứng Dụng

1. **Khởi động Backend**:
   ```bash
   cd ViXNet/web_app/backend
   python app.py
   ```
   
   Bạn sẽ thấy:
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

2. **Khởi động Frontend** (terminal mới):
   ```bash
   cd ViXNet/web_app/frontend
   npm start
   ```
   
   Trình duyệt sẽ tự động mở http://localhost:3000

### Bước 2: Upload Model

1. **Tìm phần "Model Upload & Analysis"**
   - Đây là phần đầu tiên bạn thấy khi chưa có model nào được tải

2. **Upload file model của bạn**:
   - Kéo thả file `.pth` hoặc `.pt` vào vùng dropzone
   - HOẶC click vào dropzone để chọn file
   
3. **Chọn dataset**:
   - Sau khi thả file, một dropdown sẽ xuất hiện
   - Chọn dataset muốn sử dụng để đánh giá model
   - Dataset mặc định đã được chọn sẵn

4. **Click "Analyze Model"**:
   - Quá trình phân tích sẽ bắt đầu tự động
   - Biểu tượng loading sẽ hiển thị tiến trình
   - Có thể mất 1-5 phút tùy vào kích thước dataset

5. **Xem Kết Quả**:
   - Điểm AUC (Area Under Curve)
   - Tỷ lệ chính xác (Accuracy)
   - Ma trận nhầm lẫn (Confusion matrix)
   - Dữ liệu đường cong ROC
   - Thông tin kiến trúc model

### Bước 3: Test với Hình Ảnh

Sau khi model đã được tải và phân tích:

1. **Tìm phần "Image Inference"**
   - Phần này chỉ xuất hiện sau khi model đã được tải

2. **Upload hình ảnh**:
   - Kéo thả hình ảnh (JPG, PNG, v.v.)
   - HOẶC click để chọn hình

3. **Xem Kết Quả Dự Đoán**:
   - **Prediction**: "Real" (Thật) hoặc "Fake" (Giả)
   - **Confidence**: Độ tin cậy của dự đoán (%)
   - **Probabilities**: Phân tích chi tiết xác suất Real vs Fake

### Bước 4: Upload Model Khác (Tùy Chọn)

1. Click nút "Upload Another Model"
2. Lặp lại các bước từ Bước 2
3. Model mới sẽ thay thế model hiện tại

## 🎨 Giao Diện

### Màn Hình Chào (Chưa Có Model)
```
┌─────────────────────────────────────────┐
│  🧠 Multi-Model Deepfake Detection      │
│  Hỗ trợ ViXNet, Xception, ViT          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  🔧 Model Upload & Analysis             │
│  ┌───────────────────────────────────┐  │
│  │  🔧                               │  │
│  │  Kéo thả file model vào đây      │  │
│  │  Hỗ trợ: .pth, .pt               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  👋 Chào mừng!                          │
│  Để bắt đầu:                            │
│  1. Upload file model đã train          │
│  2. Chọn dataset để đánh giá            │
│  3. Đợi quá trình phân tích             │
│  4. Upload hình để phát hiện            │
└─────────────────────────────────────────┘
```

### Sau Khi Upload File (Trước Phân Tích)
```
┌─────────────────────────────────────────┐
│  🔧 Model Upload & Analysis             │
│  ┌───────────────────────────────────┐  │
│  │  📦                               │  │
│  │  model_vixnet.pth                 │  │
│  │  Sẵn sàng để phân tích            │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Chọn Dataset Để Đánh Giá:             │
│  ┌───────────────────────────────────┐  │
│  │  Default Dataset            ▼     │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      Analyze Model                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Model Đã Tải - Giao Diện Đầy Đủ
```
┌─────────────────────────────────────────┐
│  📊 Kiến Trúc Model Hiện Tại           │
│  • Tên: ViXNet                         │
│  • Loại: vixnet                        │
│  • Trạng thái: Đã tải ✅              │
│  • AUC: 0.9543                         │
│  • Độ chính xác: 92.35%                │
└─────────────────────────────────────────┘

┌──────────────────────┬──────────────────┐
│  🔧 Upload Model     │  🖼️ Nhận Diện    │
│  [Đã tải: ✅]        │  Hình Ảnh        │
│  ┌────────────────┐  │  ┌──────────────┐│
│  │ Upload Model   │  │  │ 📸           ││
│  │ Khác           │  │  │ Kéo thả ảnh  ││
│  └────────────────┘  │  │ vào đây      ││
│                      │  └──────────────┘│
└──────────────────────┴──────────────────┘

┌─────────────────────────────────────────┐
│  📈 Kết Quả                             │
│  Dự đoán: Real ✓ (Thật)                │
│  Độ tin cậy: 95.43%                    │
│  • Fake: 4.57%                         │
│  • Real: 95.43%                        │
└─────────────────────────────────────────┘
```

## 🔧 Cấu Hình

### Thêm Dataset Mới

Chỉnh sửa file `ViXNet/config.py`:

```python
DATASETS = {
    'default': {
        'name': 'Default Dataset',
        'path': '/duong/dan/toi/dataset/mac/dinh',
        'train': '/duong/dan/toi/dataset/mac/dinh/Train',
        'val': '/duong/dan/toi/dataset/mac/dinh/Validation',
        'test': '/duong/dan/toi/dataset/mac/dinh/Test'
    },
    'celeb_df': {
        'name': 'Celeb-DF Dataset',
        'path': '/duong/dan/toi/celeb-df',
        'train': '/duong/dan/toi/celeb-df/Train',
        'val': '/duong/dan/toi/celeb-df/Validation',
        'test': '/duong/dan/toi/celeb-df/Test'
    },
    'faceforensics': {
        'name': 'FaceForensics++ Dataset',
        'path': '/duong/dan/toi/faceforensics',
        'train': '/duong/dan/toi/faceforensics/Train',
        'val': '/duong/dan/toi/faceforensics/Validation',
        'test': '/duong/dan/toi/faceforensics/Test'
    }
}
```

Sau khi thêm dataset:
1. Khởi động lại backend server
2. Dataset mới sẽ tự động xuất hiện trong dropdown

### Cấu Trúc Thư Mục Dataset

Mỗi dataset phải có cấu trúc như sau:
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

## ⚡ Mẹo & Thủ Thuật

### Hiệu Suất
- **Phân tích model**: Mất 1-5 phút tùy kích thước dataset test
- **Nhận diện ảnh**: Gần như tức thì (< 1 giây)
- **Bộ nhớ**: Không có model = sử dụng bộ nhớ tối thiểu

### Các Thực Hành Tốt
1. **Test trên nhiều dataset**: Upload cùng một model và test trên các dataset khác nhau để so sánh
2. **Lưu thông tin model**: Ghi chú điểm AUC cho các model/dataset khác nhau
3. **Xóa model**: Sử dụng "Upload Another Model" để giải phóng bộ nhớ trước khi tải model mới

### Xử Lý Sự Cố
- **Lỗi "Model not loaded"**: Bạn cần upload và phân tích model trước
- **Lỗi "Dataset not available"**: Kiểm tra đường dẫn dataset trong config.py
- **Phân tích chậm**: Dataset test lớn sẽ mất nhiều thời gian; đây là bình thường
- **Backend không phản hồi**: Đảm bảo Flask server đang chạy ở port 5000

## 🚀 Ví Dụ Quy Trình

**Kịch bản**: Test nhiều model trên cùng một dataset

1. **Tải Model 1** (ViXNet)
   - Upload `vixnet_model.pth`
   - Chọn "Default Dataset"
   - Phân tích → Nhận được AUC: 0.954, Accuracy: 92.3%

2. **Test với hình ảnh**
   - Upload hình test
   - Ghi chú kết quả dự đoán và độ tin cậy

3. **Tải Model 2** (Xception Only)
   - Click "Upload Another Model"
   - Upload `xception_model.pth`
   - Chọn "Default Dataset"
   - Phân tích → Nhận được AUC: 0.931, Accuracy: 89.7%

4. **So sánh kết quả**
   - ViXNet hoạt động tốt hơn trên dataset này
   - Cả hai model có thể test với cùng hình để so sánh

## 📊 Hiểu Kết Quả

### AUC (Area Under Curve)
- **Phạm vi**: 0.0 đến 1.0
- **Ý nghĩa**:
  - 0.9 - 1.0: Xuất sắc
  - 0.8 - 0.9: Tốt
  - 0.7 - 0.8: Khá
  - Dưới 0.7: Kém

### Độ Chính Xác (Accuracy)
- Tỷ lệ phần trăm dự đoán đúng
- Càng cao càng tốt
- So sánh với AUC để có cái nhìn cân bằng

### Độ Tin Cậy (Confidence)
- Mức độ chắc chắn của model về dự đoán
- 95%+ = Rất chắc chắn
- 70-95% = Chắc chắn
- Dưới 70% = Không chắc chắn

## 🔑 Điểm Khác Biệt So Với Trước

### Trước Đây
- ✗ Tự động tải model ViXNet khi khởi động
- ✗ Chỉ hỗ trợ model ViXNet
- ✗ Không thể chọn dataset
- ✗ Phải có model để vào web

### Bây Giờ
- ✓ KHÔNG tự động tải model (tiết kiệm bộ nhớ)
- ✓ Hỗ trợ 3 loại model: ViXNet, Xception Only, ViT Only
- ✓ Có thể chọn dataset để test model
- ✓ Vào web tự do, chỉ tải model khi cần
- ✓ Có thể đổi model dễ dàng
- ✓ Tự động nhận diện loại model khi upload

## 📝 Ghi Chú Quan Trọng

### Dataset
- Dataset hiện tại được cấu hình cố định trong `config.py`
- Để thêm dataset mới, chỉnh sửa file `config.py` và khởi động lại backend
- Trong tương lai, có thể mở rộng để upload dataset tự động

### Model
- Khi upload model, hệ thống sẽ tự động nhận diện loại model
- Nếu không nhận diện được, mặc định sẽ thử ViXNet
- Model phải có format checkpoint chuẩn với key `model_state_dict`

### Bảo Mật
- ⚠️ Chỉ upload model từ nguồn tin cậy
- Model file có thể chứa code độc hại
- Không chia sẻ ứng dụng công khai mà không có xác thực người dùng

---

**Lưu ý**: Ứng dụng này được thiết kế cho mục đích nghiên cứu và testing. Luôn xác thực kết quả và cân nhắc các vấn đề đạo đức khi làm việc với hệ thống phát hiện deepfake.
