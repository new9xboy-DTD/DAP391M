# Hướng Dẫn Ứng Dụng Web ViXNet

## Tổng Quan

Ứng dụng web ViXNet cung cấp giao diện trực quan để:
- **Hiển thị kiến trúc mô hình** - Trực quan hóa cấu trúc 2 nhánh của ViXNet
- **Phân tích ảnh** - Kéo thả ảnh để phát hiện deepfake (Real/Fake)
- **Phân tích mô hình** - Tải lên file mô hình (.pth) để tính AUC và các chỉ số
- **Trực quan hóa kết quả** - Biểu đồ ROC, confusion matrix, và các metrics

## Tính Năng Chính

### 1. 🏗️ Trực Quan Hóa Kiến Trúc Mô Hình

Hiển thị:
- Nhánh Xception (CNN) - Trích xuất đặc trưng không gian toàn cục (2048D)
- Nhánh Vision Transformer (ViT) - Học self-attention từng patch (192D)
- Lớp Feature Fusion - Kết hợp đặc trưng từ 2 nhánh (512D)
- Lớp Classification - Phân loại Real/Fake

### 2. 🖼️ Kéo Thả Ảnh Để Inference

**Cách sử dụng:**
1. Kéo và thả ảnh vào khung "Image Inference" (hoặc click để chọn file)
2. Ứng dụng tự động xử lý và phân tích
3. Xem kết quả:
   - Dự đoán: Real (Thật) hoặc Fake (Giả)
   - Độ tin cậy (Confidence): % chính xác
   - Xác suất cho cả 2 lớp

**Định dạng hỗ trợ:** JPG, PNG, JPEG

**Kết quả mẫu:**
```
Dự đoán: Real ✓
Độ tin cậy: 98.76%
Xác suất:
  Fake: 1.24%
  Real: 98.76%
```

### 3. 🔧 Kéo Thả Mô Hình Để Phân Tích

**Cách sử dụng:**
1. Kéo và thả file mô hình (.pth) vào khung "Model Analysis"
2. Đợi hệ thống phân tích (30-60 giây)
3. Xem kết quả chi tiết:
   - **AUC Score** - Diện tích dưới đường cong ROC
   - **Accuracy** - Độ chính xác trên tập test
   - **Confusion Matrix** - Ma trận nhầm lẫn
   - **ROC Curve** - Biểu đồ đường cong ROC
   - Thông tin mô hình (epoch, kiến trúc)

**Chỉ số AUC:**
- 1.0 = Phân loại hoàn hảo
- 0.9-0.99 = Mô hình tốt
- 0.5 = Phân loại ngẫu nhiên

**Kết quả mẫu:**
```
AUC Score: 0.9945
Độ chính xác: 98.23%
Số mẫu test: 1000

Confusion Matrix:
            Dự đoán
           Fake  Real
Thực tế Fake  450    12
        Real    8   530
```

## Cài Đặt

### Yêu Cầu
- Python 3.8+ (với PyTorch, Flask)
- Node.js 16+ và npm
- Dataset test tại `CNN + Transformer/Dataset/Test/` (tùy chọn, cho tính AUC)

### Bước 1: Cài Đặt Thư Viện

**Backend:**
```bash
cd ViXNet/web_app/backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd ViXNet/web_app/frontend
npm install
```

### Bước 2: Khởi Động Ứng Dụng

**Cách 1 - Sử dụng script tự động (Linux/Mac):**
```bash
cd ViXNet/web_app
./start.sh
```

**Cách 2 - Sử dụng script tự động (Windows):**
```bash
cd ViXNet\web_app
start.bat
```

**Cách 3 - Khởi động thủ công:**

Terminal 1 (Backend):
```bash
cd ViXNet/web_app/backend
python app.py
```

Terminal 2 (Frontend):
```bash
cd ViXNet/web_app/frontend
npm start
```

### Bước 3: Mở Trình Duyệt

Ứng dụng tự động mở tại: **http://localhost:3000**

## Kiến Trúc Hệ Thống

```
┌─────────────┐         HTTP          ┌──────────────┐
│   React     │ ◄─────────────────► │    Flask     │
│  Frontend   │      JSON/Files      │   Backend    │
│  (Port 3000)│                       │  (Port 5000) │
└─────────────┘                       └──────┬───────┘
                                             │
                                      ┌──────▼───────┐
                                      │   ViXNet     │
                                      │    Model     │
                                      └──────────────┘
```

## API Endpoints

### GET /api/health
Kiểm tra trạng thái backend và mô hình

### GET /api/model-info
Lấy thông tin về mô hình hiện tại

### POST /api/predict
Dự đoán ảnh (Real/Fake)
- Input: File ảnh (multipart/form-data)
- Output: JSON với prediction, confidence, probabilities

### POST /api/analyze-model
Phân tích mô hình và tính AUC
- Input: File mô hình .pth (multipart/form-data)
- Output: JSON với AUC, accuracy, confusion matrix, ROC curve

### POST /api/calculate-auc
Tính AUC cho mô hình hiện tại
- Output: JSON với metrics và biểu đồ

## Hướng Dẫn Sử Dụng Chi Tiết

### Bước 1: Xem Kiến Trúc Mô Hình
- Khi mở ứng dụng, bạn sẽ thấy sơ đồ kiến trúc ViXNet
- Hiển thị 2 nhánh: Xception và Vision Transformer
- Thông tin về dimensions và layers

### Bước 2: Test Inference Với Ảnh
1. Chuẩn bị ảnh khuôn mặt (JPG/PNG)
2. Kéo thả vào khung "Image Inference"
3. Chờ 1-2 giây để xử lý
4. Xem kết quả dự đoán với độ tin cậy

### Bước 3: Phân Tích Mô Hình (Nếu Có)
1. Chuẩn bị file mô hình đã train (.pth)
2. Kéo thả vào khung "Model Analysis"
3. Chờ 30-60 giây để tính AUC
4. Xem các metrics chi tiết:
   - AUC score
   - Accuracy
   - Confusion matrix
   - ROC curve (biểu đồ)

### Bước 4: So Sánh Các Mô Hình
1. Upload mô hình thứ nhất → ghi nhận AUC
2. Click "Upload Another Model"
3. Upload mô hình thứ hai → so sánh AUC
4. Mô hình có AUC cao hơn = tốt hơn

## Xử Lý Lỗi Thường Gặp

### "Backend API Not Available"
- Đảm bảo Flask backend đang chạy trên port 5000
- Kiểm tra terminal backend có lỗi không
- Cài đặt lại dependencies: `pip install Flask flask-cors`

### "Model not loaded"
- Backend khởi động với mô hình chưa train nếu không có checkpoint
- Upload mô hình đã train qua giao diện
- Hoặc đặt checkpoint tại `ViXNet/checkpoints/best_model.pth`

### "Dataset not available"
- Cảnh báo này xuất hiện khi tính AUC mà không có dataset test
- Inference ảnh vẫn hoạt động bình thường
- Để fix: Đặt dataset tại `CNN + Transformer/Dataset/Test/` với thư mục Fake/ và Real/

### Frontend không load
- Kiểm tra Node.js đã cài: `node --version`
- Xóa thư mục `node_modules` và chạy lại `npm install`
- Xóa cache trình duyệt (Ctrl+Shift+Delete)

## Tối Ưu Hiệu Năng

**Để inference nhanh hơn:**
- Sử dụng GPU nếu có (PyTorch với CUDA)
- Inference đầu tiên sẽ chậm (khởi tạo mô hình)
- Các lần sau nhanh hơn (~1-2 giây/ảnh)

**Để tính AUC nhanh hơn:**
- Sử dụng GPU
- Giảm kích thước dataset test nếu quá lớn
- Chỉ tính 1 lần mỗi khi upload mô hình mới

## Lưu Ý Quan Trọng

1. **Định dạng ảnh:** Chỉ hỗ trợ JPG, PNG, JPEG
2. **Kích thước ảnh:** Tự động resize về 224x224
3. **Loại mô hình:** Chỉ hỗ trợ file .pth hoặc .pt của PyTorch
4. **Dataset:** Cần có dataset test để tính AUC
5. **Port:** Backend chạy port 5000, Frontend chạy port 3000

## Tài Liệu Tham Khảo

- **README.md** - Hướng dẫn đầy đủ (tiếng Anh)
- **DEMO.md** - Hướng dẫn demo chi tiết
- **VISUAL_GUIDE.md** - Sơ đồ cấu trúc và luồng dữ liệu
- **QUICKSTART.md** - Hướng dẫn nhanh 5 phút

## Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra log backend: `backend/backend.log`
2. Mở Console trong trình duyệt (F12)
3. Xem lại các bước cài đặt
4. Đảm bảo tất cả dependencies đã được cài đặt

## Kết Luận

Ứng dụng web ViXNet cung cấp giao diện trực quan và dễ sử dụng để:
- Phân tích ảnh deepfake
- Đánh giá hiệu suất mô hình
- So sánh các phiên bản mô hình khác nhau
- Trực quan hóa metrics và biểu đồ

Chúc bạn sử dụng thành công! 🎉
