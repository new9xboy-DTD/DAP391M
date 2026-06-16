# DAP391M - Deepfake Detection

Repository này chứa mã nguồn nghiên cứu và demo web cho bài toán phát hiện ảnh deepfake. Phần triển khai chính hiện tại là **ViXNet**, kết hợp Xception và Vision Transformer, kèm giao diện React + Flask để tải model, chạy inference và xem một số chỉ số đánh giá.

## Nội Dung Repo

- `ViXNet/`: mã huấn luyện, kiến trúc model, dataset loader và ví dụ inference.
- `ViXNet/web_app/`: ứng dụng web gồm backend Flask và frontend React.
- `Paper_Group1_Final.pdf`: bài báo/báo cáo final của nhóm.
- `requirements.txt`: dependency Python cấp repo.
- `run.sh`: ví dụ cài PyTorch CUDA 11.8.

Dataset, checkpoint/model weights, file log và dữ liệu sinh trong quá trình chạy không được commit vào repo.

## Dataset

Code mặc định tìm dataset trong thư mục `data/` tại root repo:

```text
data/
  FaceForensics_new/
    train/
      Fake/
      Real/
    val/
      Fake/
      Real/
    test/
      Fake/
      Real/
```

Bạn có thể đổi đường dẫn bằng biến môi trường:

```bash
VIXNET_DATA_ROOT=/path/to/data
VIXNET_DEFAULT_DATASET=/path/to/FaceForensics_new
VIXNET_CELEB_DATASET=/path/to/Celeb_V2
VIXNET_WILDDEEPFAKE_DATASET=/path/to/wilddeepfake
VIXNET_DFDC_DATASET=/path/to/real_vs_fake/real-vs-fake
```

## Cài Đặt

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng CUDA, chọn đúng bản PyTorch theo máy của bạn từ tài liệu chính thức của PyTorch.

## Chạy Huấn Luyện

```bash
cd ViXNet
python train.py
```

Checkpoint sẽ được lưu trong `ViXNet/checkpoints/` hoặc `checkpoints/` tùy cấu hình. Các thư mục checkpoint đã được ignore để tránh commit file model lớn.

## Chạy Web App

Backend:

```bash
cd ViXNet/web_app/backend
pip install -r requirements.txt
python app.py
```

Frontend:

```bash
cd ViXNet/web_app/frontend
npm install
npm start
```

Mặc định:

- Backend: `http://127.0.0.1:5000`
- Frontend: `http://localhost:3000`

Trên Windows có thể chạy:

```bat
cd ViXNet\web_app
start.bat
```

## Lưu Ý Bảo Mật

- Chỉ upload checkpoint/model từ nguồn bạn tin tưởng. File PyTorch checkpoint có thể không an toàn nếu lấy từ nguồn lạ.
- Flask backend mặc định chạy local với `debug=False`. Nếu cần bật debug, đặt `VIXNET_DEBUG=1`.
- Không commit dataset, checkpoint, `.env`, log, file nén hoặc tài liệu nháp.
- Ứng dụng web hiện phục vụ mục đích demo/nghiên cứu, chưa phải cấu hình production.

## Tài Liệu Chi Tiết

- [ViXNet README](ViXNet/README.md)
- [Web App README](ViXNet/web_app/README.md)
- [Hướng dẫn tiếng Việt](HUONG_DAN_TIENG_VIET.md)

## License

Repo này phục vụ mục đích học tập và nghiên cứu. Nếu bạn muốn người khác tái sử dụng code rõ ràng hơn, hãy thêm file `LICENSE` phù hợp trước khi public rộng rãi.
