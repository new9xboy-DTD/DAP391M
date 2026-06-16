# ViXNet Web App

Ứng dụng demo gồm backend Flask và frontend React để:

- Upload ảnh và dự đoán `Real`/`Fake`.
- Upload checkpoint `.pth` để phân tích model.
- Tính AUC, accuracy, F1, precision, recall trên test dataset nếu dataset có sẵn.
- Hiển thị kiến trúc và kết quả model trên giao diện web.

## Cấu Trúc

```text
web_app/
  backend/
    app.py
    requirements.txt
  frontend/
    public/
    src/
    package.json
  start.bat
```

## Chạy Backend

```bash
cd ViXNet/web_app/backend
pip install -r requirements.txt
python app.py
```

Mặc định backend chạy tại `http://127.0.0.1:5000`.

Biến môi trường hữu ích:

```bash
VIXNET_API_HOST=127.0.0.1
VIXNET_API_PORT=5000
VIXNET_DEBUG=0
VIXNET_CORS_ORIGINS=http://localhost:3000
```

## Chạy Frontend

```bash
cd ViXNet/web_app/frontend
npm install
npm start
```

Frontend chạy tại `http://localhost:3000` và proxy API tới backend.

## Chạy Nhanh Trên Windows

```bat
cd ViXNet\web_app
start.bat
```

## API Chính

- `GET /api/health`
- `GET /api/datasets`
- `GET /api/model-info`
- `POST /api/predict`
- `POST /api/analyze-model`
- `POST /api/calculate-auc`

## Lưu Ý Bảo Mật

- Không upload checkpoint từ nguồn không tin tưởng.
- Không deploy public trực tiếp bằng Flask dev server.
- Khi deploy thật cần authentication, HTTPS, giới hạn upload, cleanup file tạm và CORS chặt hơn.
- Dataset, checkpoint, thư mục `uploads/`, log và build output đã được ignore.
