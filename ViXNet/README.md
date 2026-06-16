# ViXNet

ViXNet là triển khai deepfake detection kết hợp hai nhánh:

- **Xception branch** để trích xuất đặc trưng không gian.
- **Vision Transformer branch** để học quan hệ patch/token.
- **Fusion/classifier** để dự đoán ảnh `Real` hoặc `Fake`.

## Cấu Trúc

```text
ViXNet/
  config.py             # Cấu hình dataset, training, checkpoint
  dataset.py            # Dataset loader và transform
  model.py              # Kiến trúc model
  model_factory.py      # Tạo/load nhiều biến thể model
  train.py              # Script huấn luyện
  test.py               # Script kiểm thử
  inference_example.py  # Ví dụ inference
  utils.py              # Metric, checkpoint, helper
  web_app/              # React + Flask demo app
```

## Dataset

Mặc định code đọc dữ liệu từ `../data/FaceForensics_new`. Có thể đổi bằng env var:

```bash
VIXNET_DATA_ROOT=/path/to/data
VIXNET_DEFAULT_DATASET=/path/to/FaceForensics_new
```

Cấu trúc dataset:

```text
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

## Huấn Luyện

```bash
cd ViXNet
python train.py
```

Các checkpoint sinh ra trong quá trình huấn luyện không nên commit vào Git.

## Inference

```bash
cd ViXNet
python inference_example.py
```

Chỉnh `checkpoint_path` và đường dẫn ảnh trong script theo model/dataset của bạn.

## Web Demo

Xem hướng dẫn ở [web_app/README.md](web_app/README.md).

## Lưu Ý

- Model checkpoint chỉ nên load từ nguồn tin tưởng.
- Dataset và weights không được đưa vào repo public.
- Kết quả thực nghiệm phụ thuộc mạnh vào dataset, preprocessing và checkpoint sử dụng.
