# Security Notes

Repo này phục vụ mục đích học tập/nghiên cứu.

## Model Checkpoint

Chỉ load hoặc upload file checkpoint từ nguồn tin tưởng. Checkpoint PyTorch có thể chứa payload không an toàn nếu lấy từ nguồn lạ.

## Dữ Liệu Không Nên Commit

- `.env`, secret, token, API key.
- Dataset, checkpoint, model weights.
- Log runtime, cache, build output.
- File nén hoặc tài liệu nháp chưa được phép công khai.

## Web App

Backend Flask mặc định chạy local. Nếu deploy public, cần thêm authentication, HTTPS, giới hạn kích thước upload, cleanup file tạm, rate limiting và cấu hình CORS cụ thể.
