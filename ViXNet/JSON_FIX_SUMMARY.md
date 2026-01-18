# Fix for JSON Serialization Error with NumPy Types

## Problem (Vấn đề)
When saving training history to JSON files in the ViXNet folder, an error occurred because the data contained NumPy types like `int64` which cannot be directly serialized to JSON format.

Khi lưu lịch sử training vào file JSON trong thư mục ViXNet, đã xảy ra lỗi vì dữ liệu chứa các kiểu dữ liệu NumPy như `int64` không thể được serialize trực tiếp sang định dạng JSON.

### Error Example
```python
TypeError: Object of type int64 is not JSON serializable
```

## Root Cause (Nguyên nhân)
The training metrics (accuracy, loss, precision, etc.) were computed using NumPy operations, resulting in NumPy data types (np.int64, np.float64, etc.). The standard Python `json.dump()` function cannot serialize these NumPy types.

Các metrics trong quá trình training (accuracy, loss, precision, v.v.) được tính toán bằng các phép toán NumPy, dẫn đến các kiểu dữ liệu NumPy (np.int64, np.float64, v.v.). Hàm `json.dump()` của Python không thể serialize các kiểu dữ liệu NumPy này.

## Solution (Giải pháp)
Added a recursive conversion function `convert_to_json_serializable()` in `utils.py` that converts all NumPy types to Python native types before JSON serialization.

Đã thêm hàm chuyển đổi đệ quy `convert_to_json_serializable()` trong `utils.py` để chuyển đổi tất cả các kiểu dữ liệu NumPy sang kiểu dữ liệu Python gốc trước khi serialize JSON.

### What Gets Converted (Những gì được chuyển đổi)
- `np.int64, np.int32, np.int16, np.int8` → `int`
- `np.uint64, np.uint32, np.uint16, np.uint8` → `int`
- `np.float64, np.float32, np.float16` → `float`
- `np.bool_` → `bool`
- `np.ndarray` → `list`
- Nested dictionaries, lists, and tuples are handled recursively

### Changes Made (Các thay đổi)
1. **Added new function** `convert_to_json_serializable()` in `ViXNet/utils.py` (lines 315-343)
2. **Updated** `save_training_history()` function to use the converter before JSON serialization (lines 346-357)

## Files Modified (Files đã sửa)
- `ViXNet/utils.py` - Main fix implementation

## Test Files Added (Files test đã thêm)
- `ViXNet/test_json_serialization.py` - Unit tests for the conversion function
- `ViXNet/test_integration.py` - Integration test for `save_training_history()`
- `ViXNet/demo_json_fix.py` - Demonstration script showing the fix

## Verification (Xác minh)
All tests pass successfully:
- ✅ NumPy type conversion test
- ✅ Training history format test  
- ✅ Integration test with `save_training_history()`

Run tests với lệnh:
```bash
cd ViXNet
python3 test_json_serialization.py
python3 test_integration.py
python3 demo_json_fix.py
```

## Usage (Cách sử dụng)
The fix is automatically applied when calling `save_training_history()`. No changes needed to existing code.

Việc sửa lỗi được áp dụng tự động khi gọi `save_training_history()`. Không cần thay đổi code hiện có.

```python
from utils import save_training_history

# Training history with NumPy types
history = [...]  # Contains np.int64, np.float64, etc.

# This now works without errors
save_training_history(history, 'training_history.json')
```

## Impact (Ảnh hưởng)
- ✅ Minimal change - only modified the `save_training_history()` function
- ✅ Backward compatible - existing code continues to work
- ✅ No performance impact - conversion only happens during save operations
- ✅ Handles all NumPy types comprehensively

Thay đổi tối thiểu - chỉ sửa đổi hàm `save_training_history()`
