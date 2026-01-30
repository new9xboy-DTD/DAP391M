import os
import shutil
import random
from tqdm import tqdm

# ===== CONFIG =====
SRC_ROOT = "D:\\Repo\\DAP391m\\FaceForensics"     # Thư mục gốc có Train/Val/Test -> Fake/Real
DST_ROOT = "D:\\Repo\\DAP391m\\FaceForensics_new"  # Thư mục output sau khi chia lại

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ===== HELPER FUNCTIONS =====

def collect_all_images(src_root, label):
    """
    Gom tất cả ảnh của 1 label (Fake hoặc Real) từ Train + Val + Test
    
    Args:
        src_root: Thư mục gốc chứa Train/Val/Test
        label: 'Fake' hoặc 'Real'
    
    Returns:
        List of (src_path, filename) tuples
    """
    all_images = []
    splits = ['train', 'val', 'test']  # Support cả lowercase
    
    for split in splits:
        class_dir = os.path.join(src_root, split, label)
        if not os.path.exists(class_dir):
            continue
        
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                all_images.append((fpath, fname))
    
    return all_images


def split_images(images, train_ratio, val_ratio, test_ratio):
    """
    Chia list ảnh thành train/val/test
    
    Args:
        images: List of (src_path, filename) tuples
        train_ratio, val_ratio, test_ratio: Tỉ lệ chia
    
    Returns:
        Dict {'train': [...], 'val': [...], 'test': [...]}
    """
    random.shuffle(images)
    n = len(images)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }


def copy_images(split_dict, label, dst_root):
    """
    Copy ảnh vào thư mục đích
    
    Args:
        split_dict: Dict {'train': [...], 'val': [...], 'test': [...]}
        label: 'Fake' hoặc 'Real'
        dst_root: Thư mục đích
    """
    for split_name, images in split_dict.items():
        dst_dir = os.path.join(dst_root, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"\n📂 Copying {len(images)} {label} images to {split_name}...")
        
        for src_path, fname in tqdm(images, desc=f"{split_name}/{label}"):
            dst_path = os.path.join(dst_dir, fname)
            
            # Tránh ghi đè nếu file đã tồn tại với tên khác
            if os.path.exists(dst_path):
                # Thêm prefix để tránh trùng
                base, ext = os.path.splitext(fname)
                fname = f"{base}_{random.randint(1000, 9999)}{ext}"
                dst_path = os.path.join(dst_dir, fname)
            
            shutil.copy2(src_path, dst_path)


# ===== MAIN =====
if __name__ == "__main__":
    print("=" * 70)
    print("🔄 RESHUFFLING DATASET")
    print("=" * 70)
    print(f"   Source: {SRC_ROOT}")
    print(f"   Destination: {DST_ROOT}")
    print(f"   Train/Val/Test ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print("=" * 70)
    
    # Xác nhận trước khi chạy
    if os.path.exists(DST_ROOT):
        confirm = input(f"\n⚠️  Destination folder exists: {DST_ROOT}\n   Continue? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("❌ Cancelled.")
            exit()
    
    for label in ["Fake", "Real"]:
        print(f"\n{'='*70}")
        print(f"📁 Processing {label} images...")
        print("=" * 70)
        
        # 1. Gom tất cả ảnh
        all_images = collect_all_images(SRC_ROOT, label)
        print(f"   Total {label} images collected: {len(all_images):,}")
        
        if len(all_images) == 0:
            print(f"   ⚠️  No {label} images found! Skipping...")
            continue
        
        # 2. Chia train/val/test
        split_dict = split_images(all_images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        
        print(f"   Split results:")
        print(f"     - Train: {len(split_dict['train']):,} images")
        print(f"     - Val:   {len(split_dict['val']):,} images")
        print(f"     - Test:  {len(split_dict['test']):,} images")
        
        # 3. Copy vào thư mục đích
        copy_images(split_dict, label, DST_ROOT)
    
    print("\n" + "=" * 70)
    print("✅ Done reshuffling dataset!")
    print("=" * 70)
    
    # Thống kê kết quả
    print("\n📊 Final statistics:")
    for split in ['train', 'val', 'test']:
        for label in ['Fake', 'Real']:
            path = os.path.join(DST_ROOT, split, label)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"   {split}/{label}: {count:,} images")

