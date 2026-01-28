import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# ===== CONFIG =====
SRC_ROOT = "D:\\Repo\\DAP391m\\Celeb_V2"     # thư mục gốc hiện tại
DST_ROOT = "D:\\Repo\\DAP391m\\FaceForensics"   # thư mục output

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ===== HELPER FUNCTIONS =====
def get_video_id(filename):
    """
    000_003_0001.png -> 000_003
    """
    return "_".join(filename.split("_")[:2])

def collect_by_video(class_dir, label):
    """
    Gom ảnh theo video ID
    """
    video_dict = defaultdict(list)
    print("class_dir:", class_dir)
    dirs = os.listdir(class_dir)
    for d in dirs:
        for root, _, files in os.walk(os.path.join(class_dir, d)):
            if label.lower() == "fake":
                
                vids = os.listdir(root)
                for vid in vids:
                    for root2, _, files2 in os.walk(os.path.join(root, vid)):
                        filterd = files2[::2]
                        for f in filterd:
                            if not f.lower().endswith(".png"):
                                continue
                            video_dict[vid].append(os.path.join(root2, f))
            else:
                for f in files:
                    if not f.lower().endswith(".png"):
                        continue
                    vid = d
                    video_dict[vid].append(os.path.join(root, f))
    return video_dict

def split_video_ids(video_ids):
    random.shuffle(video_ids)
    n = len(video_ids)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    return {
        "train": video_ids[:n_train],
        "val": video_ids[n_train:n_train + n_val],
        "test": video_ids[n_train + n_val:]
    }

def copy_split(video_split, video_dict, label):
    for split, vids in video_split.items():
        for vid in tqdm(vids):
            for src_path in video_dict[vid]:
                method = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
                vid = os.path.basename(os.path.dirname(src_path))
                dst_dir = os.path.join(DST_ROOT, split, label)
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.exists(os.path.join(dst_dir, f"{method}_{vid}_{os.path.basename(src_path)}")):
                    shutil.copy(src_path, os.path.join(dst_dir, f"{method}_{vid}_{os.path.basename(src_path)}"))

def collect_img_paths(class_dir, label):
    img_paths = os.listdir(os.path.join(class_dir, ))
    return img_paths

# ===== MAIN =====
for split_type in ["Train", "Val"]:
    for label in ["Fake", "Real"]:
        class_dir = os.path.join(SRC_ROOT, split_type, label)
        img_paths = collect_img_paths(class_dir, label)[::2]
        
        print(f"Collected {len(img_paths)} images for {label} in {split_type} set.")
        for img_path in tqdm(img_paths):
            dst_dir = os.path.join(DST_ROOT, split_type.lower(), label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(os.path.join(class_dir, img_path), os.path.join(dst_dir, f"CelebV2_{os.path.basename(img_path)}"))
        # video_ids = list(video_dict.keys())
        # split = split_video_ids(img_paths)

        # copy_split(split, video_dict, label)

print("✅ Done splitting dataset!")
