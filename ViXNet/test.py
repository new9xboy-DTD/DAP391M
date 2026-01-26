import os
import shutil
import random
from collections import defaultdict

# ===== CONFIG =====
SRC_ROOT = "dataset_root"     # thư mục gốc hiện tại
DST_ROOT = "dataset_split"   # thư mục output

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

def collect_by_video(class_dir):
    """
    Gom ảnh theo video ID
    """
    video_dict = defaultdict(list)

    for root, _, files in os.walk(class_dir):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            vid = get_video_id(f)
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
        for vid in vids:
            for src_path in video_dict[vid]:
                method = os.path.basename(os.path.dirname(src_path))
                dst_dir = os.path.join(DST_ROOT, split, label, method)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src_path, dst_dir)

# ===== MAIN =====
for label in ["fake", "real"]:
    class_dir = os.path.join(SRC_ROOT, label)
    video_dict = collect_by_video(class_dir)

    video_ids = list(video_dict.keys())
    split = split_video_ids(video_ids)

    copy_split(split, video_dict, label)

print("✅ Done splitting dataset!")
