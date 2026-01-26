import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# ===== CONFIG =====
SRC_ROOT = "D:\\FPT\\DAP391m\\FF++"     # thư mục gốc hiện tại
DST_ROOT = "D:\\FPT\\DAP391m\\FaceForensics"   # thư mục output

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

# ===== MAIN =====
for label in ["Fake"]:
    class_dir = os.path.join(SRC_ROOT, label.lower())
    video_dict = collect_by_video(class_dir, label)

    video_ids = list(video_dict.keys())
    split = split_video_ids(video_ids)

    copy_split(split, video_dict, label)

print("✅ Done splitting dataset!")
