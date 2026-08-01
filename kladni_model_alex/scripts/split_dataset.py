import json
import random
import shutil
from pathlib import Path

# ================== Configuration ==================
PROJECT_ROOT = Path(__file__).parent.parent
GENERATED_ROOT = PROJECT_ROOT / "data" / "generated"
LABEL_PATH = GENERATED_ROOT / "labels" / "merged_all_labels.json"
IMG_DIRS = [
    GENERATED_ROOT / "images" / "clean",
    GENERATED_ROOT / "images" / "noise",
    GENERATED_ROOT / "images" / "random_color",
    GENERATED_ROOT / "images" / "random_matrix",
]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test_synthetic
# =========================================

random.seed(RANDOM_SEED)

# Step 1: Load labels
with open(LABEL_PATH, "r") as f:
    all_labels = json.load(f)
print(f"OK Loaded {len(all_labels)} samples from {LABEL_PATH}")

# Step 2: Build filename to full path mapping
file_to_path = {}
for img_dir in IMG_DIRS:
    if not img_dir.exists():
        print(f"WARNING: {img_dir} does not exist!")
        continue
    for img_path in img_dir.glob("*.png"):
        filename = img_path.name
        if filename in all_labels:
            file_to_path[filename] = img_path

missing_labels = set(all_labels.keys()) - set(file_to_path.keys())
if missing_labels:
    print(f"ERROR: {len(missing_labels)} labeled files not found! Examples:")
    for f in list(missing_labels)[:3]:
        print(f"  {f}")
    raise RuntimeError("Image-label mismatch!")

print(f"OK All {len(file_to_path)} images found.")

# === Optional: print class distribution for sanity check===
class_counts = {}
for meta in all_labels.values():
    mid = meta["modal_idx"]
    class_counts[mid] = class_counts.get(mid, 0) + 1
print(f"\nClass distribution:")
for mid in sorted(class_counts):
    print(f"  Class {mid}: {class_counts[mid]} images")

# Step 3: Random image-level split
all_filenames = list(all_labels.keys())
random.shuffle(all_filenames)

n = len(all_filenames)
n_train = int(SPLIT_RATIO[0] * n)
n_val = int(SPLIT_RATIO[1] * n)

train_files = all_filenames[:n_train]
val_files = all_filenames[n_train:n_train + n_val]
test_synth_files = all_filenames[n_train + n_val:]

print(f"\nSplitOptions (random image-level):")
print(f"  Train: {len(train_files)} images")
print(f"  Val:   {len(val_files)} images")
print(f"  Test (synthetic): {len(test_synth_files)} images")

# Step 4: Save subset labels
(OUTPUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)

def save_subset(files, out_path):
    subset = {f: all_labels[f] for f in files}
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)

save_subset(train_files, OUTPUT_ROOT / "labels" / "train.json")
save_subset(val_files, OUTPUT_ROOT / "labels" / "val.json")
save_subset(test_synth_files, OUTPUT_ROOT / "labels" / "test_synthetic.json")

# Step 5: Copy images
for split_name, files in [("train", train_files), ("val", val_files), ("test_synthetic", test_synth_files)]:
    dst_dir = OUTPUT_ROOT / "images" / split_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying {len(files)} images to {dst_dir}...")
    for fname in files:
        src = file_to_path[fname]
        dst = dst_dir / fname
        shutil.copy2(src, dst)

print(f"\nSuccess! Final structure:")
print(f"{OUTPUT_ROOT}/")
print(f"├── images/")
print(f"│   ├── train/")
print(f"│   ├── val/")
print(f"│   └── test_synthetic/")
print(f"└── labels/")
print(f"    ├── train.json")
print(f"    ├── val.json")
print(f"    └── test_synthetic.json")

