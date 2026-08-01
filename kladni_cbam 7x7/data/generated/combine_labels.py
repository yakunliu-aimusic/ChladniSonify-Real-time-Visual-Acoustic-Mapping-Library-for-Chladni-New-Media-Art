# data/generated/combine_labels.py
import json
import os
from pathlib import Path

# ====================================
# Path config: resolve project root from script location
# ====================================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent  # my_chladni_project/
SCAN_LABELS_ROOT = PROJECT_ROOT / "generated" / "labels"
MERGED_LABELS_PATH = SCAN_LABELS_ROOT / "merged_all_labels.json"

def merge_all_label_files(scan_root: Path, save_merged_path: Path):
    merged_label_dict = {}
    scanned_file_count = 0
    invalid_files = []

    print(f"\n=====================================")
    print(f"Label merge configuration")
    print(f"Project root:{PROJECT_ROOT}")
    print(f"Label scan directory:{scan_root}")
    print(f"Merged file save path:{save_merged_path}")
    print(f"=====================================\n")

    if not scan_root.exists():
        print(f"ERROR: Label scan directory does not exist!\n{scan_root}")
        return {}

    json_files = list(scan_root.glob("*.json"))
    if not json_files:
        print("ERROR: no JSON label files found!")
        return {}

    print(f"Found JSON files, starting read...\n")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            
            if not isinstance(label_data, dict):
                invalid_files.append(f"{json_file.name} -> Not a dict format")
                continue

            # Flatten merge: combine all {filename: meta} into top level
            for img_filename, meta in label_data.items():
                if img_filename in merged_label_dict:
                    print(f"WARNING: filename conflict!{img_filename} already exists (from {json_file.name})")
                merged_label_dict[img_filename] = meta

            scanned_file_count += 1
            print(f"Read OK:{json_file.name} -> {len(label_data)} records")

        except Exception as e:
            invalid_files.append(f"{json_file.name} -> Error: {str(e)[:50]}")

    # Save merged result
    save_merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_label_dict, f, indent=2, ensure_ascii=False)

    print(f"\n=====================================")
    print(f"Merge complete.")
    print(f"Total:{len(merged_label_dict)} unique labels")
    print(f"Output:{save_merged_path}")
    if invalid_files:
        print(f"ERROR: Invalid files:{invalid_files}")
    print(f"=====================================")

    return merged_label_dict

