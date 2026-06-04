# data/generated/combine_labels.py
import json
import os
from pathlib import Path

# ====================================
# 路径配置：根据脚本位置自动定位项目根目录
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
    print(f"📌 标签合并配置")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"标签扫描目录：{scan_root}")
    print(f"合并文件保存路径：{save_merged_path}")
    print(f"=====================================\n")

    if not scan_root.exists():
        print(f"❌ 错误：标签扫描目录不存在！\n{scan_root}")
        return {}

    json_files = list(scan_root.glob("*.json"))
    if not json_files:
        print("❌ 错误：未找到任何JSON标签文件！")
        return {}

    print(f"🔍 找到 {len(json_files)} 个JSON文件，开始读取...\n")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            
            if not isinstance(label_data, dict):
                invalid_files.append(f"{json_file.name} → 非字典格式")
                continue

            # 扁平化合并：将所有 {filename: meta} 合并到顶层
            for img_filename, meta in label_data.items():
                if img_filename in merged_label_dict:
                    print(f"⚠️ 警告：文件名冲突！{img_filename} 已存在（来自 {json_file.name}）")
                merged_label_dict[img_filename] = meta

            scanned_file_count += 1
            print(f"✅ 读取成功：{json_file.name} → {len(label_data)} 条记录")

        except Exception as e:
            invalid_files.append(f"{json_file.name} → 错误: {str(e)[:50]}")

    # 保存合并结果
    save_merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_label_dict, f, indent=2, ensure_ascii=False)

    print(f"\n=====================================")
    print(f"🎉 合并完成！")
    print(f"📊 总计：{len(merged_label_dict)} 条唯一标签")
    print(f"📁 输出：{save_merged_path}")
    if invalid_files:
        print(f"❌ 无效文件：{invalid_files}")
    print(f"=====================================")

    return merged_label_dict

