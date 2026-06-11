import numpy as np
import matplotlib.pyplot as plt
import os
import json  # 新增：用于保存标签文件
from scipy.ndimage import gaussian_filter
from pathlib import Path  # 保留路径相关库

# ==============================
# 定位当前代码文件（clean_data.py）所在目录：data/raw/
RAW_DIR = Path(__file__).parent.resolve()
# 向上跳1级，定位到项目的data/根目录（核心：所有路径基于此拼接）
DATA_ROOT = RAW_DIR.parent.resolve()

# 自定义标签保存路径：data/ → generated → labels/（对应你的labels存放位置）
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)  # 自动创建文件夹，无需手动建

# 原无用路径变量注释掉，避免干扰
# DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"

# 实验配置超参数
IMAGE_SIZE = 224          # 图像尺寸
RANDOM_SEED = 42          # 随机种子（保证可复现）
NUM_CLASSES = 16          # 数据集类别数

# -------------------------z- 核心物理参数配置（160mm正方形板）--------------------------
# 板物理参数（不锈钢材质，可根据实验板实际参数修改）
PLATE_SIZE_ACTUAL = 0.16          # 实际板边长160mm（m）
PLATE_THICKNESS = 0.0008         # 板厚度0.8mm（m）
E = 200e9                        # 不锈钢弹性模量（Pa）
NU = 0.3                         # 泊松比
RHO = 7850                       # 不锈钢密度（kg/m³）
D = E * PLATE_THICKNESS**3 / (12 * (1 - NU**2))  # 弯曲刚度（N·m）

# 虚拟尺寸映射（可视化用）
PLATE_SIZE_VIRTUAL = 2.0         # 虚拟长度2.0对应实际160mm
SCALE_RATIO = PLATE_SIZE_ACTUAL / PLATE_SIZE_VIRTUAL  # 1虚拟单位=0.08m
CENTER_FIXED_RADIUS_ACTUAL = 0.003  # 实际中心固定半径3mm
CENTER_FIXED_RADIUS = CENTER_FIXED_RADIUS_ACTUAL / SCALE_RATIO  # 虚拟固定半径

# 生成配置
# 自定义图片保存路径：data/ → generated → images → clean/（核心修改）
SAVE_DIR = DATA_ROOT / "generated" / "images" / "clean"
os.makedirs(SAVE_DIR, exist_ok=True)  # 自动创建嵌套文件夹，哪怕clean/不存在也会建
SAND_GRAIN_COUNT = 12000         # 沙粒数量
GRAIN_SIZE_RANGE = (0.4, 1.0)
METAL_COLOR = (0.55, 0.6, 0.65)  # 深银灰色
SAND_COLOR = (0.95, 0.95, 0.95)  # 浅灰色沙粒
DAMPING = 0.06                   # 阻尼系数
METAL_REFLECT_STRENGTH = 0.15    # 金属反光强度
IMAGES_PER_MODAL = 100           # 新增：每个模态生成的增强图数量

# -------------------------- 模态与频率核心逻辑（公式驱动）--------------------------
# 中心固定-四边自由正方形板的频率系数λ(n,m)（仅保留n≥1、m≥1、n≠m的有效模态）
LAMBDA_MN = {
    # 等级1：极简非对称模态（频率<300Hz）
    (1, 2): 21.44,  (1, 3): 29.82,
    # 等级2：简单非对称交叉（300Hz≤频率<500Hz）
    (2, 3): 33.87,  (2, 4): 42.55,  (1, 4): 40.11,
    # 等级3：中等非对称交叉（500Hz≤频率<1000Hz）
    (2, 5): 52.78,  (3, 6): 63.54,  (1, 7): 70.25,  (4, 7): 74.12,
    # 等级4：复杂非对称斜交（1000Hz≤频率<2000Hz）
    (4, 8): 85.44,  (1, 9): 88.12,  (4, 10): 95.78,
    # 等级5：极致差异化高频（频率≥2000Hz）
    (2, 10): 98.45, (1, 10): 101.22,(5, 10): 113.98
}

def calculate_kladni_frequency(n, m):
    """
    严格套用克拉德尼频率公式（中心固定-四边自由正方形板）：
    f = (1/(2πa²)) * sqrt(D/(ρh)) * λ(n,m)
    """
    if n == 0 or m == 0 or n == m:
        raise ValueError(f"无效模态：n={n}, m={m}（需满足n≥1、m≥1、n≠m）")
    lam = LAMBDA_MN[(n, m)]
    freq = (1 / (2 * np.pi * PLATE_SIZE_ACTUAL**2)) * np.sqrt(D / (RHO * PLATE_THICKNESS)) * lam
    return round(freq)  # 取整便于使用

# 生成频率参数列表（由模态n/m驱动，而非人工赋值）
frequency_params = []
for (n, m), lam in LAMBDA_MN.items():
    freq = calculate_kladni_frequency(n, m)
    # 自动分级（基于计算出的频率）
    if freq < 300:
        level = 1
    elif freq < 500:
        level = 2
    elif freq < 1000:
        level = 3
    elif freq < 2000:
        level = 4
    else:
        level = 5
    frequency_params.append({
        "freq": freq,
        "n": n,
        "m": m,
        "lambda": lam,
        "level": level
    })

# -------------------------- 工具函数 --------------------------
def clean_filename(freq, n, m, aug_idx=None):
    """生成合法文件名（适配增强索引）"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    modal_name = f"n{n}m{m}"
    if aug_idx is not None:
        filename = f"{freq}Hz_{modal_name}_aug{aug_idx}"
    else:
        filename = f"{freq}Hz_{modal_name}"
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

def generate_160mm_metal_bg(size):
    """生成金属背景图"""
    bg = np.ones((size, size, 3)) * METAL_COLOR
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 金属反光
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # 中心固定区域
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = [0.2, 0.2, 0.2]

    # 金属纹理噪声
    bg += np.random.normal(0, 0.006, bg.shape)
    bg = np.clip(bg, 0, 1)

    # 金属板掩码
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_160mm_vibration(X, Y, n, m):
    """计算振动幅值（纯二维斜对称模态，无n/m=0分支）"""
    L = 1.0
    # ========== 新增：计算到中心的距离 + 中心衰减系数 ==========
    r = np.sqrt(X ** 2 + Y ** 2)  # 每个点到中心的距离
    alpha = 1.0  # 中心衰减系数（可调整，1.0左右最贴合实验）
    # ==========================================================
    
    # 克拉德尼二维斜对称核心公式 + 中心衰减约束（核心修改）
    vibration = np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L) - \
                np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    vibration = vibration * np.exp(-alpha * r)  # 加入中心衰减
    
    # 中心固定区域振动置0（原有逻辑保留，双重保障）
    center_mask = (X ** 2 + Y ** 2) <= CENTER_FIXED_RADIUS ** 2
    vibration[center_mask] = 0
    
    # 阻尼+归一化（原有逻辑保留）
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_160mm_sand(X, Y, plate_mask, n, m):
    """分布沙粒（修复除以零+无沙粒兜底）"""
    X_plate = X[plate_mask]
    Y_plate = Y[plate_mask]
    vibration = calculate_160mm_vibration(X_plate, Y_plate, n, m)
    vib_abs = np.abs(vibration)

    # 1. 动态阈值（兜底：无有效值时强制设为0.1）
    if len(vib_abs) == 0 or np.max(vib_abs) == 0:
        sand_threshold = 0.1
        sand_mask = np.ones_like(X_plate, dtype=bool)
    else:
        sand_threshold = np.percentile(vib_abs, 15)
        sand_mask = vib_abs < sand_threshold

    # 2. 排除中心固定区域
    center_mask_plate = (X_plate ** 2 + Y_plate ** 2) <= CENTER_FIXED_RADIUS ** 2
    sand_mask = sand_mask & (~center_mask_plate)

    sand_x = X_plate[sand_mask]
    sand_y = Y_plate[sand_mask]
    sand_vib = vib_abs[sand_mask]

    # 3. 兜底：无沙粒时随机生成基础沙粒
    if len(sand_x) == 0:
        base_count = min(SAND_GRAIN_COUNT, len(X_plate))
        idx = np.random.choice(len(X_plate), base_count, replace=False)
        sand_x = X_plate[idx]
        sand_y = Y_plate[idx]
        sand_vib = np.zeros(base_count)

    # 4. 保证沙粒数量（修复除以零）
    current_count = len(sand_x)
    if current_count < SAND_GRAIN_COUNT:
        repeat_times = int(np.ceil(SAND_GRAIN_COUNT / current_count))
        sand_x = np.tile(sand_x, repeat_times)[:SAND_GRAIN_COUNT]
        sand_y = np.tile(sand_y, repeat_times)[:SAND_GRAIN_COUNT]
        sand_vib = np.tile(sand_vib, repeat_times)[:SAND_GRAIN_COUNT]
    else:
        weights = 1 - (sand_vib / (sand_threshold + 1e-8))
        weights = weights / np.sum(weights)
        idx = np.random.choice(len(sand_x), SAND_GRAIN_COUNT, p=weights, replace=False)
        sand_x, sand_y, sand_vib = sand_x[idx], sand_y[idx], sand_vib[idx]

    # 5. 动态沙粒尺寸 + 微小随机偏移（打破矩形视觉）
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)
    sand_x += np.random.normal(0, 0.01, sand_x.shape)  # 随机偏移
    sand_y += np.random.normal(0, 0.01, sand_y.shape)

    # 6. 边界检查
    plate_bounds_mask = (np.abs(sand_x) <= 1) & (np.abs(sand_y) <= 1)
    sand_x = sand_x[plate_bounds_mask]
    sand_y = sand_y[plate_bounds_mask]
    sand_sizes = sand_sizes[plate_bounds_mask]

    # 最终兜底
    if len(sand_x) == 0:
        sand_x = np.random.uniform(-1, 1, SAND_GRAIN_COUNT)
        sand_y = np.random.uniform(-1, 1, SAND_GRAIN_COUNT)
        sand_sizes = np.random.uniform(*GRAIN_SIZE_RANGE, SAND_GRAIN_COUNT)

    return sand_x, sand_y, sand_sizes

# -------------------------- 核心生成函数 --------------------------
def generate_160mm_kladni(freq, n, m, aug_idx):
    """适配增强索引的生成函数"""
    clean_name = clean_filename(freq, n, m, aug_idx)
    img_size = IMAGE_SIZE  # 使用内联的IMAGE_SIZE
    bg, plate_mask = generate_160mm_metal_bg(img_size)

    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    sand_x, sand_y, sand_sizes = distribute_160mm_sand(X_grid, Y_grid, plate_mask, n, m)

    # 创建画布
    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")

    # 绘制模糊背景
    bg_blurred = gaussian_filter(bg, sigma=0.5)
    ax.imshow(bg_blurred, extent=[-1, 1, -1, 1], origin="lower")
    
    # 绘制沙粒
    ax.scatter(
        sand_x, sand_y,
        s=sand_sizes,
        c=[SAND_COLOR],
        alpha=0.95,
        edgecolors="none",
        marker="o",
        rasterized=True
    )

    # 保存图片
    filename = f"{SAVE_DIR}/{clean_name}.png"
    plt.savefig(
        filename, 
        bbox_inches="tight", 
        pad_inches=0, 
        dpi=100, 
        facecolor="white",
        edgecolor="none"
    )
    plt.close()
    print(f"已生成：{filename}（n={n},m={m},频率={freq}Hz,增强{aug_idx}）")

# -------------------------- 批量生成（合并为单个main块）--------------------------
if __name__ == "__main__":
    # 设置随机种子（使用内联的RANDOM_SEED）
    np.random.seed(RANDOM_SEED)
    # ========== 模态映射表（One-Hot标签核心） ==========
    # 1. 提取所有唯一模态并排序（保证索引固定）
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))  # 按n、m升序排列
    # 2. 模态→索引映射（核心：One-Hot的维度=模态总数）
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    # 3. 索引→模态反向映射（训练后验证用）
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    # 打印映射表（可选，验证索引是否正确）
    print("📌 模态-索引映射表：")
    for modal, idx in modal2idx.items():
        print(f"  模态({modal[0]},{modal[1]}) → 索引{idx}")
    print(f"🔢 One-Hot标签维度：{len(modal_list)}")
    print("="*80)

    # ========== 初始化记录器 + 标签字典 ==========
    generated_freqs = set()
    generated_mn = set()
    success_count = 0
    failed_cases = []
    label_dict = {}  # key: 图片文件名, value: One-Hot标签（列表）

    # 打印配置信息
    print("克拉德尼图案生成（公式驱动版）- 160mm中心固定-四边自由正方形板")
    print(f"板参数：边长={PLATE_SIZE_ACTUAL*1000}mm | 厚度={PLATE_THICKNESS*1000}mm | 材质=不锈钢")
    print(f"有效模态数：{len(frequency_params)} | 每个模态生成：{IMAGES_PER_MODAL}张 | 生成路径：{os.path.abspath(SAVE_DIR)}")
    print(f"预计总生成量：{len(frequency_params) * IMAGES_PER_MODAL} 张")
    print("="*80)

    # ========== 批量生成循环（嵌套增强循环） ==========
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        mn_key = (n, m)

        # 跳过重复模态（理论上公式驱动不会重复，冗余校验）
        if mn_key in generated_mn:
            print(f"\n[{param_idx}/{len(frequency_params)}] 跳过重复模态：n={n},m={m}（频率={freq}Hz）")
            continue
        
        # 嵌套循环：每个模态生成IMAGES_PER_MODAL张增强图
        for aug_idx in range(IMAGES_PER_MODAL):
            try:
                print(f"\n[{param_idx}/{len(frequency_params)}][增强{aug_idx+1}/{IMAGES_PER_MODAL}] 生成等级{info['level']}模态：n={n},m={m} → 计算频率={freq}Hz...")
                generate_160mm_kladni(freq, n, m, aug_idx)

                # 生成One-Hot标签（每个增强图对应相同标签）
                modal_idx = modal2idx[(n, m)]
                one_hot = [0] * len(modal_list)
                one_hot[modal_idx] = 1
                
                # 关联图片文件名和标签
                clean_name = clean_filename(freq, n, m, aug_idx)
                label_dict[clean_name + ".png"] = {
                    "one_hot": one_hot,
                    "modal": (n, m),
                    "modal_idx": modal_idx,
                    "freq": freq,
                    "level": info["level"],
                    "aug_idx": aug_idx
                }

                success_count += 1
            except Exception as e:
                error_info = f"n={n},m={m}_aug{aug_idx}（频率={freq}Hz）: {str(e)}"
                print(f"\n[{param_idx}/{len(frequency_params)}][增强{aug_idx+1}/{IMAGES_PER_MODAL}] 生成失败：{error_info}")
                failed_cases.append(error_info)
        
        # 标记该模态已生成
        generated_freqs.add(freq)
        generated_mn.add(mn_key)

    # ========== 保存标签文件（使用内联的LABELS_PATH） ==========
    label_save_path = LABELS_ROOT / "clean_labels.json"
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    print(f"\n📝 标签文件已保存至：{label_save_path}")

    # ========== 生成结果校验 ==========
    print("\n" + "="*80)
    print(f"✅ 生成完成！共成功生成 {success_count} 张图片（目标 {len(frequency_params) * IMAGES_PER_MODAL} 张）")
    if failed_cases:
        print(f"❌ 失败案例（{len(failed_cases)}个）：")
        for err in failed_cases:
            print(f"  - {err}")
    print(f"📊 数据统计：")
    print(f"  - 频率范围：{min(generated_freqs)}Hz ~ {max(generated_freqs)}Hz")
    print(f"  - 模态数量：{len(generated_mn)} 种（均满足n≥1、m≥1、n≠m）")
    print(f"  - 生成路径：{os.path.abspath(SAVE_DIR)}")
    print(f"  - 标签文件：{label_save_path}（包含{len(label_dict)}个标签）")
    print("="*80)