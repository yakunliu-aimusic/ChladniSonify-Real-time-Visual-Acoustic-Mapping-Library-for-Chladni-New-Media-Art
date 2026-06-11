import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random  
from random import uniform
from scipy.ndimage import gaussian_filter
from pathlib import Path  # 保留路径库，用于内联config配置

# ===================== 路径配置（适配工程目录：data/raw/ → data/generated/）=====================
# 实验配置超参数（保留原核心参数，无修改）
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES = 15
# 生成相关配置（保留原参数，无修改）
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15

# 定位当前代码文件（random_color.py）所在目录：data/raw/
RAW_DIR = Path(__file__).parent.resolve()
# 向上跳1级，定位到项目的data/根目录（所有路径基于此拼接，工程统一）
DATA_ROOT = RAW_DIR.parent.resolve()

# 标签保存路径：data/generated/labels/（和clean/noise标签同目录，自动创建）
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# 随机配色图片保存路径：data/generated/images/random_color/（遵循工程设计，自动创建）
SAVE_DIR = DATA_ROOT / "generated" / "images" / "random_color"
os.makedirs(SAVE_DIR, exist_ok=True)

# 原无用路径变量注释，避免干扰
# DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
# -------------------------- 核心物理参数配置（160mm正方形板）--------------------------
# 板物理参数（不锈钢材质）
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

# -------------------------- 全色域增强配置 --------------------------

IMG_PER_FREQ = 100               # 每个频率生成100张增强图
MIN_COLOR_DIFF = 0.3             # 最小颜色差值（确保对比度）

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

# 生成频率参数列表（由模态n/m驱动）
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

# -------------------------- 全色域配色+噪声增强工具函数 --------------------------
def clean_filename(freq, img_idx, plate_color, sand_color, n, m):
    """生成合法文件名（包含模态+配色信息）"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    # 生成配色标识（RGB转整数）
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    # 基础文件名：频率+模态+索引+配色
    filename = f"{freq}Hz_n{n}m{m}_aug_{img_idx}_{plate_rgb}_{sand_rgb}"
    # 替换非法字符
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

def get_full_color_pair():
    """全色域随机生成板色与沙色（确保色差≥MIN_COLOR_DIFF）"""
    # 1. 随机生成板色（RGB 0~1全范围）
    plate_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))

    # 2. 生成沙色：循环校验，直到与板色色差满足要求
    while True:
        sand_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
        # 计算RGB三通道差值的平均值
        color_diff = (
            abs(plate_color[0] - sand_color[0]) +
            abs(plate_color[1] - sand_color[1]) +
            abs(plate_color[2] - sand_color[2])
        ) / 3
        # 色差达标则返回
        if color_diff >= MIN_COLOR_DIFF:
            return plate_color, sand_color

def generate_metal_bg(size, plate_color):
    """生成金属背景图（全色域+噪声增强）"""
    bg = np.ones((size, size, 3)) * plate_color
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 金属反光效果（适配任意板色）
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # 中心固定区域
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = (0.1, 0.1, 0.1)  # 固定深灰色

    # 金属纹理噪声（增强真实感，全色域适配）
    bg += np.random.normal(0, 0.008, bg.shape)  # 随机噪声
    bg = np.clip(bg, 0, 1)  # 确保RGB值在0~1范围

    # 金属板掩码
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_vibration(X, Y, n, m):
    """计算振动幅值（公式驱动+中心衰减）"""
    L = 1.0
    # 计算到中心的距离 + 中心衰减系数
    r = np.sqrt(X ** 2 + Y ** 2)
    alpha = 1.0
    
    # 克拉德尼二维斜对称核心公式 + 中心衰减
    vibration = np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L) - \
                np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    vibration = vibration * np.exp(-alpha * r)
    
    # 中心固定区域振动置0
    center_mask = (X ** 2 + Y ** 2) <= CENTER_FIXED_RADIUS ** 2
    vibration[center_mask] = 0
    
    # 阻尼+归一化
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_sand(X, Y, plate_mask, n, m):
    """分布沙粒（修复除以零+无沙粒兜底+随机偏移）"""
    X_plate = X[plate_mask]
    Y_plate = Y[plate_mask]
    vibration = calculate_vibration(X_plate, Y_plate, n, m)
    vib_abs = np.abs(vibration)

    # 动态阈值
    if len(vib_abs) == 0 or np.max(vib_abs) == 0:
        sand_threshold = 0.1
        sand_mask = np.ones_like(X_plate, dtype=bool)
    else:
        sand_threshold = np.percentile(vib_abs, 15)
        sand_mask = vib_abs < sand_threshold

    # 排除中心固定区域
    center_mask_plate = (X_plate ** 2 + Y_plate ** 2) <= CENTER_FIXED_RADIUS ** 2
    sand_mask = sand_mask & (~center_mask_plate)

    sand_x = X_plate[sand_mask]
    sand_y = Y_plate[sand_mask]
    sand_vib = vib_abs[sand_mask]

    # 兜底：无沙粒时随机生成
    if len(sand_x) == 0:
        base_count = min(SAND_GRAIN_COUNT, len(X_plate))
        idx = np.random.choice(len(X_plate), base_count, replace=False)
        sand_x = X_plate[idx]
        sand_y = Y_plate[idx]
        sand_vib = np.zeros(base_count)

    # 保证沙粒数量
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

    # 动态沙粒尺寸 + 微小随机偏移（增强随机性）
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)
    sand_x += np.random.normal(0, 0.01, sand_x.shape)  # 随机偏移噪声
    sand_y += np.random.normal(0, 0.01, sand_y.shape)

    # 边界检查
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

# -------------------------- 核心生成函数（全色域+增强）--------------------------
def generate_kladni(freq, n, m, img_idx):
    # 获取全色域配色对（板色≠沙色，高对比）
    plate_color, sand_color = get_full_color_pair()
    # 生成干净的文件名
    clean_name = clean_filename(freq, img_idx, plate_color, sand_color, n, m)
    img_size = IMAGE_SIZE

    # 生成背景和掩码（含噪声）
    bg, plate_mask = generate_metal_bg(img_size, plate_color)

    # 生成网格和沙粒分布
    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)
    sand_x, sand_y, sand_sizes = distribute_sand(X_grid, Y_grid, plate_mask, n, m)

    # 绘制图形
    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")

    # 绘制模糊背景（增强质感）
    bg_blurred = gaussian_filter(bg, sigma=0.5)
    ax.imshow(bg_blurred, extent=[-1, 1, -1, 1], origin="lower")
    
    # 绘制沙粒（全色域颜色+随机偏移）
    ax.scatter(
        sand_x, sand_y,
        s=sand_sizes,
        c=[sand_color],
        alpha=0.95,
        edgecolors="none",
        marker="o",
        rasterized=True
    )

    # 保存图片
    os.makedirs(SAVE_DIR, exist_ok=True)
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

    # 进度打印（每10张一次）
    if (img_idx + 1) % 10 == 0:
        print(f"  已生成 {img_idx + 1}/{IMG_PER_FREQ} 张：{os.path.basename(filename)}")
    
    return clean_name + ".png"  # 返回文件名用于标签生成

# -------------------------- 批量生成+One-Hot标签 --------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 初始化随机种子（保证可复现）
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建全色域数据集目录：{os.path.abspath(SAVE_DIR)}")

    # 模态映射表（One-Hot标签核心）
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # 打印映射表
    print("\n📌 模态-索引映射表：")
    for modal, idx in modal2idx.items():
        print(f"  模态({modal[0]},{modal[1]}) → 索引{idx}")
    print(f"🔢 One-Hot标签维度：{len(modal_list)}")
    print("="*80)

    # 校验频率唯一性
    freq_list = [info["freq"] for info in frequency_params]
    unique_freqs = set(freq_list)
    if len(freq_list) != len(unique_freqs):
        print(f"⚠️ 警告：检测到 {len(freq_list)-len(unique_freqs)} 个重复频率，已自动去重！")
        unique_params = []
        seen_freqs = set()
        for info in frequency_params:
            if info["freq"] not in seen_freqs:
                seen_freqs.add(info["freq"])
                unique_params.append(info)
        frequency_params = unique_params

    total_imgs = len(frequency_params) * IMG_PER_FREQ
    print(f"\n开始生成全色域克拉德尼增强图形（共 {total_imgs} 张）...")
    print(f"配色规则：RGB 0~1全色域随机 | 板色-沙色最小色差：{MIN_COLOR_DIFF}\n")

    # 初始化标签字典
    label_dict = {}

    # 遍历所有频率生成增强图
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        level = info["level"]

        print(f"[{param_idx}/{len(frequency_params)}] 处理频率：{freq}Hz（n={n}, m={m}，等级{level}）")

        # 每个频率生成100张增强图
        for img_idx in range(IMG_PER_FREQ):
            img_filename = generate_kladni(freq, n, m, img_idx)
            
            # 生成One-Hot标签
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # 保存标签信息
            label_dict[img_filename] = {
                "one_hot": one_hot,
                "modal": (n, m),
                "modal_idx": modal_idx,
                "freq": freq,
                "level": level
            }

        print(f"  ✅ {freq}Hz 增强图生成完成（{IMG_PER_FREQ}张）\n")

    # 保存标签文件
    label_save_path = LABELS_ROOT / "random_label.json"  # 统一存到data/labels
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # 生成结果统计
    print(f"🎉 全色域增强数据集生成完成！")
    print(f"📁 保存路径：{os.path.abspath(SAVE_DIR)}")
    print(f"📊 关键参数：")
    print(f"   - 总图片数：{total_imgs} 张")
    print(f"   - 唯一频率数：{len(frequency_params)} 个")
    print(f"   - 唯一模态数：{len(modal_list)} 种")
    print(f"   - 配色范围：RGB 0.0~1.0（全色域）")
    print(f"   - 色差阈值：≥{MIN_COLOR_DIFF}（确保对比度）")
    print(f"   - 图片尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   - 每个频率增强图数：{IMG_PER_FREQ} 张")
    print(f"   - 标签文件：{label_save_path}（含One-Hot标签）")
    print(f"   - 频率范围：{min(freq_list)}Hz ~ {max(freq_list)}Hz")
    print("="*80)