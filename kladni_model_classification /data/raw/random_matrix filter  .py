import numpy as np
import matplotlib.pyplot as plt
import os
import json
from random import uniform, choice, random, randint
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve  # 卷积核运算
from pathlib import Path  # 保留路径库，统一路径配置规范

# ===================== 路径配置（适配工程目录：data/raw/ → data/generated/）=====================
# 核心超参数（完全保留原配置，无修改）
RANDOM_SEED = 42  # 随机种子，保证可复现性
IMAGE_SIZE = 224
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15
IMG_PER_FREQ = 100
MIN_COLOR_DIFF = 0.3
NUM_CLASSES = 15 

# 定位当前代码文件（random_matrix.py）所在目录：data/raw/
RAW_DIR = Path(__file__).parent.resolve()
# 向上跳1级，定位到项目的data/根目录（工程所有脚本统一基准）
DATA_ROOT = RAW_DIR.parent.resolve()

# 标签保存路径：data/generated/labels/（和前3个脚本标签同目录，自动创建）
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# CNN矩阵滤镜图片保存路径：data/generated/images/random_matrix/（遵循工程设计，自动创建）
SAVE_DIR = DATA_ROOT / "generated" / "images" / "random_matrix"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- 核心物理参数配置（160mm正方形板）--------------------------
# 板物理参数（不锈钢材质，完全保留原逻辑，无修改）
PLATE_SIZE_ACTUAL = 0.16          # 实际板边长160mm（m）
PLATE_THICKNESS = 0.0008         # 板厚度0.8mm（m）
E = 200e9                        # 不锈钢弹性模量（Pa）
NU = 0.3                         # 泊松比
RHO = 7850                       # 不锈钢密度（kg/m³）
D = E * PLATE_THICKNESS**3 / (12 * (1 - NU**2))  # 弯曲刚度（N·m）

# 虚拟尺寸映射（可视化用，完全保留原逻辑，无修改）
PLATE_SIZE_VIRTUAL = 2.0         # 虚拟长度2.0对应实际160mm
SCALE_RATIO = PLATE_SIZE_ACTUAL / PLATE_SIZE_VIRTUAL  # 1虚拟单位=0.08m
CENTER_FIXED_RADIUS_ACTUAL = 0.003  # 实际中心固定半径3mm
CENTER_FIXED_RADIUS = CENTER_FIXED_RADIUS_ACTUAL / SCALE_RATIO  # 虚拟固定半径
# -------------------------- 原有噪声增强参数（完全保留）--------------------------
SAND_DISTRIBUTION_NOISE = 0.02
SAND_DENSITY_VARIATION = 0.2
GAUSSIAN_NOISE_SIGMA_RANGE = (0.005, 0.02)
OCCLUSION_PROB = 0.3
OCCLUSION_COUNT_RANGE = (1, 3)
OCCLUSION_SIZE_RANGE = (5, 15)
BLUR_PROB = 0.25
BLUR_SIGMA_RANGE = (0.3, 1.0)
GEOMETRY_PROB = 0.3
ROTATE_RANGE = (-5, 5)
SCALE_RANGE = (0.95, 1.05)
SHEAR_RANGE = (-0.03, 0.03)

# -------------------------- 新增：局部矩阵滤镜配置（类CNN卷积核）--------------------------
FILTER_COUNT_RANGE = (0, 3)  # 每张图随机0~3个局部滤镜
FILTER_KERNEL_SIZES = [3, 5, 7]  # 卷积核大小（CNN常用小矩阵）
FILTER_REGION_SIZE_RANGE = (32, 96)  # 滤镜作用的局部区域大小（32~96像素）
FILTER_OPACITY_RANGE = (0.1, 0.3)  # 局部滤镜透明度
FILTER_PADDING = 1  # 卷积填充（保持局部区域尺寸不变）

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
        type_ = "simple"
    elif freq < 500:
        level = 2
        type_ = "easy_cross"
    elif freq < 1000:
        level = 3
        type_ = "medium_cross"
    elif freq < 2000:
        level = 4
        type_ = "complex_oblique"
    else:
        level = 5
        type_ = "high_freq_diff"
    frequency_params.append({
        "freq": freq,
        "n": n,
        "m": m,
        "lambda": lam,
        "level": level,
        "type": type_
    })

# -------------------------- 工具函数（融合公式驱动+滤镜适配）--------------------------
def clean_filename(freq, n, m, img_idx, plate_color, sand_color, noise_types):
    """生成干净的文件名：频率+模态+索引+配色+噪声/滤镜信息"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    # 基础标识：频率+模态
    base_name = f"{freq}Hz_n{n}m{m}_aug{img_idx}"
    # 配色标识（简化RGB）
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    # 噪声/滤镜标识（简化，避免文件名过长）
    noise_str = "_".join([n[:4] for n in noise_types]) if noise_types else "no_noise"
    # 拼接并清理
    filename = f"{base_name}_{plate_rgb}_{sand_rgb}_{noise_str}"
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

def get_full_color_pair():
    """生成高对比度的全色域配色（金属板+沙粒）"""
    plate_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
    while True:
        sand_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
        # 计算平均色差
        color_diff = (
            abs(plate_color[0] - sand_color[0]) +
            abs(plate_color[1] - sand_color[1]) +
            abs(plate_color[2] - sand_color[2])
        ) / 3
        if color_diff >= MIN_COLOR_DIFF:
            return plate_color, sand_color

def generate_metal_bg(size, plate_color):
    """生成带金属质感的背景（含中心固定区+纹理噪声）"""
    bg = np.ones((size, size, 3)) * plate_color
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # 金属反光效果
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # 中心固定区域
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = (0.1, 0.1, 0.1)

    # 轻微噪声模拟金属纹理
    bg += np.random.normal(0, 0.008, bg.shape)
    bg = np.clip(bg, 0, 1)

    # 金属板区域掩码
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_vibration(X, Y, n, m):
    """计算金属板振动幅值（公式驱动+中心衰减）"""
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
    
    # 边缘阻尼
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    
    # 归一化
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_sand(X, Y, plate_mask, n, m):
    """分布沙粒（仅在节线区域，含兜底逻辑）"""
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

    # 动态沙粒尺寸
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)

    return sand_x, sand_y, sand_sizes

def add_sand_distribution_noise(sand_x, sand_y, sand_sizes):
    """添加沙粒位置偏移和密度随机波动"""
    offset_x = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_x))
    offset_y = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_y))
    sand_x += offset_x
    sand_y += offset_y
    current_count = len(sand_x)
    variation = int(current_count * uniform(-SAND_DENSITY_VARIATION, SAND_DENSITY_VARIATION))
    new_count = current_count + variation
    new_count = max(10000, min(14000, new_count))
    if new_count > current_count:
        supplement_x = np.random.uniform(-1, 1, new_count - current_count)
        supplement_y = np.random.uniform(-1, 1, new_count - current_count)
        supplement_sizes = np.random.uniform(*GRAIN_SIZE_RANGE, new_count - current_count)
        sand_x = np.concatenate([sand_x, supplement_x])
        sand_y = np.concatenate([sand_y, supplement_y])
        sand_sizes = np.concatenate([sand_sizes, supplement_sizes])
    elif new_count < current_count:
        idx = np.random.choice(current_count, new_count, replace=False)
        sand_x, sand_y, sand_sizes = sand_x[idx], sand_y[idx], sand_sizes[idx]
    return sand_x, sand_y, sand_sizes

# -------------------------- 核心修改：局部矩阵滤镜应用函数（完全保留逻辑）--------------------------
def get_cnn_style_kernel(kernel_size):
    """生成类CNN的小尺寸卷积核，模拟不同滤波效果"""
    kernel_type = choice([
        "edge_detect", "sharpen", "blur", "emboss",
        "sobel_x", "sobel_y", "gaussian", "laplacian"
    ])
    if kernel_size % 2 == 0:
        kernel_size += 1  # 确保奇数核

    # 经典CNN卷积核模板
    kernels = {
        "edge_detect": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "blur": np.ones((kernel_size, kernel_size)) / (kernel_size ** 2),
        "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "gaussian": np.exp(-np.linspace(-1, 1, kernel_size) ** 2).reshape(-1, 1) @
                    np.exp(-np.linspace(-1, 1, kernel_size) ** 2).reshape(1, -1),
        "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    }
    # 适配大核（如5×5）
    if kernel_size > 3 and kernel_type in ["edge_detect", "sharpen", "emboss"]:
        kernel_type = choice(["blur", "gaussian"])

    kernel = kernels[kernel_type]
    # 归一化卷积核（避免亮度溢出）
    if kernel_type not in ["blur", "gaussian"]:
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
    return kernel, kernel_type

def apply_local_cnn_filters(img):
    """
    应用类CNN的局部矩阵滤镜
    1. 随机选择局部区域
    2. 小卷积核在局部区域做卷积
    3. 透明度混合，不影响其他区域
    """
    img = np.copy(img)
    filter_info = []
    filter_count = randint(*FILTER_COUNT_RANGE)
    h, w = img.shape[:2]

    for _ in range(filter_count):
        # 1. 随机生成局部区域（x1,y1）为左上角，(x2,y2)为右下角
        region_size = randint(*FILTER_REGION_SIZE_RANGE)
        x1 = randint(0, w - region_size)
        y1 = randint(0, h - region_size)
        x2 = x1 + region_size
        y2 = y1 + region_size
        local_region = img[y1:y2, x1:x2]  # 截取局部区域

        # 2. 随机选择CNN卷积核
        kernel_size = choice(FILTER_KERNEL_SIZES)
        kernel, kernel_type = get_cnn_style_kernel(kernel_size)
        opacity = uniform(*FILTER_OPACITY_RANGE)

        # 3. 对局部区域的每个通道做卷积（类CNN操作）
        filtered_local = np.zeros_like(local_region)
        for c in range(3):
            filtered_local[:, :, c] = convolve(
                local_region[:, :, c],
                kernel,
                mode='reflect',  # 边缘反射填充
                cval=0.0
            )
        filtered_local = np.clip(filtered_local, 0, 1)

        # 4. 透明度混合：滤镜区域 = 原图 * (1-透明度) + 滤波图 * 透明度
        img[y1:y2, x1:x2] = (1 - opacity) * local_region + opacity * filtered_local
        img = np.clip(img, 0, 1)

        # 5. 记录滤镜信息（简化标识，避免文件名过长）
        filter_info.append(f"{kernel_type}_k{kernel_size}")

    return img, filter_info

# -------------------------- 图像噪声增强函数（集成局部滤镜，完全保留逻辑）--------------------------
def add_image_noise(img):
    img = np.copy(img)
    noise_types = []

    # 原有噪声增强（高斯、遮挡、模糊、几何变换）
    sigma = uniform(*GAUSSIAN_NOISE_SIGMA_RANGE)
    gaussian_noise = np.random.normal(0, sigma, img.shape)
    img = np.clip(img + gaussian_noise, 0, 1)
    noise_types.append("gaussian")

    if random() < OCCLUSION_PROB:
        occlusion_count = choice(range(OCCLUSION_COUNT_RANGE[0], OCCLUSION_COUNT_RANGE[1] + 1))
        for _ in range(occlusion_count):
            x = np.random.randint(0, IMAGE_SIZE)
            y = np.random.randint(0, IMAGE_SIZE)
            size = np.random.randint(OCCLUSION_SIZE_RANGE[0], OCCLUSION_SIZE_RANGE[1] + 1)
            occlusion_color = (uniform(0, 0.3) if random() < 0.5 else uniform(0.7, 1.0))
            x = max(size, min(IMAGE_SIZE - size, x))
            y = max(size, min(IMAGE_SIZE - size, y))
            cv2.circle(img, (x, y), size, (occlusion_color, occlusion_color, occlusion_color), thickness=-1)
        noise_types.append(f"occl{occlusion_count}")  # 简化标识

    if random() < BLUR_PROB:
        sigma = uniform(*BLUR_SIGMA_RANGE)
        img = gaussian_filter(img, sigma=sigma)
        noise_types.append(f"blur{sigma:.1f}")  # 简化标识

    if random() < GEOMETRY_PROB:
        rows, cols = img.shape[:2]
        angle = uniform(*ROTATE_RANGE)
        scale = uniform(*SCALE_RANGE)
        shear_x = uniform(*SHEAR_RANGE)
        shear_y = uniform(*SHEAR_RANGE)
        shear_mat = np.array([[1, shear_x, -shear_x * cols / 2], [shear_y, 1, -shear_y * rows / 2]], dtype=np.float32)
        img = cv2.warpAffine(img, shear_mat, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(1.0, 1.0, 1.0))
        center = (cols / 2, rows / 2)
        rotate_scale_mat = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, rotate_scale_mat, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(1.0, 1.0, 1.0))
        noise_types.append(f"geo_rot{angle:.0f}")  # 简化标识，只保留旋转角度（核心几何变换）

    # 新增：应用局部CNN矩阵滤镜
    img, filter_info = apply_local_cnn_filters(img)
    if filter_info:
        noise_types.extend(filter_info)

    return img, noise_types

# -------------------------- 核心生成函数（适配公式驱动，频率为核心）--------------------------
def generate_kladni(freq, n, m, img_idx):
    img_size = IMAGE_SIZE

    plate_color, sand_color = get_full_color_pair()
    bg, plate_mask = generate_metal_bg(img_size, plate_color)
    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)
    sand_x, sand_y, sand_sizes = distribute_sand(X_grid, Y_grid, plate_mask, n, m)
    sand_x, sand_y, sand_sizes = add_sand_distribution_noise(sand_x, sand_y, sand_sizes)

    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")
    ax.imshow(bg, extent=[-1, 1, -1, 1], origin="lower")
    ax.scatter(sand_x, sand_y, s=sand_sizes, c=[sand_color], alpha=1.0, edgecolors="none")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0, dpi=100, facecolor="white")
    plt.close()

    img = plt.imread("temp.png")[:, :, :3]
    img, noise_types = add_image_noise(img)
    os.remove("temp.png")

    # 生成唯一文件名（频率为核心）
    clean_name = clean_filename(freq, n, m, img_idx, plate_color, sand_color, noise_types)
    filename = f"{SAVE_DIR}/{clean_name}.png"

    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    if (img_idx + 1) % 10 == 0:
        print(f"  已生成 {img_idx + 1}/{IMG_PER_FREQ} 张：{os.path.basename(filename)}")
    
    return os.path.basename(filename), noise_types  # 返回文件名和噪声/滤镜信息

# -------------------------- 批量生成（适配公式驱动+频率唯一性校验）--------------------------
if __name__ == "__main__":
    # 初始化随机种子，保证生成结果可复现
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"✅ 创建含CNN滤镜数据集目录：{os.path.abspath(SAVE_DIR)}")

    # 频率唯一性校验（公式驱动已保证唯一性，双重校验）
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

    # 模态映射表（用于标签生成）
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # 打印映射表
    print("\n📌 模态-索引映射表：")
    for modal, idx in modal2idx.items():
        print(f"  模态({modal[0]},{modal[1]}) → 索引{idx}")
    print(f"🔢 One-Hot标签维度：{len(modal_list)}")
    print("="*80)

    total_imgs = len(frequency_params) * IMG_PER_FREQ
    print(f"\n📊 开始生成含CNN滤镜全色域克拉德尼图形（共 {total_imgs} 张）...")
    print(f"核心配置：")
    print(f"  - 配色规则：RGB 0~1全色域 | 最小色差：{MIN_COLOR_DIFF}")
    print(f"  - 局部矩阵滤镜：类CNN卷积核（3/5/7核）| 随机区域生效 | 0~3个滤镜/图")
    print(f"  - 卷积核类型：边缘检测/锐化/模糊/浮雕/Sobel/Gaussian等8种")
    print(f"  - 图片尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  - 频率生成：公式驱动（160mm中心固定不锈钢板）\n")

    # 初始化标签字典
    label_dict = {}

    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        type_ = info["type"]
        level = info["level"]

        print(f"[{param_idx}/{len(frequency_params)}] 处理频率：{freq}Hz（n={n}, m={m}，{type_}）")

        for img_idx in range(IMG_PER_FREQ):
            img_filename, noise_types = generate_kladni(freq, n, m, img_idx)
            
            # 生成One-Hot标签
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # 保存标签信息（含滤镜/噪声类型）
            label_dict[img_filename] = {
                "one_hot": one_hot,
                "modal": (n, m),
                "modal_idx": modal_idx,
                "freq": freq,
                "level": level,
                "type": type_,
                "noise_filter_types": noise_types
            }

        print(f"  ✅ {freq}Hz 生成完成（{IMG_PER_FREQ}张）\n")

    # 保存标签文件
    label_save_path = LABELS_ROOT / "labels_matrix_filters.json"  # 统一存到data/labels
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # 生成完成提示
    print(f"🎉 含CNN滤镜全色域数据集生成完成！")
    print(f"📁 保存路径：{os.path.abspath(SAVE_DIR)}")
    print(f"📊 数据集特征：")
    print(f"   - 总图片数：{total_imgs} 张")
    print(f"   - 唯一频率数：{len(frequency_params)} 个（公式驱动计算）")
    print(f"   - 局部滤镜：类CNN小卷积核，仅作用于随机区域，模拟真实图像处理效果")
    print(f"   - 泛化能力：适配CNN模型训练、数字万花筒纹理增强等场景")
    print(f"   - 标签文件：{label_save_path}（含One-Hot、频率、滤镜/噪声类型）")
    print(f"   - 文件名格式：{freq}Hz_n{n}m{m}_aug{idx}_配色_噪声滤镜类型.png（频率为核心标签）")