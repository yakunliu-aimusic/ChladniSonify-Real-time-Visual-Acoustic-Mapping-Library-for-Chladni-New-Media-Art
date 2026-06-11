import numpy as np
import matplotlib.pyplot as plt
import os
import json
from random import uniform, choice, random, randint
import cv2
from scipy.ndimage import gaussian_filter
from pathlib import Path  # 保留路径库，用于内联config配置

# ===================== 原config配置完全内联（移除外部导入，其余无修改）=====================
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES = 15
# 生成相关配置（原代码中已定义，此处保留与config对齐）
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15


# 定位当前代码文件（noise.py）所在目录：data/raw/
RAW_DIR = Path(__file__).parent.resolve()
# 向上跳1级，定位到项目的data/根目录（所有路径基于此拼接，和clean_data.py统一）
DATA_ROOT = RAW_DIR.parent.resolve()

# 标签保存路径：data/generated/labels/（自动创建，和clean标签同目录）
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# 噪声图片保存路径：data/generated/images/noise/（遵循工程设计，自动创建）
SAVE_DIR = DATA_ROOT / "generated" / "images" / "noise"
os.makedirs(SAVE_DIR, exist_ok=True)
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

# -------------------------- 全色域+噪声增强核心配置 --------------------------
IMG_PER_FREQ = 100  # 每个频率生成100张增强图
MIN_COLOR_DIFF = 0.3  # 金属板与沙粒最小色差（保证对比度）

# -------------------------- 噪声增强参数（移除旋转相关）--------------------------
# 沙粒分布噪声
SAND_DISTRIBUTION_NOISE = 0.02  # 沙粒位置随机偏移幅度
SAND_DENSITY_VARIATION = 0.2  # 沙粒密度随机波动比例（±20%）
# 高斯噪声
GAUSSIAN_NOISE_SIGMA_RANGE = (0.005, 0.02)  # 噪声强度范围
# 遮挡噪声
OCCLUSION_PROB = 0.3  # 出现遮挡的概率
OCCLUSION_COUNT_RANGE = (1, 3)  # 遮挡物数量
OCCLUSION_SIZE_RANGE = (5, 15)  # 遮挡物尺寸（像素）
# 模糊噪声
BLUR_PROB = 0.25  # 出现模糊的概率
BLUR_SIGMA_RANGE = (0.3, 1.0)  # 模糊程度范围
# 原有几何变换（删除旋转相关）
GEOMETRY_PROB = 0.3  # 出现剪切/缩放的概率
SCALE_RANGE = (0.95, 1.05)  # 原有缩放比例范围
SHEAR_RANGE = (-0.03, 0.03)  # 剪切强度范围

# -------------------------- 扩展几何变换参数（移除旋转相关）--------------------------
TRANSLATE_RANGE = (-10, 10)  # 平移范围（像素，x/y方向独立）
EXTENDED_SCALE_RANGE = (0.8, 1.2)  # 扩展缩放比例范围
FLIP_PROB = 0.5  # 水平/垂直翻转的总概率
# 修复：用具体数值表示翻转方向（OpenCV标准）
FLIP_TYPES = [None, 1, 0]  # 翻转类型（无/水平/垂直）

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

# 生成频率参数列表（由模态n/m驱动，补充label/type字段）
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
    # 生成label（模态+频率）
    label = f"n{n}m{m}_{freq}Hz_level{level}"
    frequency_params.append({
        "freq": freq,
        "n": n,
        "m": m,
        "lambda": lam,
        "level": level,
        "label": label,
        "type": type_
    })

# -------------------------- 工具函数（保留+适配新参数）--------------------------
def clean_filename(label):
    """清理文件名中的非法字符"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in illegal_chars:
        label = label.replace(char, '_')
    return label

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

# -------------------------- 沙粒分布噪声函数（保留原逻辑）--------------------------
def add_sand_distribution_noise(sand_x, sand_y, sand_sizes):
    """添加沙粒位置偏移和密度随机波动"""
    # 1. 沙粒位置随机偏移
    offset_x = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_x))
    offset_y = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_y))
    sand_x += offset_x
    sand_y += offset_y

    # 2. 沙粒密度随机波动
    current_count = len(sand_x)
    variation = int(current_count * uniform(-SAND_DENSITY_VARIATION, SAND_DENSITY_VARIATION))
    new_count = current_count + variation
    new_count = max(10000, min(14000, new_count))  # 限制数量范围

    if new_count > current_count:
        # 补充沙粒
        supplement_x = np.random.uniform(-1, 1, new_count - current_count)
        supplement_y = np.random.uniform(-1, 1, new_count - current_count)
        supplement_sizes = np.random.uniform(*GRAIN_SIZE_RANGE, new_count - current_count)
        sand_x = np.concatenate([sand_x, supplement_x])
        sand_y = np.concatenate([sand_y, supplement_y])
        sand_sizes = np.concatenate([sand_sizes, supplement_sizes])
    elif new_count < current_count:
        # 减少沙粒
        idx = np.random.choice(current_count, new_count, replace=False)
        sand_x, sand_y, sand_sizes = sand_x[idx], sand_y[idx], sand_sizes[idx]

    return sand_x, sand_y, sand_sizes

# -------------------------- 扩展几何变换函数（移除旋转）--------------------------
def apply_extended_geometry_transforms(img):
    """应用随机平移、缩放、翻转（移除旋转）"""
    img = np.copy(img)
    transform_info = []
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # 1. 随机平移
    tx = randint(*TRANSLATE_RANGE)
    ty = randint(*TRANSLATE_RANGE)
    translate_mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(
        img, translate_mat, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0)
    )
    transform_info.append(f"translate(tx={tx},ty={ty})")

    # 2. 随机缩放（移除旋转后序号调整）
    scale = uniform(*EXTENDED_SCALE_RANGE)
    scale_mat = cv2.getRotationMatrix2D(center, 0, scale)  # 旋转角度固定为0
    img = cv2.warpAffine(
        img, scale_mat, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0)
    )
    transform_info.append(f"scale({scale:.2f})")

    # 3. 随机翻转（修复逻辑）
    if random() < FLIP_PROB:
        flip_type = choice(FLIP_TYPES[1:])  # 排除None
        img = cv2.flip(img, flip_type)
        flip_str = "flip_horizontal" if flip_type == 1 else "flip_vertical"
        transform_info.append(flip_str)

    return img, transform_info

# -------------------------- 图像噪声增强函数（移除旋转相关）--------------------------
def add_image_noise(img):
    """多维度噪声增强：高斯+遮挡+模糊+几何变换+扩展几何变换（移除旋转）"""
    img = np.copy(img)
    noise_types = []

    # 1. 高斯噪声
    sigma = uniform(*GAUSSIAN_NOISE_SIGMA_RANGE)
    gaussian_noise = np.random.normal(0, sigma, img.shape)
    img = np.clip(img + gaussian_noise, 0, 1)
    noise_types.append("gaussian_noise")

    # 2. 遮挡噪声
    if random() < OCCLUSION_PROB:
        occlusion_count = randint(*OCCLUSION_COUNT_RANGE)
        for _ in range(occlusion_count):
            x = randint(0, IMAGE_SIZE)
            y = randint(0, IMAGE_SIZE)
            size = randint(*OCCLUSION_SIZE_RANGE)
            # 随机遮挡颜色（暗斑/亮斑）
            occlusion_color = uniform(0, 0.3) if random() < 0.5 else uniform(0.7, 1.0)
            # 确保遮挡在图像范围内
            x = max(size, min(IMAGE_SIZE - size, x))
            y = max(size, min(IMAGE_SIZE - size, y))
            cv2.circle(
                img, (x, y), size,
                (occlusion_color, occlusion_color, occlusion_color),
                thickness=-1
            )
        noise_types.append(f"occlusion_{occlusion_count}")

    # 3. 模糊噪声
    if random() < BLUR_PROB:
        sigma = uniform(*BLUR_SIGMA_RANGE)
        img = gaussian_filter(img, sigma=sigma)
        noise_types.append(f"blur_{sigma:.2f}")

    # 4. 原有几何变换（仅保留剪切+缩放，移除旋转）
    if random() < GEOMETRY_PROB:
        rows, cols = img.shape[:2]
        scale = uniform(*SCALE_RANGE)
        shear_x = uniform(*SHEAR_RANGE)
        shear_y = uniform(*SHEAR_RANGE)

        # 剪切变换
        shear_mat = np.array([
            [1, shear_x, -shear_x * cols / 2],
            [shear_y, 1, -shear_y * rows / 2]
        ], dtype=np.float32)
        img = cv2.warpAffine(
            img, shear_mat, (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(1.0, 1.0, 1.0)
        )

        noise_types.append(f"geo_scale{scale:.2f}_shear{shear_x:.3f}")

    # 5. 扩展几何变换（已移除旋转）
    img, transform_info = apply_extended_geometry_transforms(img)
    noise_types.extend(transform_info)

    return img, noise_types

# -------------------------- 核心生成函数（融合噪声+全色域+公式驱动）--------------------------
def generate_kladni(freq, n, m, label, img_idx, type_):
    """生成单张含噪声的克拉尼图案"""
    clean_label = clean_filename(label)
    img_size = IMAGE_SIZE

    # 1. 生成全色域配色
    plate_color, sand_color = get_full_color_pair()

    # 2. 生成金属背景
    bg, plate_mask = generate_metal_bg(img_size, plate_color)

    # 3. 生成沙粒分布并添加噪声
    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)
    sand_x, sand_y, sand_sizes = distribute_sand(X_grid, Y_grid, plate_mask, n, m)
    sand_x, sand_y, sand_sizes = add_sand_distribution_noise(sand_x, sand_y, sand_sizes)

    # 4. 绘制基础图形
    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")
    ax.imshow(bg, extent=[-1, 1, -1, 1], origin="lower")
    ax.scatter(
        sand_x, sand_y,
        s=sand_sizes,
        c=[sand_color],
        alpha=1.0,
        edgecolors="none"
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # 5. 保存临时图像
    os.makedirs(SAVE_DIR, exist_ok=True)
    temp_file = os.path.join(SAVE_DIR, "temp_kladni.png")  
    plt.savefig(temp_file, bbox_inches="tight", pad_inches=0, dpi=100, facecolor="white")
    plt.close()

    # 6. 读取并添加噪声
    img = plt.imread(temp_file)[:, :, :3]  # 去除alpha通道
    img, noise_types = add_image_noise(img)
    os.remove(temp_file)  # 删除临时文件

    # 7. 生成最终文件名（包含频率、索引、噪声、配色信息）
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    noise_str = "_".join([n.replace(" ", "_") for n in noise_types])
    filename = f"{SAVE_DIR}/{freq}Hz_{clean_label}_{img_idx}_{plate_rgb}_{sand_rgb}_{noise_str}.png"

    # 8. 保存最终图像（转换为uint8避免保存异常）
    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    # 进度打印
    if (img_idx + 1) % 10 == 0:
        print(f"    已生成 {img_idx + 1}/{IMG_PER_FREQ} 张：{os.path.basename(filename)}")
    
    return os.path.basename(filename), noise_types  # 返回文件名和噪声信息用于标签

# -------------------------- 批量生成主逻辑（新增One-Hot标签）--------------------------
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)  # 与训练/生成脚本种子一致
    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"✅ 创建数据集目录：{os.path.abspath(SAVE_DIR)}")

    # 模态映射表（One-Hot标签核心）
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # 打印映射表
    print("\n📌 模态-索引映射表：")
    for modal, idx in modal2idx.items():
        print(f"  模态({modal[0]},{modal[1]}) → 索引{idx}")
    print(f"🔢 One-Hot标签维度：{len(modal_list)}")
    if NUM_CLASSES != len(modal_list):
        print(f"⚠️ 警告：config.NUM_CLASSES={NUM_CLASSES} 与实际模态数={len(modal_list)}不匹配！")
    print("="*80)

    # 打印配置信息（更新描述，移除旋转）
    total_freq = len(frequency_params)
    total_imgs = total_freq * IMG_PER_FREQ
    print(f"\n📊 开始生成含噪声克拉尼图案数据集：")
    print(f"   - 总频率数：{total_freq} 组（公式驱动计算）")
    print(f"   - 每组生成：{IMG_PER_FREQ} 张")
    print(f"   - 总图片数：{total_imgs} 张")
    print(f"   - 图片尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   - 噪声类型：沙粒分布噪声、高斯噪声、遮挡、模糊、几何变换（剪切/缩放/平移/翻转）\n")

    # 初始化标签字典
    label_dict = {}

    # 批量生成
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        label = info["label"]
        type_ = info["type"]
        level = info["level"]

        print(f"[{param_idx}/{total_freq}] 处理 {type_} 模式：{freq}Hz（n={n}, m={m}，等级{level}）")
        
        # 每个频率生成100张增强图
        for img_idx in range(IMG_PER_FREQ):
            img_filename, noise_types = generate_kladni(freq, n, m, label, img_idx, type_)
            
            # 生成One-Hot标签
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # 保存标签信息到字典 (所有标签信息都存在这里，生成 labels.json 文件的数据源)
            label_dict[img_filename] = {
                "one_hot": one_hot, # ✅ 最关键：模型训练用的 独热编码标签 (核心！)
                "modal": (n, m), # 模态编号 (1,2)/(1,3)等
                "modal_idx": modal_idx, # 模态对应的数字索引 (0~13)
                "freq": freq, # 频率值
                "level": level, # 频率等级
                "type": type_,  # 模态类型
                "noise_types": noise_types # 噪声增强类型
            }

        print(f"    ✅ {freq}Hz 模式生成完成\n")

    # 保存标签文件
    label_save_path = LABELS_ROOT / "noise_label.json"
    # ======================================================
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # 生成完成提示（更新描述，移除旋转）
    print(f"🎉 数据集生成完成！")
    print(f"📁 保存路径：{os.path.abspath(SAVE_DIR)}")
    print(f"🔍 数据集特征：")
    print(f"   - 配色：全色域RGB，最小色差{MIN_COLOR_DIFF}（保证对比度）")
    print(f"   - 噪声：7类增强（覆盖真实拍摄场景的噪声/变换）")
    print(f"   - 频率：公式驱动计算（中心固定-四边自由160mm不锈钢板）")
    print(f"   - 模态：{len(modal_list)}种有效非对称模态（n≥1、m≥1、n≠m）")
    print(f"   - 标签：{label_save_path}（含One-Hot、噪声类型等信息）")
    print(f"   - 泛化性：支持平移、缩放、翻转等几何变换，适配不同拍摄角度（已移除旋转）")