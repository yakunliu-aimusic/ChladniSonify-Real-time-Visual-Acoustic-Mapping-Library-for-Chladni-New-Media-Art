import numpy as np
import matplotlib.pyplot as plt
import os
import json
from random import uniform, choice, random, randint
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve  # Convolution operations
from pathlib import Path  # Path utilities for unified path conventions

# ===================== Path config: data/raw/ -> data/generated/=====================
# Core hyperparameters (unchanged)
RANDOM_SEED = 42  # Random seed for reproducibility
IMAGE_SIZE = 224
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15
IMG_PER_FREQ = 100
MIN_COLOR_DIFF = 0.3
NUM_CLASSES = 15 

# Script directory: data/raw/ (random_matrix.py)
RAW_DIR = Path(__file__).parent.resolve()
# Parent dir: project data/ root (shared baseline for all scripts)
DATA_ROOT = RAW_DIR.parent.resolve()

# Label path: data/generated/labels/ (shared with first 3 scripts, auto-created)
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# Matrix filter image path: data/generated/images/random_matrix/
SAVE_DIR = DATA_ROOT / "generated" / "images" / "random_matrix"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- Core physical parameters (160mm square plate)--------------------------
# Plate parameters (stainless steel; logic unchanged)
PLATE_SIZE_ACTUAL = 0.16          # Actual plate edge length 160mm (m)
PLATE_THICKNESS = 0.0008         # Plate thickness 0.8mm (m)
E = 200e9                        # Stainless steel Young's modulus (Pa)
NU = 0.3                         # Poisson's ratio
RHO = 7850                       # Stainless steel density (kg/m^3)
D = E * PLATE_THICKNESS**3 / (12 * (1 - NU**2))  # Flexural rigidity (N*m)

# Virtual size mapping (visualization; logic unchanged)
PLATE_SIZE_VIRTUAL = 2.0         # Virtual length 2.0 maps to 160mm
SCALE_RATIO = PLATE_SIZE_ACTUAL / PLATE_SIZE_VIRTUAL  # 1 virtual unit = 0.08m
CENTER_FIXED_RADIUS_ACTUAL = 0.003  # Actual center fixation radius 3mm
CENTER_FIXED_RADIUS = CENTER_FIXED_RADIUS_ACTUAL / SCALE_RATIO  # Virtual fixation radius
# -------------------------- Original noise augmentation params (unchanged) --------------------------
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

# -------------------------- Local matrix filter config (CNN-style kernels)--------------------------
FILTER_COUNT_RANGE = (0, 3)  # 0-3 local filters per image
FILTER_KERNEL_SIZES = [3, 5, 7]  # Kernel sizes (CNN-style small matrices)
FILTER_REGION_SIZE_RANGE = (32, 96)  # Local filter region size (32-96 px)
FILTER_OPACITY_RANGE = (0.1, 0.3)  # Local filter opacity
FILTER_PADDING = 1  # Convolution padding (preserve local region size)

# -------------------------- Modal and frequency logic (formula-driven)--------------------------
# Frequency coefficients lambda(n,m) for center-fixed, four-edge-free square plate (valid modes: n >= 1, m >= 1, n != m)
LAMBDA_MN = {
    # Level 1: minimal asymmetric modes (freq < 300Hz)
    (1, 2): 21.44,  (1, 3): 29.82,
    # Level 2: simple asymmetric crossing (300Hz <= freq < 500Hz)
    (2, 3): 33.87,  (2, 4): 42.55,  (1, 4): 40.11,
    # Level 3: moderate asymmetric crossing (500Hz <= freq < 1000Hz)
    (2, 5): 52.78,  (3, 6): 63.54,  (1, 7): 70.25,  (4, 7): 74.12,
    # Level 4: complex asymmetric oblique (1000Hz <= freq < 2000Hz)
    (4, 8): 85.44,  (1, 9): 88.12,  (4, 10): 95.78,
    # Level 5: high-frequency differentiation (freq >= 2000Hz)
    (2, 10): 98.45, (1, 10): 101.22,(5, 10): 113.98
}

def calculate_kladni_frequency(n, m):
    """
    Chladni frequency formula (center-fixed, four-edge-free square plate):
    f = (1/(2pia²)) * sqrt(D/(ρh)) * lambda(n,m)
    """
    if n == 0 or m == 0 or n == m:
        raise ValueError(f"Invalid mode:n={n}, m={m}(requires n >= 1, m >= 1, n != m)")
    lam = LAMBDA_MN[(n, m)]
    freq = (1 / (2 * np.pi * PLATE_SIZE_ACTUAL**2)) * np.sqrt(D / (RHO * PLATE_THICKNESS)) * lam
    return round(freq)  # Round to integer for practical use

# Build frequency list driven by modal n/m
frequency_params = []
for (n, m), lam in LAMBDA_MN.items():
    freq = calculate_kladni_frequency(n, m)
    # Auto level assignment based on computed frequency
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

# -------------------------- Utilities (formula-driven + filter adaptation) --------------------------
def clean_filename(freq, n, m, img_idx, plate_color, sand_color, noise_types):
    """Generate clean filename: freq+mode+index+color+noise/filter info"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    # Base ID: frequency + mode
    base_name = f"{freq}Hz_n{n}m{m}_aug{img_idx}"
    # Short color ID from RGB
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    # Short noise/filter ID in filename
    noise_str = "_".join([n[:4] for n in noise_types]) if noise_types else "no_noise"
    # Join and sanitize
    filename = f"{base_name}_{plate_rgb}_{sand_rgb}_{noise_str}"
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

def get_full_color_pair():
    """Generate high-contrast full-gamut plate/sand colors"""
    plate_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
    while True:
        sand_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
        # Compute average color difference
        color_diff = (
            abs(plate_color[0] - sand_color[0]) +
            abs(plate_color[1] - sand_color[1]) +
            abs(plate_color[2] - sand_color[2])
        ) / 3
        if color_diff >= MIN_COLOR_DIFF:
            return plate_color, sand_color

def generate_metal_bg(size, plate_color):
    """Generate metal background with center fixation and texture noise"""
    bg = np.ones((size, size, 3)) * plate_color
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # Metal reflection effect
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # Center fixation region
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = (0.1, 0.1, 0.1)

    # Light noise for metal texture
    bg += np.random.normal(0, 0.008, bg.shape)
    bg = np.clip(bg, 0, 1)

    # Plate region mask
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_vibration(X, Y, n, m):
    """Compute plate vibration amplitude (formula + center attenuation)"""
    L = 1.0
    # Distance to center + attenuation coefficient
    r = np.sqrt(X ** 2 + Y ** 2)
    alpha = 1.0
    
    # Chladni 2D skew-symmetric formula with center attenuation
    vibration = np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L) - \
                np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    vibration = vibration * np.exp(-alpha * r)
    
    # Zero vibration in center fixation region
    center_mask = (X ** 2 + Y ** 2) <= CENTER_FIXED_RADIUS ** 2
    vibration[center_mask] = 0
    
    # Edge damping
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    
    # Normalize
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_sand(X, Y, plate_mask, n, m):
    """Distribute sand grains on nodal regions with fallback"""
    X_plate = X[plate_mask]
    Y_plate = Y[plate_mask]
    vibration = calculate_vibration(X_plate, Y_plate, n, m)
    vib_abs = np.abs(vibration)

    # Dynamic threshold
    if len(vib_abs) == 0 or np.max(vib_abs) == 0:
        sand_threshold = 0.1
        sand_mask = np.ones_like(X_plate, dtype=bool)
    else:
        sand_threshold = np.percentile(vib_abs, 15)
        sand_mask = vib_abs < sand_threshold

    # Exclude center fixation region
    center_mask_plate = (X_plate ** 2 + Y_plate ** 2) <= CENTER_FIXED_RADIUS ** 2
    sand_mask = sand_mask & (~center_mask_plate)

    sand_x = X_plate[sand_mask]
    sand_y = Y_plate[sand_mask]
    sand_vib = vib_abs[sand_mask]

    # Fallback: random sand when empty
    if len(sand_x) == 0:
        base_count = min(SAND_GRAIN_COUNT, len(X_plate))
        idx = np.random.choice(len(X_plate), base_count, replace=False)
        sand_x = X_plate[idx]
        sand_y = Y_plate[idx]
        sand_vib = np.zeros(base_count)

    # Ensure sand grain count
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

    # Dynamic sand grain size
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)

    return sand_x, sand_y, sand_sizes

def add_sand_distribution_noise(sand_x, sand_y, sand_sizes):
    """Add sand position offset and density variation"""
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

# -------------------------- Local matrix filter application (logic unchanged)--------------------------
def get_cnn_style_kernel(kernel_size):
    """Generate CNN-style small kernels for filter effects"""
    kernel_type = choice([
        "edge_detect", "sharpen", "blur", "emboss",
        "sobel_x", "sobel_y", "gaussian", "laplacian"
    ])
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size

    # Classic CNN kernel templates
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
    # Adapt for large kernels (e.g. 5x5)
    if kernel_size > 3 and kernel_type in ["edge_detect", "sharpen", "emboss"]:
        kernel_type = choice(["blur", "gaussian"])

    kernel = kernels[kernel_type]
    # Normalize kernel (prevent brightness overflow)
    if kernel_type not in ["blur", "gaussian"]:
        kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)
    return kernel, kernel_type

def apply_local_cnn_filters(img):
    """
    Apply CNN-style local matrix filters
    1. Randomly select local region
    2. Apply small kernel convolution locally
    3. Alpha blend without affecting other regions
    """
    img = np.copy(img)
    filter_info = []
    filter_count = randint(*FILTER_COUNT_RANGE)
    h, w = img.shape[:2]

    for _ in range(filter_count):
        # 1. Random local region (x1,y1) top-left, (x2,y2) bottom-right
        region_size = randint(*FILTER_REGION_SIZE_RANGE)
        x1 = randint(0, w - region_size)
        y1 = randint(0, h - region_size)
        x2 = x1 + region_size
        y2 = y1 + region_size
        local_region = img[y1:y2, x1:x2]  # Crop local region

        # 2. Random CNN kernel
        kernel_size = choice(FILTER_KERNEL_SIZES)
        kernel, kernel_type = get_cnn_style_kernel(kernel_size)
        opacity = uniform(*FILTER_OPACITY_RANGE)

        # 3. Convolve each channel locally (CNN-style)
        filtered_local = np.zeros_like(local_region)
        for c in range(3):
            filtered_local[:, :, c] = convolve(
                local_region[:, :, c],
                kernel,
                mode='reflect',  # Reflect padding at edges
                cval=0.0
            )
        filtered_local = np.clip(filtered_local, 0, 1)

        # 4. Alpha blend: region = orig*(1-opacity) + filtered*opacity
        img[y1:y2, x1:x2] = (1 - opacity) * local_region + opacity * filtered_local
        img = np.clip(img, 0, 1)

        # 5. Record filter info with short ID
        filter_info.append(f"{kernel_type}_k{kernel_size}")

    return img, filter_info

# -------------------------- Image noise augmentation with local filters --------------------------
def add_image_noise(img):
    img = np.copy(img)
    noise_types = []

    # Original noise augmentation (Gaussian, occlusion, blur, geometry)
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
        noise_types.append(f"occl{occlusion_count}")  # Short identifier

    if random() < BLUR_PROB:
        sigma = uniform(*BLUR_SIGMA_RANGE)
        img = gaussian_filter(img, sigma=sigma)
        noise_types.append(f"blur{sigma:.1f}")  # Short identifier

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
        noise_types.append(f"geo_rot{angle:.0f}")  # Short identifier,rotation angle only (core geometry)

    # Apply local CNN matrix filters
    img, filter_info = apply_local_cnn_filters(img)
    if filter_info:
        noise_types.extend(filter_info)

    return img, noise_types

# -------------------------- Core generation (formula-driven, frequency-centric) --------------------------
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

    # Generate unique filename (frequency-centric)
    clean_name = clean_filename(freq, n, m, img_idx, plate_color, sand_color, noise_types)
    filename = f"{SAVE_DIR}/{clean_name}.png"

    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    if (img_idx + 1) % 10 == 0:
        print(f"  Generated {img_idx + 1}/{IMG_PER_FREQ} images:{os.path.basename(filename)}")
    
    return os.path.basename(filename), noise_types  # Return filename and noise/filter info

# -------------------------- Batch generation with frequency uniqueness check --------------------------
if __name__ == "__main__":
    # Initialize random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"OK Created CNN-filter dataset directory: {os.path.abspath(SAVE_DIR)}")

    # Frequency uniqueness check (formula ensures uniqueness)
    freq_list = [info["freq"] for info in frequency_params]
    unique_freqs = set(freq_list)
    if len(freq_list) != len(unique_freqs):
        print(f"WARNING: WARNING: detected {len(freq_list)-len(unique_freqs)}  duplicate frequencies, auto-deduplicated!")
        unique_params = []
        seen_freqs = set()
        for info in frequency_params:
            if info["freq"] not in seen_freqs:
                seen_freqs.add(info["freq"])
                unique_params.append(info)
        frequency_params = unique_params

    # Mode mapping table for label generation
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # Print mapping table
    print("\nMode-index mapping table:")
    for modal, idx in modal2idx.items():
        print(f"  mode({modal[0]},{modal[1]}) -> index{idx}")
    print(f"One-hot label dimension:{len(modal_list)}")
    print("="*80)

    total_imgs = len(frequency_params) * IMG_PER_FREQ
    print(f"\nStarting CNN-filter full-gamut Chladni generation (total {total_imgs} images)...")
    print(f"Core configuration:")
    print(f"  - Color rules: RGB 0-1 full gamut | min color diff: {MIN_COLOR_DIFF}")
    print(f"  - Local matrix filters: CNN kernels (3/5/7) | random regions | 0-3 per image")
    print(f"  - Kernel types: edge, sharpen, blur, emboss, Sobel, Gaussian (8 types)")
    print(f"  - Image size:{IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Frequency: formula-driven (160mm center-fixed stainless plate)\n")

    # Initialize label dictionary
    label_dict = {}

    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        type_ = info["type"]
        level = info["level"]

        print(f"[{param_idx}/{len(frequency_params)}] Processing frequency: {freq}Hz (n={n}, m={m}, {type_})")

        for img_idx in range(IMG_PER_FREQ):
            img_filename, noise_types = generate_kladni(freq, n, m, img_idx)
            
            # Generate one-hot label
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # Save label metadata (filter/noise types)
            label_dict[img_filename] = {
                "one_hot": one_hot,
                "modal": (n, m),
                "modal_idx": modal_idx,
                "freq": freq,
                "level": level,
                "type": type_,
                "noise_filter_types": noise_types
            }

        print(f"  OK {freq}Hz generation complete ({IMG_PER_FREQ} images)\n")

    # Save label file
    label_save_path = LABELS_ROOT / "labels_matrix_filters.json"  # Save under data/labels
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # Completion message
    print(f"CNN-filter full-gamut dataset generation complete!")
    print(f"Save path:{os.path.abspath(SAVE_DIR)}")
    print(f"Dataset characteristics:")
    print(f"   - Total images:{total_imgs} images")
    print(f"   - Unique frequency count:{len(frequency_params)} case(s) (formula-driven)")
    print(f"   - Local filters: CNN-style kernels on random regions")
    print(f"   - Generalization: CNN training, kaleidoscope texture augmentation")
    print(f"   - Label file:{label_save_path} (includes one-hot, frequency, filter/noise types)")
    print(f"   - Filename format:{freq}Hz_n{n}m{m}_aug{idx}_color_noise_filter.png (frequency as primary label)")