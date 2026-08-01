import numpy as np
import matplotlib.pyplot as plt
import os
import json
from random import uniform, choice, random, randint
import cv2
from scipy.ndimage import gaussian_filter
from pathlib import Path  # Path utilities for inlined config

# ===================== Original config fully inlined (no external import)=====================
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES = 15
# Generation config (aligned with original config)
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15


# Script directory: data/raw/ (noise.py)
RAW_DIR = Path(__file__).parent.resolve()
# Parent dir: project data/ root (consistent with clean_data.py)
DATA_ROOT = RAW_DIR.parent.resolve()

# Label path: data/generated/labels/ (shared with clean labels)
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# Noise image path: data/generated/images/noise/
SAVE_DIR = DATA_ROOT / "generated" / "images" / "noise"
os.makedirs(SAVE_DIR, exist_ok=True)
# -------------------------- Core physical parameters (160mm square plate)--------------------------
# Plate parameters (stainless steel)
PLATE_SIZE_ACTUAL = 0.16          # Actual plate edge length 160mm (m)
PLATE_THICKNESS = 0.0008         # Plate thickness 0.8mm (m)
E = 200e9                        # Stainless steel Young's modulus (Pa)
NU = 0.3                         # Poisson's ratio
RHO = 7850                       # Stainless steel density (kg/m^3)
D = E * PLATE_THICKNESS**3 / (12 * (1 - NU**2))  # Flexural rigidity (N*m)

# Virtual size mapping (visualization)
PLATE_SIZE_VIRTUAL = 2.0         # Virtual length 2.0 maps to 160mm
SCALE_RATIO = PLATE_SIZE_ACTUAL / PLATE_SIZE_VIRTUAL  # 1 virtual unit = 0.08m
CENTER_FIXED_RADIUS_ACTUAL = 0.003  # Actual center fixation radius 3mm
CENTER_FIXED_RADIUS = CENTER_FIXED_RADIUS_ACTUAL / SCALE_RATIO  # Virtual fixation radius

# -------------------------- Full-gamut + noise augmentation config --------------------------
IMG_PER_FREQ = 100  # 100 augmented images per frequency
MIN_COLOR_DIFF = 0.3  # Minimum plate-sand color difference (contrast)

# -------------------------- Noise augmentation params (rotation removed) --------------------------
# Sand distribution noise
SAND_DISTRIBUTION_NOISE = 0.02  # Sand position random offset amplitude
SAND_DENSITY_VARIATION = 0.2  # Sand density variation (±20%)
# Gaussian noise
GAUSSIAN_NOISE_SIGMA_RANGE = (0.005, 0.02)  # Noise intensity range
# Occlusion noise
OCCLUSION_PROB = 0.3  # Occlusion probability
OCCLUSION_COUNT_RANGE = (1, 3)  # Occlusion count
OCCLUSION_SIZE_RANGE = (5, 15)  # Occlusion size (pixels)
# Blur noise
BLUR_PROB = 0.25  # Blur probability
BLUR_SIGMA_RANGE = (0.3, 1.0)  # Blur sigma range
# Original geometry transforms (rotation removed)
GEOMETRY_PROB = 0.3  # Shear/scale probability
SCALE_RANGE = (0.95, 1.05)  # Original scale range
SHEAR_RANGE = (-0.03, 0.03)  # Shear range

# -------------------------- Extended geometry transform params (rotation removed)--------------------------
TRANSLATE_RANGE = (-10, 10)  # Translation range (pixels, independent x/y)
EXTENDED_SCALE_RANGE = (0.8, 1.2)  # Extended scale range
FLIP_PROB = 0.5  # Horizontal/vertical flip probability
# Fix: numeric flip codes (OpenCV standard)
FLIP_TYPES = [None, 1, 0]  # Flip type (none/horizontal/vertical)

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

# Build frequency list driven by modal n/m with label/type fields
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
    # Generate label (mode + frequency)
    label = f"n{n}m{m}_{freq}Hz_level {level}"
    frequency_params.append({
        "freq": freq,
        "n": n,
        "m": m,
        "lambda": lam,
        "level": level,
        "label": label,
        "type": type_
    })

# -------------------------- Utility functions (adapted for new params) --------------------------
def clean_filename(label):
    """Sanitize illegal filename characters"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in illegal_chars:
        label = label.replace(char, '_')
    return label

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

# -------------------------- Sand distribution noise function (original logic) --------------------------
def add_sand_distribution_noise(sand_x, sand_y, sand_sizes):
    """Add sand position offset and density variation"""
    # 1. Random sand position offset
    offset_x = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_x))
    offset_y = np.random.normal(0, SAND_DISTRIBUTION_NOISE, len(sand_y))
    sand_x += offset_x
    sand_y += offset_y

    # 2. Random sand density variation
    current_count = len(sand_x)
    variation = int(current_count * uniform(-SAND_DENSITY_VARIATION, SAND_DENSITY_VARIATION))
    new_count = current_count + variation
    new_count = max(10000, min(14000, new_count))  # Clamp count range

    if new_count > current_count:
        # Supplement sand grains
        supplement_x = np.random.uniform(-1, 1, new_count - current_count)
        supplement_y = np.random.uniform(-1, 1, new_count - current_count)
        supplement_sizes = np.random.uniform(*GRAIN_SIZE_RANGE, new_count - current_count)
        sand_x = np.concatenate([sand_x, supplement_x])
        sand_y = np.concatenate([sand_y, supplement_y])
        sand_sizes = np.concatenate([sand_sizes, supplement_sizes])
    elif new_count < current_count:
        # Reduce sand grains
        idx = np.random.choice(current_count, new_count, replace=False)
        sand_x, sand_y, sand_sizes = sand_x[idx], sand_y[idx], sand_sizes[idx]

    return sand_x, sand_y, sand_sizes

# -------------------------- Extended geometry transforms (rotation removed)--------------------------
def apply_extended_geometry_transforms(img):
    """Apply random translate, scale, flip (rotation removed)"""
    img = np.copy(img)
    transform_info = []
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # 1. Random translation
    tx = randint(*TRANSLATE_RANGE)
    ty = randint(*TRANSLATE_RANGE)
    translate_mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(
        img, translate_mat, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0)
    )
    transform_info.append(f"translate(tx={tx},ty={ty})")

    # 2. Random scale
    scale = uniform(*EXTENDED_SCALE_RANGE)
    scale_mat = cv2.getRotationMatrix2D(center, 0, scale)  # Rotation angle fixed at 0
    img = cv2.warpAffine(
        img, scale_mat, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(1.0, 1.0, 1.0)
    )
    transform_info.append(f"scale({scale:.2f})")

    # 3. Random flip (fixed logic)
    if random() < FLIP_PROB:
        flip_type = choice(FLIP_TYPES[1:])  # Exclude None
        img = cv2.flip(img, flip_type)
        flip_str = "flip_horizontal" if flip_type == 1 else "flip_vertical"
        transform_info.append(flip_str)

    return img, transform_info

# -------------------------- Image noise augmentation (rotation removed) --------------------------
def add_image_noise(img):
    """Multi-dimensional noise: Gaussian, occlusion, blur, geometry (rotation removed)"""
    img = np.copy(img)
    noise_types = []

    # 1. Gaussian noise
    sigma = uniform(*GAUSSIAN_NOISE_SIGMA_RANGE)
    gaussian_noise = np.random.normal(0, sigma, img.shape)
    img = np.clip(img + gaussian_noise, 0, 1)
    noise_types.append("gaussian_noise")

    # 2. Occlusion noise
    if random() < OCCLUSION_PROB:
        occlusion_count = randint(*OCCLUSION_COUNT_RANGE)
        for _ in range(occlusion_count):
            x = randint(0, IMAGE_SIZE)
            y = randint(0, IMAGE_SIZE)
            size = randint(*OCCLUSION_SIZE_RANGE)
            # Random occlusion color (dark/bright spots)
            occlusion_color = uniform(0, 0.3) if random() < 0.5 else uniform(0.7, 1.0)
            # Clamp occlusion to image bounds
            x = max(size, min(IMAGE_SIZE - size, x))
            y = max(size, min(IMAGE_SIZE - size, y))
            cv2.circle(
                img, (x, y), size,
                (occlusion_color, occlusion_color, occlusion_color),
                thickness=-1
            )
        noise_types.append(f"occlusion_{occlusion_count}")

    # 3. Blur noise
    if random() < BLUR_PROB:
        sigma = uniform(*BLUR_SIGMA_RANGE)
        img = gaussian_filter(img, sigma=sigma)
        noise_types.append(f"blur_{sigma:.2f}")

    # 4. Geometry transforms (shear + scale, rotation removed)
    if random() < GEOMETRY_PROB:
        rows, cols = img.shape[:2]
        scale = uniform(*SCALE_RANGE)
        shear_x = uniform(*SHEAR_RANGE)
        shear_y = uniform(*SHEAR_RANGE)

        # Shear transform
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

    # 5. Extended geometry (rotation removed)
    img, transform_info = apply_extended_geometry_transforms(img)
    noise_types.extend(transform_info)

    return img, noise_types

# -------------------------- Core generation (noise + full gamut + formula-driven) --------------------------
def generate_kladni(freq, n, m, label, img_idx, type_):
    """Generate a single noisy Chladni pattern"""
    clean_label = clean_filename(label)
    img_size = IMAGE_SIZE

    # 1. Generate full-gamut colors
    plate_color, sand_color = get_full_color_pair()

    # 2. Generate metal background
    bg, plate_mask = generate_metal_bg(img_size, plate_color)

    # 3. Generate sand distribution and apply noise
    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)
    sand_x, sand_y, sand_sizes = distribute_sand(X_grid, Y_grid, plate_mask, n, m)
    sand_x, sand_y, sand_sizes = add_sand_distribution_noise(sand_x, sand_y, sand_sizes)

    # 4. Draw base figure
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

    # 5. Save temporary image
    os.makedirs(SAVE_DIR, exist_ok=True)
    temp_file = os.path.join(SAVE_DIR, "temp_kladni.png")  
    plt.savefig(temp_file, bbox_inches="tight", pad_inches=0, dpi=100, facecolor="white")
    plt.close()

    # 6. Load and apply noise
    img = plt.imread(temp_file)[:, :, :3]  # Strip alpha channel
    img, noise_types = add_image_noise(img)
    os.remove(temp_file)  # Remove temp file

    # 7. Build final filename (freq, index, noise, color info)
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    noise_str = "_".join([n.replace(" ", "_") for n in noise_types])
    filename = f"{SAVE_DIR}/{freq}Hz_{clean_label}_{img_idx}_{plate_rgb}_{sand_rgb}_{noise_str}.png"

    # 8. Save final image as uint8
    img_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    # Progress logging
    if (img_idx + 1) % 10 == 0:
        print(f"    Generated {img_idx + 1}/{IMG_PER_FREQ} images:{os.path.basename(filename)}")
    
    return os.path.basename(filename), noise_types  # Return filename and noise info for labels

# -------------------------- Batch generation main logic with one-hot labels--------------------------
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)  # Same seed as training/generation scripts
    # Create output directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"OK Created dataset directory: {os.path.abspath(SAVE_DIR)}")

    # Mode mapping table (one-hot label core)
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # Print mapping table
    print("\nMode-index mapping table:")
    for modal, idx in modal2idx.items():
        print(f"  mode({modal[0]},{modal[1]}) -> index{idx}")
    print(f"One-hot label dimension:{len(modal_list)}")
    if NUM_CLASSES != len(modal_list):
        print(f"WARNING: WARNING: config.NUM_CLASSES={NUM_CLASSES}  vs actual mode count={len(modal_list)}mismatch!")
    print("="*80)

    # Print configuration (rotation removed)
    total_freq = len(frequency_params)
    total_imgs = total_freq * IMG_PER_FREQ
    print(f"\nStarting noisy Chladni pattern dataset generation:")
    print(f"   - Total frequencies:{total_freq} groups (formula-driven)")
    print(f"   - Images per group:{IMG_PER_FREQ} images")
    print(f"   - Total images:{total_imgs} images")
    print(f"   - Image size:{IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   - Noise types: sand distribution, Gaussian, occlusion, blur, geometry (shear/scale/translate/flip)\n")

    # Initialize label dictionary
    label_dict = {}

    # Batch generation
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        label = info["label"]
        type_ = info["type"]
        level = info["level"]

        print(f"[{param_idx}/{total_freq}] Processing {type_} mode:{freq}Hz (n={n}, m={m}, level {level})")
        
        # 100 augmented images per frequency
        for img_idx in range(IMG_PER_FREQ):
            img_filename, noise_types = generate_kladni(freq, n, m, label, img_idx, type_)
            
            # Generate one-hot label
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # Save label metadata to dict (source for labels.json)
            label_dict[img_filename] = {
                "one_hot": one_hot, # OK One-hot label for model training (core)
                "modal": (n, m), # Mode index e.g. (1,2)/(1,3)
                "modal_idx": modal_idx, # Numeric mode index (0~13)
                "freq": freq, # Frequency value
                "level": level, # frequencylevel
                "type": type_,  # Mode type
                "noise_types": noise_types # Noise augmentation types
            }

        print(f"    OK {freq}Hz mode generation complete\n")

    # Save label file
    label_save_path = LABELS_ROOT / "noise_label.json"
    # ======================================================
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # Completion message (rotation removed)
    print(f"Dataset generation complete!")
    print(f"Save path:{os.path.abspath(SAVE_DIR)}")
    print(f"Dataset characteristics:")
    print(f"   - Colors: full-gamut RGB, min diff{MIN_COLOR_DIFF}(ensure contrast)")
    print(f"   - Noise: 7 augmentation types for real capture scenarios")
    print(f"   - Frequency: formula-driven (160mm center-fixed stainless plate)")
    print(f"   - mode:{len(modal_list)} valid asymmetric modes(n >= 1, m >= 1, n != m)")
    print(f"   - Labels: {label_save_path} (includes one-hot and noise type metadata)")
    print(f"   - Generalization: translate, scale, flip (rotation removed)")