import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random  
from random import uniform
from scipy.ndimage import gaussian_filter
from pathlib import Path  # Path utilities for inlined config

# ===================== Path config: data/raw/ -> data/generated/=====================
# Experiment hyperparameters (core params unchanged)
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_CLASSES = 15
# Generation config (params unchanged)
SAND_GRAIN_COUNT = 12000
GRAIN_SIZE_RANGE = (0.4, 1.0)
DAMPING = 0.06
METAL_REFLECT_STRENGTH = 0.15

# Script directory: data/raw/ (random_color.py)
RAW_DIR = Path(__file__).parent.resolve()
# Parent dir: project data/ root (project-wide convention)
DATA_ROOT = RAW_DIR.parent.resolve()

# Label path: data/generated/labels/ (shared with clean/noise)
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)
# Random color image path: data/generated/images/random_color/
SAVE_DIR = DATA_ROOT / "generated" / "images" / "random_color"
os.makedirs(SAVE_DIR, exist_ok=True)

# Legacy unused path vars commented out
# DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
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

# -------------------------- Full-gamut augmentation config --------------------------

IMG_PER_FREQ = 100               # 100 augmented images per frequency
MIN_COLOR_DIFF = 0.3             # Minimum color difference (contrast)

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

# -------------------------- Full-gamut color + noise augmentation utilities --------------------------
def clean_filename(freq, img_idx, plate_color, sand_color, n, m):
    """Generate valid filename (mode + color info)"""
    illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    # Generate color ID from RGB integers
    plate_rgb = f"P{int(plate_color[0]*255)}_{int(plate_color[1]*255)}_{int(plate_color[2]*255)}"
    sand_rgb = f"S{int(sand_color[0]*255)}_{int(sand_color[1]*255)}_{int(sand_color[2]*255)}"
    # Base filename: freq + mode + index + color
    filename = f"{freq}Hz_n{n}m{m}_aug_{img_idx}_{plate_rgb}_{sand_rgb}"
    # Replace illegal characters
    for char in illegal_chars:
        filename = filename.replace(char, '_')
    return filename

def get_full_color_pair():
    """Random full-gamut plate and sand colors (diff >= MIN_COLOR_DIFF)"""
    # 1. Random plate color (RGB 0-1)
    plate_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))

    # 2. Generate sand color until contrast threshold met
    while True:
        sand_color = (uniform(0, 1), uniform(0, 1), uniform(0, 1))
        # Average RGB channel difference
        color_diff = (
            abs(plate_color[0] - sand_color[0]) +
            abs(plate_color[1] - sand_color[1]) +
            abs(plate_color[2] - sand_color[2])
        ) / 3
        # Return when color difference meets threshold
        if color_diff >= MIN_COLOR_DIFF:
            return plate_color, sand_color

def generate_metal_bg(size, plate_color):
    """Generate metal background (full gamut + noise augmentation)"""
    bg = np.ones((size, size, 3)) * plate_color
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # Metal reflection effect(适配任意板色)
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # Center fixation region
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = (0.1, 0.1, 0.1)  # Fixed dark gray

    # Metal texture noise (realism, full gamut)
    bg += np.random.normal(0, 0.008, bg.shape)  # Random noise
    bg = np.clip(bg, 0, 1)  # Clamp RGB to [0, 1]

    # Plate mask
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_vibration(X, Y, n, m):
    """Compute vibration amplitude (formula + center attenuation)"""
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
    
    # Damping and normalization
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_sand(X, Y, plate_mask, n, m):
    """Distribute sand grains (divide-by-zero fix, fallback, random offset)"""
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

    # Dynamic grain size + small random offset (extra randomness)
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)
    sand_x += np.random.normal(0, 0.01, sand_x.shape)  # Random offset noise
    sand_y += np.random.normal(0, 0.01, sand_y.shape)

    # Boundary check
    plate_bounds_mask = (np.abs(sand_x) <= 1) & (np.abs(sand_y) <= 1)
    sand_x = sand_x[plate_bounds_mask]
    sand_y = sand_y[plate_bounds_mask]
    sand_sizes = sand_sizes[plate_bounds_mask]

    # Final fallback
    if len(sand_x) == 0:
        sand_x = np.random.uniform(-1, 1, SAND_GRAIN_COUNT)
        sand_y = np.random.uniform(-1, 1, SAND_GRAIN_COUNT)
        sand_sizes = np.random.uniform(*GRAIN_SIZE_RANGE, SAND_GRAIN_COUNT)

    return sand_x, sand_y, sand_sizes

# -------------------------- Core generation (full gamut + augmentation) --------------------------
def generate_kladni(freq, n, m, img_idx):
    # Full-gamut color pair (plate != sand, high contrast)
    plate_color, sand_color = get_full_color_pair()
    # Generate clean filename
    clean_name = clean_filename(freq, img_idx, plate_color, sand_color, n, m)
    img_size = IMAGE_SIZE

    # Generate background and mask with noise
    bg, plate_mask = generate_metal_bg(img_size, plate_color)

    # Build grid and sand distribution
    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)
    sand_x, sand_y, sand_sizes = distribute_sand(X_grid, Y_grid, plate_mask, n, m)

    # Draw figure
    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")

    # Draw blurred background (enhanced texture)
    bg_blurred = gaussian_filter(bg, sigma=0.5)
    ax.imshow(bg_blurred, extent=[-1, 1, -1, 1], origin="lower")
    
    # Draw sand grains (full gamut colors + random offset)
    ax.scatter(
        sand_x, sand_y,
        s=sand_sizes,
        c=[sand_color],
        alpha=0.95,
        edgecolors="none",
        marker="o",
        rasterized=True
    )

    # Save image
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

    # Progress logging every 10 images
    if (img_idx + 1) % 10 == 0:
        print(f"  Generated {img_idx + 1}/{IMG_PER_FREQ} images:{os.path.basename(filename)}")
    
    return clean_name + ".png"  # Return filename for label generation

# -------------------------- Batch generation + one-hot labels --------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Initialize random seed (reproducibility)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Create output directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created full-gamut dataset directory: {os.path.abspath(SAVE_DIR)}")

    # Mode mapping table (one-hot label core)
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    
    # Print mapping table
    print("\nMode-index mapping table:")
    for modal, idx in modal2idx.items():
        print(f"  mode({modal[0]},{modal[1]}) -> index{idx}")
    print(f"One-hot label dimension:{len(modal_list)}")
    print("="*80)

    # Validate frequency uniqueness
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

    total_imgs = len(frequency_params) * IMG_PER_FREQ
    print(f"\nStarting full-gamut Chladni augmentation (total {total_imgs} images)...")
    print(f"Color rules: RGB 0-1 full gamut random | Minimum plate-sand color difference:{MIN_COLOR_DIFF}\n")

    # Initialize label dictionary
    label_dict = {}

    # Iterate all frequencies for augmentation
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        level = info["level"]

        print(f"[{param_idx}/{len(frequency_params)}] Processing frequency: {freq}Hz (n={n}, m={m}, level {level})")

        # 100 augmented images per frequency
        for img_idx in range(IMG_PER_FREQ):
            img_filename = generate_kladni(freq, n, m, img_idx)
            
            # Generate one-hot label
            modal_idx = modal2idx[(n, m)]
            one_hot = [0] * len(modal_list)
            one_hot[modal_idx] = 1
            
            # Save label metadata
            label_dict[img_filename] = {
                "one_hot": one_hot,
                "modal": (n, m),
                "modal_idx": modal_idx,
                "freq": freq,
                "level": level
            }

        print(f"  OK {freq}Hz augmented images complete ({IMG_PER_FREQ} images)\n")

    # Save label file
    label_save_path = LABELS_ROOT / "random_label.json"  # Save under data/labels
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    # Generation statistics
    print(f"全色域augDataset generation complete!")
    print(f"Save path:{os.path.abspath(SAVE_DIR)}")
    print(f"Key parameters:")
    print(f"   - Total images:{total_imgs} images")
    print(f"   - Unique frequency count:{len(frequency_params)} case(s)")
    print(f"   - Unique mode count:{len(modal_list)}  types")
    print(f"   - Color range: RGB 0.0-1.0 (full gamut)")
    print(f"   - Color difference threshold: >= {MIN_COLOR_DIFF}(ensure contrast)")
    print(f"   - Image size:{IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   - Augmented images per frequency:{IMG_PER_FREQ} images")
    print(f"   - Label file:{label_save_path} (includes one-hot labels)")
    print(f"   - Frequency range:{min(freq_list)}Hz ~ {max(freq_list)}Hz")
    print("="*80)