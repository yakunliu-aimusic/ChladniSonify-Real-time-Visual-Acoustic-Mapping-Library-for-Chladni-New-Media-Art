import numpy as np
import matplotlib.pyplot as plt
import os
import json  # For saving label files
from scipy.ndimage import gaussian_filter
from pathlib import Path  # Path utilities

# ==============================
# Script directory: data/raw/ (clean_data.py)
RAW_DIR = Path(__file__).parent.resolve()
# Parent dir: project data/ root (all paths relative to this)
DATA_ROOT = RAW_DIR.parent.resolve()

# Label save path: data/generated/labels/
LABELS_ROOT = DATA_ROOT / "generated" / "labels"
os.makedirs(LABELS_ROOT, exist_ok=True)  # Auto-create directory

# Legacy unused path vars commented out
# DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"

# Experiment hyperparameters
IMAGE_SIZE = 224          # Image size
RANDOM_SEED = 42          # Random seed (reproducibility)
NUM_CLASSES = 16          # Number of dataset classes

# -------------------------z- Core physical parameters (160mm square plate)--------------------------
# Plate parameters (stainless steel; adjust to match experimental plate)
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

# Generation configuration
# Image save path: data/generated/images/clean/
SAVE_DIR = DATA_ROOT / "generated" / "images" / "clean"
os.makedirs(SAVE_DIR, exist_ok=True)  # Auto-create nested directories including clean/
SAND_GRAIN_COUNT = 12000         # Number of sand grains
GRAIN_SIZE_RANGE = (0.4, 1.0)
METAL_COLOR = (0.55, 0.6, 0.65)  # Dark silver-gray
SAND_COLOR = (0.95, 0.95, 0.95)  # Light gray sand grains
DAMPING = 0.06                   # Damping coefficient
METAL_REFLECT_STRENGTH = 0.15    # Metal reflection strength
IMAGES_PER_MODAL = 100           # Augmented images per modal pattern

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

# Build frequency list driven by modal n/m (not manual assignment)
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

# -------------------------- Utility functions --------------------------
def clean_filename(freq, n, m, aug_idx=None):
    """Generate valid filename (with augmentation index)"""
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
    """Generate metal background"""
    bg = np.ones((size, size, 3)) * METAL_COLOR
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    # Metal reflection
    reflection = np.exp(-(X_grid ** 2 + Y_grid ** 2) / 0.8) * METAL_REFLECT_STRENGTH
    bg[:, :, 0] += reflection
    bg[:, :, 1] += reflection
    bg[:, :, 2] += reflection

    # Center fixation region
    center_mask = (X_grid ** 2 + Y_grid ** 2) <= CENTER_FIXED_RADIUS ** 2
    bg[center_mask] = [0.2, 0.2, 0.2]

    # Metal texture noise
    bg += np.random.normal(0, 0.006, bg.shape)
    bg = np.clip(bg, 0, 1)

    # Plate mask
    plate_mask = (np.abs(X_grid) <= 1) & (np.abs(Y_grid) <= 1)
    return bg, plate_mask

def calculate_160mm_vibration(X, Y, n, m):
    """Compute vibration amplitude (2D skew-symmetric modes, no n/m=0 branch)"""
    L = 1.0
    # ========== Distance to center + attenuation coefficient ==========
    r = np.sqrt(X ** 2 + Y ** 2)  # Distance from each point to center
    alpha = 1.0  # Center attenuation coefficient (~1.0 matches experiments)
    # ==========================================================
    
    # Chladni 2D skew-symmetric formula with center attenuation (core modification)
    vibration = np.sin(n * np.pi * X / L) * np.sin(m * np.pi * Y / L) - \
                np.sin(m * np.pi * X / L) * np.sin(n * np.pi * Y / L)
    vibration = vibration * np.exp(-alpha * r)  # Apply center attenuation
    
    # Zero vibration in center fixation region (legacy safeguard)
    center_mask = (X ** 2 + Y ** 2) <= CENTER_FIXED_RADIUS ** 2
    vibration[center_mask] = 0
    
    # Damping and normalization
    damping = np.exp(-DAMPING * (np.abs(X) + np.abs(Y)))
    vibration = vibration * damping
    max_vib = np.max(np.abs(vibration))
    if max_vib > 0:
        vibration = vibration / max_vib
    return vibration

def distribute_160mm_sand(X, Y, plate_mask, n, m):
    """Distribute sand grains (divide-by-zero fix + empty fallback)"""
    X_plate = X[plate_mask]
    Y_plate = Y[plate_mask]
    vibration = calculate_160mm_vibration(X_plate, Y_plate, n, m)
    vib_abs = np.abs(vibration)

    # 1. Dynamic threshold (fallback 0.1 when no valid values)
    if len(vib_abs) == 0 or np.max(vib_abs) == 0:
        sand_threshold = 0.1
        sand_mask = np.ones_like(X_plate, dtype=bool)
    else:
        sand_threshold = np.percentile(vib_abs, 15)
        sand_mask = vib_abs < sand_threshold

    # 2. Exclude center fixation region
    center_mask_plate = (X_plate ** 2 + Y_plate ** 2) <= CENTER_FIXED_RADIUS ** 2
    sand_mask = sand_mask & (~center_mask_plate)

    sand_x = X_plate[sand_mask]
    sand_y = Y_plate[sand_mask]
    sand_vib = vib_abs[sand_mask]

    # 3. Fallback: random base sand grains when empty
    if len(sand_x) == 0:
        base_count = min(SAND_GRAIN_COUNT, len(X_plate))
        idx = np.random.choice(len(X_plate), base_count, replace=False)
        sand_x = X_plate[idx]
        sand_y = Y_plate[idx]
        sand_vib = np.zeros(base_count)

    # 4. Ensure sand grain count (divide-by-zero fix)
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

    # 5. Dynamic grain size + small random offset (break grid artifacts)
    sand_sizes = np.interp(sand_vib, [0, sand_threshold], [0.3, 1.2])
    sand_sizes = np.clip(sand_sizes, *GRAIN_SIZE_RANGE)
    sand_x += np.random.normal(0, 0.01, sand_x.shape)  # Random offset
    sand_y += np.random.normal(0, 0.01, sand_y.shape)

    # 6. Boundary check
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

# -------------------------- Core generation function --------------------------
def generate_160mm_kladni(freq, n, m, aug_idx):
    """Generation function with augmentation index"""
    clean_name = clean_filename(freq, n, m, aug_idx)
    img_size = IMAGE_SIZE  # Use inline IMAGE_SIZE
    bg, plate_mask = generate_160mm_metal_bg(img_size)

    X = np.linspace(-1, 1, img_size)
    Y = np.linspace(-1, 1, img_size)
    X_grid, Y_grid = np.meshgrid(X, Y)

    sand_x, sand_y, sand_sizes = distribute_160mm_sand(X_grid, Y_grid, plate_mask, n, m)

    # Create canvas
    plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.axis("off")

    # Draw blurred background
    bg_blurred = gaussian_filter(bg, sigma=0.5)
    ax.imshow(bg_blurred, extent=[-1, 1, -1, 1], origin="lower")
    
    # Draw sand grains
    ax.scatter(
        sand_x, sand_y,
        s=sand_sizes,
        c=[SAND_COLOR],
        alpha=0.95,
        edgecolors="none",
        marker="o",
        rasterized=True
    )

    # Save image
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
    print(f"Generated:{filename} (n={n},m={m},frequency={freq}Hz,aug{aug_idx})")

# -------------------------- Batch generation (single main block)--------------------------
if __name__ == "__main__":
    # Set random seed (inline RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    # ========== Mode mapping table (one-hot label core) ==========
    # 1. Extract unique modes sorted for fixed indices
    modal_list = sorted(LAMBDA_MN.keys(), key=lambda x: (x[0], x[1]))  # Sort by n, m ascending
    # 2. Mode-to-index mapping (one-hot dimension = mode count)
    modal2idx = {modal: idx for idx, modal in enumerate(modal_list)}
    # 3. Index-to-mode reverse mapping (for post-training validation)
    idx2modal = {idx: modal for modal, idx in modal2idx.items()}
    # Print mapping table (optional validation)
    print("Mode-index mapping table:")
    for modal, idx in modal2idx.items():
        print(f"  mode({modal[0]},{modal[1]}) -> index{idx}")
    print(f"One-hot label dimension:{len(modal_list)}")
    print("="*80)

    # ========== Initialize recorder and label dictionary ==========
    generated_freqs = set()
    generated_mn = set()
    success_count = 0
    failed_cases = []
    label_dict = {}  # key: image filename, value: one-hot label list

    # Print configuration
    print("Chladni pattern generation (formula-driven) - 160mm center-fixed, four-edge-free square plate")
    print(f"Plate: edge length={PLATE_SIZE_ACTUAL*1000}mm | thickness={PLATE_THICKNESS*1000}mm | material=stainless steel")
    print(f"Valid mode count:{len(frequency_params)} | Images per mode:{IMAGES_PER_MODAL} images | Output path:{os.path.abspath(SAVE_DIR)}")
    print(f"Expected total output:{len(frequency_params) * IMAGES_PER_MODAL} images")
    print("="*80)

    # ========== Batch generation loop (nested augmentation) ==========
    for param_idx, info in enumerate(frequency_params, 1):
        freq = info["freq"]
        n = info["n"]
        m = info["m"]
        mn_key = (n, m)

        # Skip duplicate modes (redundant check)
        if mn_key in generated_mn:
            print(f"\n[{param_idx}/{len(frequency_params)}] Skip duplicate mode:n={n},m={m}(frequency={freq} Hz)")
            continue
        
        # Nested loop: IMAGES_PER_MODAL augmented images per mode
        for aug_idx in range(IMAGES_PER_MODAL):
            try:
                print(f"\n[{param_idx}/{len(frequency_params)}][aug{aug_idx+1}/{IMAGES_PER_MODAL}] Generating level {info['level']} mode:n={n},m={m} -> computed frequency={freq}Hz...")
                generate_160mm_kladni(freq, n, m, aug_idx)

                # Generate one-hot label (same label per augmented image)
                modal_idx = modal2idx[(n, m)]
                one_hot = [0] * len(modal_list)
                one_hot[modal_idx] = 1
                
                # Associate filename with label
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
                error_info = f"n={n},m={m}_aug{aug_idx}(frequency={freq} Hz): {str(e)}"
                print(f"\n[{param_idx}/{len(frequency_params)}][aug{aug_idx+1}/{IMAGES_PER_MODAL}] Generation failed:{error_info}")
                failed_cases.append(error_info)
        
        # Mark mode as generated
        generated_freqs.add(freq)
        generated_mn.add(mn_key)

    # ========== Save label file (inline LABELS_PATH) ==========
    label_save_path = LABELS_ROOT / "clean_labels.json"
    with open(label_save_path, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, indent=2, ensure_ascii=False)
    print(f"\nLabel file saved to:{label_save_path}")

    # ========== Generation result validation ==========
    print("\n" + "="*80)
    print(f"OK Generation complete! Successfully generated {success_count} images (target {len(frequency_params) * IMAGES_PER_MODAL} images)")
    if failed_cases:
        print(f"ERROR: Failed cases({len(failed_cases)}):")
        for err in failed_cases:
            print(f"  - {err}")
    print(f"Data statistics:")
    print(f"  - Frequency range:{min(generated_freqs)}Hz ~ {max(generated_freqs)}Hz")
    print(f"  - Mode count: {len(generated_mn)} types (all satisfy n >= 1, m >= 1, n != m)")
    print(f"  - Output path:{os.path.abspath(SAVE_DIR)}")
    print(f"  - Label file:{label_save_path} (contains {len(label_dict)} labels)")
    print("="*80)