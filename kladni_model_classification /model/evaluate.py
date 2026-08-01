import sys
import time  # Timing module
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from model.dataset import ChladniDataset
from model.network import BasicCNN

# ===== Configuration =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 15
BATCH_SIZE = 32

# ===== Data =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ChladniDataset(
    image_dir=DATA_ROOT / "images" / "test_synthetic",
    label_path=DATA_ROOT / "labels" / "test_synthetic.json",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== Load model =====
model = BasicCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.eval()

# ===== Pure model inference speed (excluding data loading) =====
print("Pure model inference speed (excluding data loading)...")
# 1. Build a single test image (224x224x3 input)
test_img = torch.randn(1, 3, 224, 224).to(DEVICE)

# 2. Warm up model (avoid slow first inference)
for _ in range(10):
    with torch.no_grad():
        _ = model(test_img)

# 3. Benchmark: 100 runs averaged to reduce error
start_pure = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(test_img)
end_pure = time.time()

# Compute pure model per-image inference speed
pure_infer_time = (end_pure - start_pure) / 100  # Per-image time (seconds)
pure_throughput = 1 / pure_infer_time            # Throughput (images/sec)
print(f"Pure model inference speed:{pure_infer_time*1000:.2f} ms/image | Throughput: {pure_throughput:.2f} images/sec")

# ===== Prediction with data loading (overall inference speed) =====
all_preds, all_labels = [], []
start_total = time.time()  # Record overall inference start time

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Overall inference speed (including data loading)
end_total = time.time()
total_time = end_total - start_total
total_samples = len(test_dataset)
avg_total_time = total_time / total_samples  # Average per-image time (including loading)
total_throughput = 1 / avg_total_time        # Overall throughput

# Convert to NumPy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ===== Compute core metrics =====
acc = np.mean(all_preds == all_labels)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')

# ===== Output results =====
print("\n" + "="*60)
print("BasicCNN test set evaluation results")
print("="*60)
# Inference speed output
print(f"Overall inference speed (with data loading):{avg_total_time*1000:.2f} ms/image | Throughput: {total_throughput:.2f} images/sec")
# Accuracy/F1-scoreOutput
print(f"Test Accuracy: {acc:.4f}")
print(f"Macro-F1 Score (macro average): {macro_f1:.4f}")
print(f"Micro-F1 Score (micro average): {micro_f1:.4f}")
# Classification report
print("\nClassification Report (per-class Precision/Recall/F1):")
target_names = [f"Mode_{i}" for i in range(NUM_CLASSES)]
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=target_names,
    digits=4
))

# Optional: uncomment to print confusion matrix
# print("\n🌀 Confusion Matrix:")
# cm = confusion_matrix(all_labels, all_preds)
# print(cm)