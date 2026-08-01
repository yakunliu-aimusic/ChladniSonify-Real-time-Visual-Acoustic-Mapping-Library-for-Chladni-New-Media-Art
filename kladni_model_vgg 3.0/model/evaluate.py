import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import time
import warnings  # For suppressing warnings
from model.dataset import ChladniDataset
from model.network import VGG16Classifier

# ===== Suppress PyTorch pretrained deprecation warnings =====
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ===== Configuration =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 15
BATCH_SIZE = 16  # Smaller batch size for faster CPU inference
NUM_WORKERS = 0

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
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False
)

# ===== Load model =====
# Use pretrained=False for custom VGG16Classifier
model = VGG16Classifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    print(f"OK VGG16 model weights loaded successfully:{MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Weight file not found! Check path:{MODEL_PATH}")
    raise
except RuntimeError as e:
    print(f"ERROR: Weight file incompatible with model:{e}")
    raise

model.eval()
print(f"Device:{DEVICE} | Test set size:{len(test_dataset)}")
print(f"Inference config: batch size={BATCH_SIZE} | Starting prediction...")

# ===== Prediction with progress and timing=====
all_preds, all_labels = [], []
start_time = time.time()

with torch.no_grad():
    # CPU multi-threading optimization for VGG16 inference
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        # Live progress and ETA
        elapsed_time = time.time() - start_time
        progress = (batch_idx + 1) / len(test_loader)
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        
        print(f"\rPrediction progress:{batch_idx + 1}/{len(test_loader)} batches "
              f"({progress*100:.1f}%) | Elapsed:{elapsed_time:.1f}s "
              f"| ETA:{eta:.1f}s", end="")
        
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Process results
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
total_time = time.time() - start_time
print(f"\nOK Prediction complete! Total time:{total_time:.1f}s | Average per sample:{total_time/len(test_dataset):.3f}s")

# ===== Compute metrics =====
acc = np.mean(all_preds == all_labels)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')

# ===== Output results =====
print("\n" + "="*60)
print("VGG16Classifier test set evaluation results")
print("="*60)
print(f"Test Accuracy: {acc:.4f}")
print(f"Macro-F1 Score (macro average): {macro_f1:.4f}")
print(f"Micro-F1 Score (micro average): {micro_f1:.4f}")
print("\nClassification Report (per-class Precision/Recall/F1):")
target_names = [f"Mode_{i}" for i in range(NUM_CLASSES)]
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=target_names,
    digits=4
))