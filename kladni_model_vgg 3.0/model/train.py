# ======================
# Standard library imports
# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress irrelevant warnings to reduce overhead
import time
from datetime import timedelta

# ======================
# Local imports
# ======================
from dataset import ChladniDataset
from network import VGG16Classifier  

# ======================
# Plotting utilities
# ======================
def plot_training_curves(train_losses, val_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, color=color, marker='s', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress: Loss and Accuracy')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # ===== Configuration =====
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "processed"
    MODEL_SAVE_PATH = PROJECT_ROOT / "model" / "best_model.pth"
    PLOT_SAVE_PATH = PROJECT_ROOT / "plots" / "training_curve.png"
    (PROJECT_ROOT / "plots").mkdir(exist_ok=True)

    # Set to 8 for Apple M4 (reduce to 4 if thermals are high)
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-4  # OK Lower learning rate for pretrained model
    
    # Prefer MPS (Apple Silicon) to reduce CPU load and thermals
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        # Remove unsupported memory limit API on older PyTorch
        # torch.backends.mps.set_per_process_memory_fraction(0.8)  # Removed
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Using device: {DEVICE}")

    # ===== Data preprocessing (memory optimized)=====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ===== Dataset =====
    train_dataset = ChladniDataset(
        image_dir=DATA_ROOT / "images" / "train",
        label_path=DATA_ROOT / "labels" / "train.json",
        transform=transform
    )
    val_dataset = ChladniDataset(
        image_dir=DATA_ROOT / "images" / "val",
        label_path=DATA_ROOT / "labels" / "val.json",
        transform=transform
    )
    
    # pin_memory=False (not needed on M4, avoids extra copies)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        drop_last=False  # Avoid memory spikes from final partial batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # ===== Model and optimizer (memory optimized)=====
    model = VGG16Classifier(num_classes=15, pretrained=True).to(DEVICE)
    
    # Gradient checkpointing to reduce VGG16 memory (~30%)
    # Comment out if VGG16Classifier is not Sequential
    try:
        if DEVICE in ["mps", "cuda"]:
            model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
    except:
        print("WARNING: Gradient checkpointing disabled (non-Sequential model); training unaffected")
    
    print("Model loaded successfully.")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")

    criterion = nn.CrossEntropyLoss()
    # Weight decay to stabilize learning rate
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # ===== Manual dataset smoke test =====
    print("Testing dataset...")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    try:
        img, lbl = train_dataset[0]
        print(f"OK Sample loaded: image shape={img.shape}, label={lbl}")
    except Exception as e:
        print(f"ERROR: Dataset error: {e}")
        raise

    # ===== Training records =====
    train_losses = []
    val_accuracies = []
    epoch_times = []  # Record per-epoch duration
    total_train_start = time.time()  # Overall training start time
    
    # ===== Training loop (with memory cleanup)=====
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        epoch_start = time.time()  # Start timing current epoch
        # Training
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} - Training =====")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Release intermediate tensors after forward pass
            with torch.autograd.detect_anomaly(False):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        # Compute average loss
        avg_train_loss = train_loss / total_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():  # Disable gradients to reduce memory usage
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best Val Acc: {best_val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save from CPU to avoid MPS memory pinning
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")

        # Compute and print current epoch duration
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch+1} Duration: {timedelta(seconds=int(epoch_duration))}")
        # Clear cache after plotting
        plot_training_curves(train_losses, val_accuracies, PLOT_SAVE_PATH)
        if DEVICE == "mps":
            torch.mps.empty_cache()

    print(f"\nOK Training finished!")
    # Compute total training time和平均EpochDuration
    total_train_end = time.time()
    total_duration = total_train_end - total_train_start
    print(f"Total training time: {timedelta(seconds=int(total_duration))}")
    print(f"Average per-epoch duration: {timedelta(seconds=int(sum(epoch_times)/len(epoch_times)))}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Training curve saved to: {PLOT_SAVE_PATH}")

if __name__ == '__main__':
    main()