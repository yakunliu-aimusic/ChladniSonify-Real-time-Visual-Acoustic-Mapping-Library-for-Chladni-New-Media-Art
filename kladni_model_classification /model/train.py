# ======================
# 基础库导入
# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt  # 👈 新增：用于绘图
import time
from datetime import timedelta
# ======================
# 同级导入
# ======================
from dataset import ChladniDataset
from network import BasicCNN

# ======================
# 可视化函数（新增）
# ======================
def plot_training_curves(train_losses, val_accuracies, save_path):
    """绘制训练损失和验证准确率曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左侧：训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 右侧：验证准确率
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, color=color, marker='s', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress: Loss and Accuracy')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 避免内存泄漏

def main():
    # ===== 配置 =====
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "processed"
    MODEL_SAVE_PATH = PROJECT_ROOT / "model" / "best_model.pth"
    PLOT_SAVE_PATH = PROJECT_ROOT / "plots" / "training_curve.png"  # 👈 新增：绘图保存路径
    (PROJECT_ROOT / "plots").mkdir(exist_ok=True)  # 自动创建 plots 文件夹

    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    # ===== 数据预处理 =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ===== 加载数据 =====
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ===== 模型 & 优化器 =====
    model = BasicCNN(num_classes=15).to(DEVICE)
    
    print("Model loaded successfully.")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ===== 手动测试数据集 =====
    print("🔍 Testing dataset...")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    try:
        img, lbl = train_dataset[0]
        print(f"✅ Sample loaded: image shape={img.shape}, label={lbl}")
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        raise

    # ===== 初始化记录列表（新增）=====
    train_losses = []
    val_accuracies = []
    epoch_times = []
    total_train_start = time.time()

    # ===== 训练循环 =====
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} - Training =====")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{total_batches} | Current Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / total_batches
        train_losses.append(avg_train_loss)  # 👈 记录

        # Validation
        model.eval()
        correct, total = 0, 0
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} - Validation =====")
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)  # 👈 记录

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best Val Acc: {best_val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"📌 New best model saved! Val Acc: {best_val_acc:.4f}")

        # 👇 新增：每个 epoch 后更新可视化（可选：也可只在最后画）
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        print(f"⏱️ Epoch {epoch+1} 耗时: {timedelta(seconds=int(epoch_duration))}")
        plot_training_curves(train_losses, val_accuracies, PLOT_SAVE_PATH)

    print(f"\n✅ Training finished completely!")
    print(f"🏆 Best Val Acc: {best_val_acc:.4f}")
    total_train_end = time.time()
    total_duration = total_train_end - total_train_start
    print(f"⏱️ 总训练时间: {timedelta(seconds=int(total_duration))}")
    print(f"⏱️ 平均每个Epoch耗时: {timedelta(seconds=int(sum(epoch_times)/len(epoch_times)))}")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📊 Training curve saved to: {PLOT_SAVE_PATH}")

if __name__ == '__main__':
    main()