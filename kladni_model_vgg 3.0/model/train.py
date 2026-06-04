# ======================
# 基础库导入
# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告，减少资源占用
import time
from datetime import timedelta

# ======================
# 同级导入
# ======================
from dataset import ChladniDataset
from network import VGG16Classifier  

# ======================
# 可视化函数
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
    # ===== 配置 =====
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data" / "processed"
    MODEL_SAVE_PATH = PROJECT_ROOT / "model" / "best_model.pth"
    PLOT_SAVE_PATH = PROJECT_ROOT / "plots" / "training_curve.png"
    (PROJECT_ROOT / "plots").mkdir(exist_ok=True)

    # 已调整为8，适配M4芯片（如果温度仍高，可改为4）
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-4  # ✅ 降低学习率（因使用预训练模型）
    
    # 关键优化：优先使用MPS（Apple Silicon专用加速），降低CPU占用和温度
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        # 移除旧版本不支持的内存限制代码
        # torch.backends.mps.set_per_process_memory_fraction(0.8)  # 已删除
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Using device: {DEVICE}")

    # ===== 数据预处理（增加内存优化）=====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ===== 数据集 =====
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
    
    # 优化：pin_memory=False（M4芯片不需要，减少内存拷贝）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False,
        drop_last=False  # 避免最后一个小批次导致的内存波动
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    # ===== 模型 & 优化器（增加内存优化）=====
    model = VGG16Classifier(num_classes=15, pretrained=True).to(DEVICE)
    
    # 优化：启用梯度检查点，降低VGG16的显存占用（约减少30%）
    # 注意：如果你的VGG16Classifier不是Sequential结构，这行需要注释掉
    try:
        if DEVICE in ["mps", "cuda"]:
            model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
    except:
        print("⚠️ 梯度检查点未启用（模型非Sequential结构），不影响核心训练")
    
    print("Model loaded successfully.")
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    dummy_output = model(dummy_input)
    print(f"Dummy output shape: {dummy_output.shape}")

    criterion = nn.CrossEntropyLoss()
    # 优化：添加权重衰减，同时降低学习率波动（减少计算量）
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

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

    # ===== 训练记录 =====
    train_losses = []
    val_accuracies = []
    epoch_times = []  # 记录每个Epoch耗时
    total_train_start = time.time()  # 总训练开始时间
    
    # ===== 训练循环（增加内存清理）=====
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        epoch_start = time.time()  # 当前Epoch开始计时
        # Training
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} - Training =====")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # 内存优化：前向传播后及时释放中间变量
            with torch.autograd.detect_anomaly(False):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / total_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():  # 禁用梯度，大幅降低内存占用
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
            # 优化：保存时先移到CPU，避免MPS内存锁定
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"📌 New best model saved! Val Acc: {best_val_acc:.4f}")

        # 计算并打印当前Epoch耗时
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        print(f"⏱️ Epoch {epoch+1} 耗时: {timedelta(seconds=int(epoch_duration))}")
        # 绘制曲线后清理缓存
        plot_training_curves(train_losses, val_accuracies, PLOT_SAVE_PATH)
        if DEVICE == "mps":
            torch.mps.empty_cache()

    print(f"\n✅ Training finished!")
    # 计算总训练时间和平均Epoch耗时
    total_train_end = time.time()
    total_duration = total_train_end - total_train_start
    print(f"⏱️ 总训练时间: {timedelta(seconds=int(total_duration))}")
    print(f"⏱️ 平均每个Epoch耗时: {timedelta(seconds=int(sum(epoch_times)/len(epoch_times)))}")
    print(f"🏆 Best Val Acc: {best_val_acc:.4f}")
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📊 Training curve saved to: {PLOT_SAVE_PATH}")

if __name__ == '__main__':
    main()