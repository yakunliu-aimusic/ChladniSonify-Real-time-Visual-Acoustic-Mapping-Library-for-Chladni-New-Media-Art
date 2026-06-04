import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import time
import warnings  # 新增：用于屏蔽警告
from model.dataset import ChladniDataset
from model.network import VGG16Classifier

# ===== 关键：屏蔽PyTorch的pretrained弃用警告 =====
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ===== 配置 =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 15
BATCH_SIZE = 16  # 减小批次，提升CPU推理速度
NUM_WORKERS = 0

# ===== 数据 =====
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

# ===== 加载模型 =====
# 改回pretrained=False，适配你的自定义VGG16Classifier类
model = VGG16Classifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    print(f"✅ VGG16模型权重加载成功：{MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ 权重文件不存在！请检查路径：{MODEL_PATH}")
    raise
except RuntimeError as e:
    print(f"❌ 权重文件与模型不匹配：{e}")
    raise

model.eval()
print(f"🔧 使用设备：{DEVICE} | 测试集样本数：{len(test_dataset)}")
print(f"⚡ 推理配置：批次大小={BATCH_SIZE} | 开始预测...")

# ===== 预测（带进度+计时）=====
all_preds, all_labels = [], []
start_time = time.time()

with torch.no_grad():
    # CPU多线程优化，提升VGG16推理速度
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        # 实时进度+预估剩余时间
        elapsed_time = time.time() - start_time
        progress = (batch_idx + 1) / len(test_loader)
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        
        print(f"\r📊 预测进度：{batch_idx + 1}/{len(test_loader)} 批次 "
              f"({progress*100:.1f}%) | 已耗时：{elapsed_time:.1f}s "
              f"| 预估剩余：{eta:.1f}s", end="")
        
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 结果处理
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
total_time = time.time() - start_time
print(f"\n✅ 预测完成！总耗时：{total_time:.1f}s | 平均每样本：{total_time/len(test_dataset):.3f}s")

# ===== 计算指标 =====
acc = np.mean(all_preds == all_labels)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
micro_f1 = f1_score(all_labels, all_preds, average='micro')

# ===== 输出结果 =====
print("\n" + "="*60)
print("📈 VGG16Classifier 测试集评估结果")
print("="*60)
print(f"🎯 Test Accuracy (测试准确率): {acc:.4f}")
print(f"🏆 Macro-F1 Score (宏平均F1): {macro_f1:.4f}")
print(f"🔍 Micro-F1 Score (微平均F1): {micro_f1:.4f}")
print("\n📋 分类报告（每类Precision/Recall/F1）:")
target_names = [f"Mode_{i}" for i in range(NUM_CLASSES)]
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=target_names,
    digits=4
))