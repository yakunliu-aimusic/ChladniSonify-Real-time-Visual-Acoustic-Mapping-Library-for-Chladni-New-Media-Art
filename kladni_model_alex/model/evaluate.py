import sys
import time  # 新增：计时模块
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from model.dataset import ChladniDataset
from model.network import AlexNet  # 保持AlexNet模型导入不变

# ===== 配置 =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 15  # 统一类别数，方便维护
BATCH_SIZE = 32   # 新增：批量大小配置，方便统一管理

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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== 加载模型 =====
model = AlexNet(num_classes=NUM_CLASSES).to(DEVICE)
# 优化模型加载逻辑，兼容CPU/GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.eval()

# ===== 测试纯模型推理速度（不含数据加载）=====
print("⚡ 测试纯模型推理速度（不含数据加载）...")
# 1. 构造单张测试图片（模拟224×224×3的输入，匹配AlexNet输入尺寸）
test_img = torch.randn(1, 3, 224, 224).to(DEVICE)

# 2. 预热模型（避免第一次推理慢，保证测速准确）
for _ in range(10):
    with torch.no_grad():
        _ = model(test_img)

# 3. 正式测试（取100次平均，降低误差）
start_pure = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(test_img)
end_pure = time.time()

# 计算纯模型单张推理速度
pure_infer_time = (end_pure - start_pure) / 100  # 单张耗时（秒）
pure_throughput = 1 / pure_infer_time            # 吞吐量（张/秒）
print(f"📊 纯模型推理速度：{pure_infer_time*1000:.2f} ms/张 | 吞吐量：{pure_throughput:.2f} 张/秒")

# ===== 预测（含数据加载，计算整体推理速度）=====
all_preds, all_labels = [], []
start_total = time.time()  # 记录整体推理开始时间

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 计算整体推理速度（含数据加载+模型推理）
end_total = time.time()
total_time = end_total - start_total
total_samples = len(test_dataset)
avg_total_time = total_time / total_samples  # 单张平均耗时（含加载）
total_throughput = 1 / avg_total_time        # 整体吞吐量

# 转换为numpy数组，适配sklearn指标计算
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ===== 计算核心指标 =====
# 1. 准确率
acc = np.mean(all_preds == all_labels)
# 2. 宏平均F1-score（多分类核心指标）
macro_f1 = f1_score(all_labels, all_preds, average='macro')
# 3. 微平均F1-score（备用参考）
micro_f1 = f1_score(all_labels, all_preds, average='micro')

# ===== 输出结果 =====
print("\n" + "="*60)
print("📈 AlexNet 测试集评估结果")
print("="*60)
# 推理速度输出
print(f"⚡ 整体推理速度（含数据加载）：{avg_total_time*1000:.2f} ms/张 | 吞吐量：{total_throughput:.2f} 张/秒")
# 准确率/F1-score输出
print(f"🎯 Test Accuracy: {acc:.4f}")
print(f"🏆 Macro-F1 Score (宏平均): {macro_f1:.4f}")
print(f"🔍 Micro-F1 Score (微平均): {micro_f1:.4f}")

# 分类报告
print("\n📋 Classification Report (包含每类Precision/Recall/F1):")
target_names = [f"Mode_{i}" for i in range(NUM_CLASSES)]
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=target_names,
    digits=4  # 保留4位小数，便于论文制表
))

# 可选：输出混淆矩阵（如需分析AlexNet的类别误判情况，取消注释）
# print("\n🌀 Confusion Matrix:")
# cm = confusion_matrix(all_labels, all_preds)
# print(cm)