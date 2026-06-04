# model/network.py
import torch.nn as nn
from cbam import CBAM  # ← 新增导入

class BasicCNN_CBAM(nn.Module):
    def __init__(self, num_classes=15):
        super(BasicCNN_CBAM, self).__init__()
        # 前3个卷积块保持不变
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (128, 28, 28)
        )
        # 第4个卷积层 + CBAM（不能塞进 Sequential，因为要插模块）
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.cbam = CBAM(in_channels=256, reduction=16, kernel_size=5)  # ← 推荐 kernel_size=5（更适合细线）

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> (256, 4, 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)           # 到 (128, 28, 28)
        x = self.conv4(x)              # (256, 28, 28)
        x = self.relu4(x)
        x = self.cbam(x)               # ← 应用 CBAM
        x = self.adaptive_pool(x)      # (256, 4, 4)
        x = self.classifier(x)
        return x