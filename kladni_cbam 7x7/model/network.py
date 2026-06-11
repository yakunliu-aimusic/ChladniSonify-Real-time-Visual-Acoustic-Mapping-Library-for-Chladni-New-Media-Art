# model/network.py
import torch.nn as nn
from cbam import CBAM

class BasicCNN_CBAM(nn.Module):
    def __init__(self, num_classes=15):
        super(BasicCNN_CBAM, self).__init__()
        # 所有卷积层 kernel_size=5, padding=2
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # 5×5
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 5×5
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),# 5×5
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 第4层也改为 5×5 卷积
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.relu4 = nn.ReLU()
        
        # ✅ CBAM 空间卷积核明确设置为 5×5（你论文的优化版本）
        self.cbam = CBAM(in_channels=256, reduction=16, kernel_size=5)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.cbam(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x