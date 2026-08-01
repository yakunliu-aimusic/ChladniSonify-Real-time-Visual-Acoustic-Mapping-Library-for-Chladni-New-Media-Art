# model/network.py
import torch.nn as nn
from cbam import CBAM  # ← New import

class BasicCNN_CBAM(nn.Module):
    def __init__(self, num_classes=15):
        super(BasicCNN_CBAM, self).__init__()
        # First three conv blocks unchanged
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
        # 4th conv layer + CBAM (not in Sequential due to module insertion)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.cbam = CBAM(in_channels=256, reduction=16, kernel_size=5)  # ← Recommended kernel_size=5 (better for thin nodal lines)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> (256, 4, 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)           # -> (128, 28, 28)
        x = self.conv4(x)              # (256, 28, 28)
        x = self.relu4(x)
        x = self.cbam(x)               # ← Apply CBAM
        x = self.adaptive_pool(x)      # (256, 4, 4)
        x = self.classifier(x)
        return x