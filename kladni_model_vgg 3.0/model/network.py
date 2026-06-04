# model/network.py
import torch.nn as nn
import torchvision.models as models

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(VGG16Classifier, self).__init__()
        # 加载预训练的 VGG16
        vgg = models.vgg16(pretrained=pretrained)
        
        # 冻结特征提取层（可选：可根据数据量决定是否微调）
        # 如果数据量小，建议冻结；如果大，可以解冻部分层
        # for param in vgg.features.parameters():
        #     param.requires_grad = False
        
        # 替换最后的分类器
        # 原 VGG16 分类器：4096 -> 4096 -> 1000
        # 我们改为：4096 -> 512 -> num_classes
        vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.vgg = vgg

    def forward(self, x):
        return self.vgg(x)