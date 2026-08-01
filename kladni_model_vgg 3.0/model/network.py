# model/network.py
import torch.nn as nn
import torchvision.models as models

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(VGG16Classifier, self).__init__()
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=pretrained)
        
        # Freeze feature extractor (optional fine-tuning based on dataset size)
        # Freeze for small datasets; unfreeze layers for larger datasets
        # for param in vgg.features.parameters():
        #     param.requires_grad = False
        
        # Replace final classifier
        # Original VGG16 classifier: 4096 -> 4096 -> 1000
        # Replaced with: 4096 -> 512 -> num_classes
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