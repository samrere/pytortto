import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


#################################
#       Transfer Learning       #
#################################
class YOLOv1ResNet(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.depth = B * 5 + C

        # Load backbone ResNet
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # backbone.requires_grad_(False) # Freeze backbone weights

        # Delete last two layers and attach detection layers
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone=backbone

        self.model = DetectionNet(2048, S=S, B=B, C=C)              # 4 conv, 2 linear


    def forward(self, x):
        x=self.backbone(x)
        x=torch.reshape(x,(-1, 2048, 14, 14))
        x=self.model(x)
        return x

class DetectionNet(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels, S=7, B=2, C=20):
        super().__init__()
        self.S=S
        inner_channels = 1024
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, S * S * (5 * B + C))
        )

    def forward(self, x):
        return self.model(x)

