# src/models/cnn.py
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=50, input_channels=1):
        super().__init__()
        # 4 khối Conv (như paper), nhưng chỉ dùng 3 MaxPool + AdaptiveAvgPool ở cuối
        self.features = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, padding=1),   # Conv thứ 4 (giống paper)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))         # ép về 1×1, không gây lỗi
        )
        # Tự động suy ra số kênh cuối cùng (256) và nhân với 1*1 = 256
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),   # 256 kênh → 256 nơ-ron (paper dùng 256)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))