# -*- coding: utf-8 -*-
"""CNN 模型定义（PyTorch 版）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SteelDefectCNN(nn.Module):
    """热轧钢带表面缺陷分类 CNN（与原 Keras 版结构对应）。

    结构：
      Conv2D(3->32, 2x2) + ReLU + MaxPool(2)
      Conv2D(32->64, 2x2) + ReLU + MaxPool(2)
      Conv2D(64->128, 2x2) + ReLU + MaxPool(2)
      Flatten -> Linear(73728->256) + ReLU + Dropout(0.2)
      Linear(256->6)
    """

    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.pool = nn.MaxPool2d(2, 2)
        # 输入 200x200: conv1->199, pool->99, conv2->98, pool->49, conv3->48, pool->24
        self.fc1 = nn.Linear(128 * 24 * 24, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> 99x99x32
        x = self.pool(F.relu(self.conv2(x)))   # -> 49x49x64
        x = self.pool(F.relu(self.conv3(x)))   # -> 24x24x128
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = SteelDefectCNN()
    x = torch.randn(1, 3, 200, 200)
    out = model(x)
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")
