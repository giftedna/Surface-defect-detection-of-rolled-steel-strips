# -*- coding: utf-8 -*-
"""数据集加载（PyTorch 版）。"""
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 原项目 NEU 数据集路径
DATA_ROOT = r"D:\Python_project\Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips-master\NEU Metal Surface Defects Data"

BATCH_SIZE = 32
IMG_SIZE = (200, 200)

# 训练集：数据增强 + 归一化
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 验证/测试集：仅归一化
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders():
    """返回 train_loader, val_loader, test_loader, class_names。"""
    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "valid")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = train_set.classes
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"类别: {class_names}")
    print(f"训练集: {len(train_loader.dataset)} 张")
    print(f"验证集: {len(val_loader.dataset)} 张")
    print(f"测试集: {len(test_loader.dataset)} 张")
