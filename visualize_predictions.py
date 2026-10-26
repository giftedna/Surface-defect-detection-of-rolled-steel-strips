# -*- coding: utf-8 -*-
"""测试集预测可视化：随机展示16张测试样本，绿色=预测正确，红色=预测错误。"""
import os
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

from model import SteelDefectCNN
from dataset import DATA_ROOT, IMG_SIZE, eval_transform

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试集
test_dir = os.path.join(DATA_ROOT, "test")
test_set = datasets.ImageFolder(test_dir, transform=eval_transform)
class_names = test_set.classes

# 加载模型
model = SteelDefectCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(BASE, "best_model.pth"), map_location=DEVICE))
model.eval()

# 预测所有测试集
all_preds = []
all_labels = []
all_imgs = []
with torch.no_grad():
    for i in range(len(test_set)):
        img_tensor, label = test_set[i]
        out = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred = out.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(label)
        # 保留原始图像用于可视化（反归一化）
        img = img_tensor.clone()
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1)
        all_imgs.append(img)

# 随机选16张
indices = random.sample(range(len(test_set)), 16)

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle("Test Set Predictions (Green=Correct, Red=Wrong)", fontsize=14, y=0.98)

for idx, ax in zip(indices, axes.flat):
    img = all_imgs[idx]
    pred = all_preds[idx]
    true = all_labels[idx]
    correct = pred == true

    ax.imshow(img)
    ax.axis("off")
    color = "green" if correct else "red"
    ax.set_title(f"Pred:{class_names[pred]}\nTrue:{class_names[true]}",
                 color=color, fontsize=11, fontweight="bold")

plt.tight_layout()
out_path = os.path.join(BASE, "test_predictions.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"已保存: {out_path}")

# 统计正确率
correct_count = sum(1 for p, t in zip(all_preds, all_labels) if p == t)
print(f"测试集准确率: {correct_count}/{len(test_set)} = {correct_count/len(test_set)*100:.2f}%")
