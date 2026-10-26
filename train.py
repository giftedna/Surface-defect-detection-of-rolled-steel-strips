# -*- coding: utf-8 -*-
"""训练脚本（PyTorch + GPU）。"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SteelDefectCNN
from dataset import get_dataloaders

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 数据
train_loader, val_loader, test_loader, class_names = get_dataloaders()
print(f"类别: {class_names}")

# 模型
model = SteelDefectCNN(num_classes=len(class_names)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# 训练记录
# train_losses, train_accs = [], []
# val_losses, val_accs = [], []
#
# best_val_acc = 0.0
# for epoch in range(EPOCHS):
#     # ---- 训练 ----
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     t0 = time.time()
#     for images, labels in train_loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     train_loss = running_loss / total
#     train_acc = correct / total
#     train_losses.append(train_loss)
#     train_accs.append(train_acc)
#
#     # ---- 验证 ----
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     val_loss /= total
#     val_acc = correct / total
#     val_losses.append(val_loss)
#     val_accs.append(val_acc)
#
#     dt = time.time() - t0
#     print(f"Epoch [{epoch+1}/{EPOCHS}]  "
#           f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
#           f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
#           f"({dt:.1f}s/epoch)")
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), os.path.join(BASE, "best_model.pth"))

# ---- 测试集评估 ----
model.load_state_dict(torch.load(os.path.join(BASE, "best_model.pth")))
model.eval()
test_loss = 0.0
correct = 0
total = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_loss /= total
test_acc = correct / total
print(f"\n=== 测试集结果 ===")
print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

# ---- 训练曲线 ----
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
# ax1.plot(train_accs, label="训练准确率")
# ax1.plot(val_accs, label="验证准确率")
# ax1.set_title("模型准确率"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
# ax1.legend(); ax1.grid(alpha=0.3)
# ax2.plot(train_losses, label="训练损失")
# ax2.plot(val_losses, label="验证损失")
# ax2.set_title("模型损失"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
# ax2.legend(); ax2.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(BASE, "training_curves.png"), dpi=150)
# plt.close()

# ---- 混淆矩阵 ----
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
ax.set_title(f"测试集混淆矩阵（准确率 {test_acc:.2%}）")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "confusion_matrix.png"), dpi=150)
plt.close()

print("\n=== 分类报告 ===")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("\n模型已保存: best_model.pth")
print("训练曲线: training_curves.png")
print("混淆矩阵: confusion_matrix.png")
