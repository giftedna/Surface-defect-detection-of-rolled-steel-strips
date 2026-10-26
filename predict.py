# -*- coding: utf-8 -*-
"""推理脚本（单图/批量检测，GPU 加速）。"""
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import SteelDefectCNN

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weight_path=None):
    """加载训练好的模型。"""
    if weight_path is None:
        weight_path = os.path.join(BASE, "best_model.pth")
    model = SteelDefectCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model


def predict_image(model, img_path):
    """对单张图片进行预测，返回 (类别名, 置信度)。"""
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
    return CLASS_NAMES[pred.item()], conf.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="钢带表面缺陷检测")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径或文件夹")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    args = parser.parse_args()

    model = load_model(args.weights)

    if os.path.isfile(args.input):
        # 单张图片
        cls, conf = predict_image(model, args.input)
        print(f"图片: {args.input}")
        print(f"预测: {cls}  (置信度: {conf:.4f})")
    elif os.path.isdir(args.input):
        # 批量检测文件夹
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(args.input) if f.lower().endswith(exts)]
        print(f"共 {len(files)} 张图片")
        for f in sorted(files):
            path = os.path.join(args.input, f)
            cls, conf = predict_image(model, path)
            print(f"  {f:30s} -> {cls:12s} ({conf:.4f})")
    else:
        print(f"路径不存在: {args.input}")
