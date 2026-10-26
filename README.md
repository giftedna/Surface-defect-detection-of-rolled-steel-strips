# Hot-Rolled Steel Strip Surface Defect Detection (PyTorch + GPU)

A six-class surface defect detection project for hot-rolled steel strips, implemented in PyTorch with GPU-accelerated training and inference.

## Requirements

- Python 3.11
- PyTorch 2.x (CUDA 13.0 verified)
- Dataset: NEU Metal Surface Defects Data (6 defect classes)

## Dataset Important Notice

- **Dataset Name**: NEU Metal Surface Defects Data
- **Source**: https://github.com/aviralchharia/Surface-Defect-Detection-in-Hot-Rolled-Steel-Strips
- **Author**: Aviral Chharia

The original repository does not provide an explicit open-source license. This repository only contains self-implemented training and inference code for academic research and graduation project purposes. Dataset files are NOT included in this repository. Please download the dataset from the link above. Redistribution or commercial use of the NEU Metal Surface Defects Data is prohibited without explicit permission from the original author.

### Pre-trained Weights
The trained weight `best_model.pth` is not stored in this repository to reduce repository size.
Option 1: Run `train.py` from scratch to train and generate the checkpoint.
Option 2: Download the pre-trained weight from https://pan.baidu.com/s/1b_-rX_51-2tbneil8dhfdQ?pwd=tsbn, and place it in the project root directory before inference.

## Project Structure

```
Surface defect detection of rolled steel strips/
├── model.py                    # CNN model definition
├── dataset.py                  # Dataset loading
├── train.py                    # Training script (GPU accelerated)
├── predict.py                  # Inference script
├── visualize_predictions.py    #Visualize your predictions
├── README.md                   # This file
└── best_model.pth              # Trained weights (generated after training)

```

## Quick Start

### 1. Training

```powershell
cd "D:\Python_project\Surface defect detection of rolled steel strips"
D:\Anaconda3\envs\pyenv\python.exe train.py
```

Default: 20 epochs, ~1-2 minutes on GPU. To change epochs:

```powershell
$env:EPOCHS="30"; D:\Anaconda3\envs\pyenv\python.exe train.py
```

After training, the following are generated automatically:
- `best_model.pth` — best model weights
- `training_curves.png` — training/validation accuracy and loss curves
- `confusion_matrix.png` — test set confusion matrix

### 2. Inference (single image)

```powershell
D:\Anaconda3\envs\pyenv\python.exe predict.py --input "test_image.jpg"
```

### 3. Inference (batch)

```powershell
D:\Anaconda3\envs\pyenv\python.exe predict.py --input "image_folder"
```

## Six Defect Classes

| Class | Description |
|---|---|
| Crazing | Surface cracks |
| Inclusion | Foreign material pressed in |
| Patches | Local patches |
| Pitted | Surface pits |
| Rolled | Oxide scale pressed in |
| Scratches | Surface scratches |

## Mapping to Original Keras Version

| Keras | PyTorch |
|---|---|
| Conv2D(32, 2x2) + ReLU + MaxPool | `nn.Conv2d(3,32,2)` + ReLU + MaxPool |
| Conv2D(64, 2x2) + ReLU + MaxPool | `nn.Conv2d(32,64,2)` + ReLU + MaxPool |
| Conv2D(128, 2x2) + ReLU + MaxPool | `nn.Conv2d(64,128,2)` + ReLU + MaxPool |
| Dense(256) + Dropout(0.2) | `nn.Linear(73728,256)` + Dropout |
| Dense(6, softmax) | `nn.Linear(256,6)` |
| RMSprop | `optim.RMSprop` |
| 20 epochs | Default 20 epochs |

## Notes

- Dataset path is configured in `dataset.py` (defaults to the original NEU dataset)
- Model weights are saved in `.pth` format (weights only, no optimizer state), ~75 MB
- Automatically falls back to CPU if GPU is unavailable
