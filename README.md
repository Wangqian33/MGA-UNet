# MGA-UNet

This repository contains the official PyTorch implementation of the paper:  
"Integration of Multi-scale Features and Attention Mechanisms for Colorectal Tumor CT Image Segmentation" (2025)

# Requirements

- Python 3.9+
- PyTorch 1.12+
- Install dependencies: `pip install -r requirements.txt`

# Dataset Preparation

Your raw data should be organized as:
raw_data/
├── train/
│ ├── patient001/ # CT slices
│ └── patient002/
└── label/
├── patient001/ # corresponding masks
└── patient002/

Run `organize_dataset.py` to convert it into the required structure (train/val/test split, images and masks flattened).

# Training

Modify `config.py` to set your data path, then:
python train.py

# Testing

python test.py

# Demo

To verify the model works, run:
python demo.py
