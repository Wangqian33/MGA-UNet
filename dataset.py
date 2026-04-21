# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CTDataset(Dataset):
    """
    CT图像分割数据集加载器。
    假设：
    - 图像和掩膜均为2D灰度图（PNG/JPG/BMP等格式）。
    - 图像像素值范围为0-255（已进行过窗宽截断和直方图均衡化预处理）。
    - 掩膜为二值图：前景（肿瘤）像素值>128，背景≤128。
    因此，代码中仅进行归一化（除以255）和二值化（>128），不再重复预处理。
    """
    def __init__(self, img_dir, mask_dir, image_size=256, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
        assert len(self.images) == len(self.masks), "图片和掩膜数量不一致"
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # 读取图像（灰度）和掩膜（灰度）
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 二值化掩膜：前景>128 → 1，背景→0
        if mask.max() > 1:
            mask = (mask > 128).astype(np.float32)
        else:
            mask = (mask > 0.5).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 归一化图像到 [0,1]（假设原始像素范围0-255）
        image = image.astype(np.float32) / 255.0
        
        # 添加通道维度 (C, H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

def get_train_transform(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=0.01, p=0.3),
        ToTensorV2(),
    ])

def get_val_transform(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])