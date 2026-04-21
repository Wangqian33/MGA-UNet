# train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from dataset import CTDataset, get_train_transform, get_val_transform
from model import MGAUNet
from utils import BCEDiceLoss, compute_dice

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # 数据集
    train_dataset = CTDataset(
        cfg.train_img_dir, cfg.train_mask_dir,
        image_size=cfg.image_size,
        transform=get_train_transform(cfg.image_size)
    )
    val_dataset = CTDataset(
        cfg.val_img_dir, cfg.val_mask_dir,
        image_size=cfg.image_size,
        transform=get_val_transform(cfg.image_size)
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    # 模型
    model = MGAUNet(in_channels=1, num_classes=1, ghost_ratio=cfg.ghost_ratio, dropout_rate=cfg.dropout_rate)
    model.to(cfg.device)
    
    # 损失与优化器
    criterion = BCEDiceLoss(beta=cfg.beta)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.initial_lr, weight_decay=cfg.weight_decay)
    
    if cfg.use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.lr_T_max)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_dice = 0.0
    early_stop_counter = 0
    train_losses, val_losses, val_dices = [], [], []
    
    for epoch in range(cfg.num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Train]'):
            imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Val]'):
                imgs, masks = imgs.to(cfg.device), masks.to(cfg.device)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                for i in range(imgs.size(0)):
                    val_dice += compute_dice(preds[i], masks[i])
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_dataset)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.4f}')
        
        scheduler.step()
        
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= cfg.early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 绘制曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(cfg.log_dir, 'loss_curve.png'))
    plt.figure()
    plt.plot(val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig(os.path.join(cfg.log_dir, 'dice_curve.png'))
    
    print(f'Training finished. Best Val Dice: {best_dice:.4f}')

if __name__ == '__main__':
    train()