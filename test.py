# test.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from config import Config
from dataset import CTDataset, get_val_transform
from model import MGAUNet
from utils import compute_dice, compute_iou

def test():
    cfg = Config()
    test_dataset = CTDataset(
        cfg.test_img_dir, cfg.test_mask_dir,
        image_size=cfg.image_size,
        transform=get_val_transform(cfg.image_size)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = MGAUNet(in_channels=1, num_classes=1, ghost_ratio=cfg.ghost_ratio, dropout_rate=cfg.dropout_rate)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'best_model.pth'), map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    
    dice_list = []
    iou_list = []
    
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(test_loader, desc='Testing')):
            img = img.to(cfg.device)
            pred = model(img)
            pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            
            dice = compute_dice(pred, mask)
            iou = compute_iou(pred, mask)
            dice_list.append(dice)
            iou_list.append(iou)
            
            # 保存预测结果
            Image.fromarray(pred_np).save(os.path.join(cfg.result_dir, f'pred_{idx}.png'))
    
    avg_dice = np.mean(dice_list)
    avg_iou = np.mean(iou_list)
    print(f'Test Results: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}')

if __name__ == '__main__':
    test()