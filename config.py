# config.py
import os
import torch

class Config:
    # 数据集路径（请修改为您的 data_organized 所在路径）
    data_root = r"C:\Users\User\Desktop\图像分割数据集\（脑肿瘤）U-Net基础篇\data_organized"
    
    # 训练参数
    image_size = 256
    batch_size = 8
    num_epochs = 100
    initial_lr = 1e-4
    weight_decay = 1e-5
    beta = 0.5          # BCE-Dice 平衡权重
    dropout_rate = 0.5
    
    # Ghost 模块压缩比 s
    ghost_ratio = 4
    
    # 早停
    early_stop_patience = 15
    
    # 学习率调度
    use_cosine_annealing = True
    lr_T_max = 100
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 保存路径
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    result_dir = './results'
    
    # 随机种子
    seed = 42
    
    # 数据集自动构建路径（无需手动修改）
    train_img_dir = os.path.join(data_root, 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'train', 'masks')
    val_img_dir = os.path.join(data_root, 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'val', 'masks')
    test_img_dir = os.path.join(data_root, 'test', 'images')
    test_mask_dir = os.path.join(data_root, 'test', 'masks')