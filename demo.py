# demo.py
import torch
from model import MGAUNet

def demo():
    print("=" * 50)
    print("MGA-UNet Demo: Testing model forward pass")
    print("=" * 50)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = MGAUNet(in_channels=1, num_classes=1, ghost_ratio=4, dropout_rate=0.5)
    model.to(device)
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建随机输入 (batch=1, channel=1, height=256, width=256)
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    print(f"Input shape: {dummy_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print("Forward pass successful! Model is working correctly.")
    print("=" * 50)

if __name__ == '__main__':
    demo()