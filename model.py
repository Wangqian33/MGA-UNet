# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Ghost Module --------------------
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, ratio=2):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

# -------------------- RFB Module --------------------
class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, dilation=5, padding=10),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        out = self.fusion(out)
        return out

# -------------------- Ghost+RFB 模块 --------------------
class GhostRFB(nn.Module):
    def __init__(self, in_channels, out_channels, ghost_ratio=4):
        super(GhostRFB, self).__init__()
        self.ghost1 = GhostModule(in_channels, out_channels, kernel_size=3, padding=1, ratio=ghost_ratio)
        self.rfb = RFB(out_channels, out_channels)
        self.ghost2 = GhostModule(out_channels, out_channels, kernel_size=3, padding=1, ratio=ghost_ratio)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.ghost1(x)
        x = self.rfb(x) + x  # 残差连接
        x = self.ghost2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# -------------------- CBAM 注意力 --------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# -------------------- Attention Gate --------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------------------- MGA-UNet --------------------
class MGAUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, ghost_ratio=4, dropout_rate=0.5):
        super(MGAUNet, self).__init__()
        
        # 编码器
        self.enc1 = GhostRFB(in_channels, 64, ghost_ratio)
        self.enc2 = GhostRFB(64, 128, ghost_ratio)
        self.enc3 = GhostRFB(128, 256, ghost_ratio)
        self.enc4 = GhostRFB(256, 512, ghost_ratio)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 桥接
        self.bridge = GhostRFB(512, 1024, ghost_ratio)
        
        # 注意力门控
        self.att4 = AttentionGate(1024, 512, 256)
        self.att3 = AttentionGate(512, 256, 128)
        self.att2 = AttentionGate(256, 128, 64)
        self.att1 = AttentionGate(128, 64, 32)
        
        # CBAM
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam_bridge = CBAM(1024)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = GhostRFB(1024, 512, ghost_ratio)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = GhostRFB(512, 256, ghost_ratio)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = GhostRFB(256, 128, ghost_ratio)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = GhostRFB(128, 64, ghost_ratio)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e1 = self.cbam1(e1)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        e2 = self.cbam2(e2)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        e3 = self.cbam3(e3)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        e4 = self.cbam4(e4)
        p4 = self.pool(e4)
        
        b = self.bridge(p4)
        b = self.cbam_bridge(b)
        b = self.dropout(b)
        
        d4 = self.up4(b)
        att4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([att4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        att3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([att3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        att2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([att2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        att1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([att1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return out