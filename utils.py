# utils.py
import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, beta=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth)
        self.beta = beta
        
    def forward(self, pred, target):
        return self.beta * self.bce(pred, target) + (1 - self.beta) * self.dice(pred, target)

def compute_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target.sum() + 1e-5)
    return dice.item()

def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-5)
    return iou.item()