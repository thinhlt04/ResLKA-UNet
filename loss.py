import torch.nn as nn
import torch
import numpy as np

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(2,3))  
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))  
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean() 

def focal_loss(pred, target, alpha=0.8, gamma=2.0, smooth=1e-6):
    pred = torch.clamp(pred, smooth, 1.0 - smooth) 
    bce = -(alpha * target * torch.log(pred) + (1 - alpha) * (1 - target) * torch.log(1 - pred))
    focal = (1 - pred * target - (1 - pred) * (1 - target)) ** gamma * bce
    return focal.mean()