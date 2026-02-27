import torch.nn as nn
import torch
import numpy as np

def dice(y_pred, y_true, smooth=1e-6):
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
def iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou
def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "iou": iou,
        "dice": dice,
    }
    scores = {metric_name: [] for metric_name in metrics}
    
    for predicted_mask, mask in zip(predicted_masks, masks):  
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(predicted_mask, mask)) 
    
    return {metric_name: np.mean(values) for metric_name, values in scores.items()}

