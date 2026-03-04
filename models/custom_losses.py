"""
Custom Loss Functions - Fixes the DiceLoss weight calculation bug in original u-sam.py

Implemented in the Experiment folder without modifying the original u-sam.py.

Author: U-SAM Adaptive Prompt Team
Date: 2026-02-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedDiceLoss(nn.Module):
    """
    Fixed Dice Loss - Correctly handles class weights
    
    Original Bug: loss += dice * weight[i]  # Larger weight leads to smaller loss (Incorrect!)
    Fixed: loss += (1 - dice) * weight[i]  # Larger weight penalizes that class more
    """
    def __init__(self, n_classes):
        super(FixedDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        """
        Calculate Dice coefficient for a single class
        
        Returns:
            dice: Dice coefficient (larger is better, range [0,1])
        """
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice  # Returns dice coefficient, not loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Args:
            inputs: Model output [B, C, H, W]
            target: GT mask [B, H, W], values in [0, C-1]
            weight: List of class weights [w0, w1, ..., wC-1]
            softmax: Whether to apply softmax to inputs
        
        Returns:
            loss: Weighted Dice Loss (smaller is better)
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Convert to one-hot
        target = target.unsqueeze(1)  # [B, 1, H, W]
        target_one_hot = torch.zeros_like(inputs)
        target_one_hot.scatter_(1, target, 1)  # [B, C, H, W]
        
        # Default weights
        if weight is None:
            weight = [1.0] * self.n_classes
        
        # Accumulate weighted Dice Loss for each class
        loss = 0
        class_wise_dice = []
        
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target_one_hot[:, i])
            class_wise_dice.append(dice.item())  # Record dice coefficient
            
            # [Critical Fix] weight should be applied to the loss, not the dice score!
            # dice is a coefficient (larger is better), loss is the cost (smaller is better)
            # Larger weight[i] means this class contributes more to the loss
            loss += (1 - dice) * weight[i]
        
        # Average only over non-background classes (assuming class 0 is background)
        # If all classes should participate, change to self.n_classes
        return loss / (self.n_classes - 1)
