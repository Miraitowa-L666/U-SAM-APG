"""
Stage 1 Training Script: Classification Branch Pre-training

Objective: Train the classification branch network to generate high-quality lesion heatmaps.
Supervision Signal: Foreground binary mask (regions with mask > 0, including rectum + tumor).
Training Configuration:
  - Batch Size: 32
  - Learning Rate: 1e-3 (AdamW)
  - LR Schedule: Warmup + Cosine Annealing
  - Epochs: 50
  - Loss Function: 0.5*Dice + 0.3*Focal + 0.2*BCE

Author: U-SAM Adaptive Prompt Team
Date: 2025-12-14
"""

import os
import sys
# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from models.adaptive_prompt_generator import ClassificationBranch
from dataset.rectum_dataloader import RectumDataloader
import utils.misc as utils


def simple_collate_fn(batch):
    """Simple collate function for Stage 1, without NestedTensor."""
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)  # [B, 3, 224, 224]
    return images, list(targets)


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss."""
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, 1, H, W], range [0,1]
            target: Target binary mask [B, 1, H, W], range {0,1}
        """
        smooth = 1e-5
        pred = pred.view(pred.size(0), -1)
        target = target.view(pred.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss: Addressing class imbalance problem."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, 1, H, W], range [0,1]
            target: Target binary mask [B, 1, H, W], range {0,1}
        """
        # Ensure numerical stability
        pred = pred.clamp(1e-6, 1 - 1e-6)
        
        # Compute cross-entropy
        ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Compute focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weight
        alpha_weight = torch.where(target == 1, 
                                   torch.tensor(self.alpha, device=pred.device),
                                   torch.tensor(1 - self.alpha, device=pred.device))
        
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined Loss: Dice + Focal + BCE."""
    def __init__(self, dice_weight=0.5, focal_weight=0.3, bce_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = BinaryDiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.bce_loss = nn.BCELoss()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, C, H, W]
            target: Target mask [B, C, H, W]
        """
        loss_dice = self.dice_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        loss_bce = self.bce_loss(pred, target)
        
        total_loss = (self.dice_weight * loss_dice + 
                     self.focal_weight * loss_focal + 
                     self.bce_weight * loss_bce)
        
        return total_loss, {
            'dice': loss_dice.item(),
            'focal': loss_focal.item(),
            'bce': loss_bce.item()
        }


class BinaryDiceIndex(nn.Module):
    """Binary Dice Metric (no gradient computation)."""
    def __init__(self):
        super(BinaryDiceIndex, self).__init__()
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, 1, H, W], range [0,1]
            target: Target binary mask [B, 1, H, W], range {0,1}
        Returns:
            dice: Dice coefficient
        """
        smooth = 1e-5
        
        # Binarize prediction
        pred_binary = (pred >= 0.5).float()
        
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        target = target.view(target.size(0), -1)
        
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    combined_loss_fn: nn.Module,
    iter_num: int = 0
):
    """Train for one epoch using combined loss and scheduler."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = max(1, 100 // args.batch_size)
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Prepare data
        images = samples.to(device)  # [B, 3, 224, 224]
        
        foreground_masks = []
        tumor_masks = []
        for target in targets:
            mask = target['mask'].to(device)  # [224, 224]
            foreground_mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]
            foreground_masks.append(foreground_mask)
            tumor_mask = (mask == 2).float().unsqueeze(0).unsqueeze(0)
            tumor_masks.append(tumor_mask)
        foreground_masks = torch.cat(foreground_masks, dim=0)  # [B, 1, 224, 224]
        tumor_masks = torch.cat(tumor_masks, dim=0)  # [B, 1, 224, 224]
        
        heatmap_all = model(images)  # [B, 2, 224, 224]
        heatmap_fg = heatmap_all[:, 0:1]
        heatmap_tu = heatmap_all[:, 1:2]
        
        # Use combined loss
        loss_fg, loss_dict_fg = combined_loss_fn(heatmap_fg, foreground_masks)
        loss_tu, loss_dict_tu = combined_loss_fn(heatmap_tu, tumor_masks)
        
        # Total loss: Foreground loss + Tumor loss
        loss = 0.4 * loss_fg + 0.6 * loss_tu
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log detailed metrics
        loss_dict = {
            'loss': loss.detach(),
            'loss_fg': loss_fg.detach(),
            'loss_tu': loss_tu.detach(),
            'loss_fg_dice': loss_dict_fg['dice'],
            'loss_fg_focal': loss_dict_fg['focal'],
            'loss_fg_bce': loss_dict_fg['bce'],
            'loss_tu_dice': loss_dict_tu['dice'],
            'loss_tu_focal': loss_dict_tu['focal'],
            'loss_tu_bce': loss_dict_tu['bce'],
        }
        lr_dict = {'lr': optimizer.param_groups[0]['lr']}
        
        metric_logger.update(**loss_dict)
        metric_logger.update(**lr_dict)
    
    # Update learning rate after each epoch
    scheduler.step()
    
    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Training stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, iter_num


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    dice_index_fn: nn.Module
):
    """Evaluate the model."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 50
    
    total_dice_fg = 0.0
    total_dice_tu = 0.0
    total_samples = 0
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = samples.to(device)
        
        foreground_masks = []
        tumor_masks = []
        for target in targets:
            mask = target['mask'].to(device)
            foreground_mask = (mask > 0).float().unsqueeze(0).unsqueeze(0)
            foreground_masks.append(foreground_mask)
            tumor_mask = (mask == 2).float().unsqueeze(0).unsqueeze(0)
            tumor_masks.append(tumor_mask)
        foreground_masks = torch.cat(foreground_masks, dim=0)
        tumor_masks = torch.cat(tumor_masks, dim=0)
        
        heatmap_all = model(images)
        heatmap_fg = heatmap_all[:, 0:1]
        heatmap_tu = heatmap_all[:, 1:2]
        
        dice_fg = dice_index_fn(heatmap_fg, foreground_masks)
        dice_tu = dice_index_fn(heatmap_tu, tumor_masks)
        total_dice_fg += dice_fg.item() * images.size(0)
        total_dice_tu += dice_tu.item() * images.size(0)
        total_samples += images.size(0)
    
    mean_dice_fg = total_dice_fg / total_samples
    mean_dice_tu = total_dice_tu / total_samples
    
    print(f"Validation Dice (FG): {mean_dice_fg:.4f}")
    print(f"Validation Dice (Tumor): {mean_dice_tu:.4f}")
    
    return {'dice': (mean_dice_tu + mean_dice_fg) / 2, 'dice_fg': mean_dice_fg, 'dice_tu': mean_dice_tu}


def parse_args():
    parser = argparse.ArgumentParser('Stage1: Classification Branch Pretraining')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='rectum', help='Dataset name')
    parser.add_argument('--root', type=str, default='./data/rectum', help='Dataset root path')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Early stopping parameters
    parser.add_argument('--early_stop_patience', type=int, default=10, 
                       help='Early stopping patience')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                       default='./exp/adaptive_prompt/stage1',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=202312, help='Random seed')
    
    # Distributed training
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--dist_url', type=str, default='env://', help='Distributed training URL')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Evaluation only')
    
    return parser.parse_args()


def main(args):
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)
    
    # Set random seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    print("Loading dataset...")
    dataset_train = RectumDataloader(
        args.root, 
        mode='train', 
        imgsize=(args.img_size, args.img_size)
    )
    dataset_val = RectumDataloader(
        args.root, 
        mode='test', 
        imgsize=(args.img_size, args.img_size)
    )
    
    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")
    
    # Create data loaders
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train,
        collate_fn=simple_collate_fn,
        num_workers=args.num_workers
    )
    
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=simple_collate_fn,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = ClassificationBranch(pretrained=True, dropout=0.1)
    model.to(device)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=False
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_parameters:,}')
    
    # Create optimizer (AdamW with increased weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=5e-4,
        betas=(0.9, 0.999)
    )
    
    # Create LR scheduler
    def lr_lambda(epoch):
        if epoch < 3:  # 3 epochs warmup
            return (epoch + 1) / 3
        else:
            # Cosine annealing, T_max=30
            progress = min(1.0, (epoch - 3) / 30)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create loss function
    combined_loss_fn = CombinedLoss(dice_weight=0.5, focal_weight=0.3, bce_weight=0.2)
    dice_index_fn = BinaryDiceIndex()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if utils.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume training
    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
    
    # Evaluation mode
    if args.eval:
        print("Evaluation mode")
        test_stats = evaluate(model, data_loader_val, device, dice_index_fn)
        print(f"Validation Dice: {test_stats['dice']:.4f}")
        return
    
    # Start training
    print("Start training...")
    start_time = time.time()
    iter_num = 0
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train
        train_stats, iter_num = train_one_epoch(
            model, data_loader_train, optimizer, scheduler, device, epoch, args,
            combined_loss_fn, iter_num
        )
        
        # Validation
        test_stats = evaluate(model, data_loader_val, device, dice_index_fn)
        val_dice = test_stats['dice']
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            
            if args.output_dir and utils.is_main_process():
                checkpoint_path = output_dir / f'best_dice_{best_dice:.4f}.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'args': args,
                }, checkpoint_path)
                print(f"Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'best_dice': best_dice,
            'patience': patience_counter,
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        print(f"Epoch {epoch}: Val Dice={val_dice:.4f}, Best Dice={best_dice:.4f}, "
              f"Patience={patience_counter}/{args.early_stop_patience}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    print(f'Best validation Dice: {best_dice:.4f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)