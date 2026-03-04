"""
Stage 2 Training Script: End-to-End Fine-tuning

Objective: Jointly optimize the classification branch and U-SAM backbone.
Training Strategy:
  - First N epochs: Freeze classification branch, train only SAM backbone (warmup).
  - Remaining epochs: Joint training, with classification branch LR reduced (e.g., 10x smaller).
Training Configuration:
  - Batch Size: 24
  - Learning Rates: SAM(1e-4), ViT(1e-4), Backbone(1e-4), Classification(1e-5)
  - Epochs: 100
  - Loss Function: 0.6*Dice + 0.4*CE + Auxiliary Heatmap Loss

Author: U-SAM Adaptive Prompt Team
Date: 2025-12-14
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import time
import datetime
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import importlib.util

# Dynamically import u-sam.py
usam_path = project_root / 'u-sam.py'
spec = importlib.util.spec_from_file_location('usam_module', usam_path)
usam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usam_module)

train_one_epoch = usam_module.train_one_epoch
evaluate = usam_module.evaluate

from models.adaptive_sam import AdaptiveSAM
import utils.misc as utils


def train_one_epoch_adaptive(args, model, data_loader, optimizer, device, epoch, iter_num=0):
    """
    Wrapper for original train_one_epoch, adding classification branch LR scheduling.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 2000 // (args.batch_size * utils.get_world_size())

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        bs = len(targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs, losses, loss_dice, loss_ce = model(samples, targets)

        # Extract heatmap losses for logging
        model_without_ddp = model.module if hasattr(model, 'module') else model
        loss_dict = {
            'loss': losses.clone().detach(),
            'loss_dice': loss_dice.clone().detach(),
            'loss_ce': loss_ce.clone().detach(),
        }
        
        if hasattr(model_without_ddp, 'heatmap_focal_tumor'):
            loss_dict['loss_hm_tu'] = model_without_ddp.heatmap_focal_tumor.detach()
        if hasattr(model_without_ddp, 'heatmap_focal_foreground'):
            loss_dict['loss_hm_fg'] = model_without_ddp.heatmap_focal_foreground.detach()

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm and args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        if args.lr_schedule == 'cosine':
            warmup_iters = int(args.max_iter * args.warmup_ratio) if args.warmup else 0
            if args.warmup and iter_num < warmup_iters:
                factor = float(iter_num) / max(1, warmup_iters)
            else:
                if args.warmup:
                    progress = float(iter_num - warmup_iters) / max(1, args.max_iter - warmup_iters)
                else:
                    progress = float(iter_num) / max(1, args.max_iter)
                progress = max(0.0, min(1.0, progress))
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            if args.warmup:
                warmup = args.max_iter // args.epochs // 2
                factor = (1.0 - (iter_num - warmup) / args.max_iter) ** 0.9
                factor = min(factor, iter_num / warmup)
            else:
                factor = (1.0 - iter_num / args.max_iter) ** 0.9
        
        optimizer.param_groups[0]['lr'] = args.lr * factor
        optimizer.param_groups[1]['lr'] = args.lr_vit * factor
        optimizer.param_groups[2]['lr'] = args.lr_backbone * factor
        
        # Update classification branch LR (if using adaptive prompt)
        if args.use_adaptive_prompt and len(optimizer.param_groups) > 3:
            optimizer.param_groups[3]['lr'] = args.lr_classification * factor

        # Update stats
        iter_num += bs * utils.get_world_size()

        # Log all training metrics
        lr_dict = {
            'lr': optimizer.param_groups[0]["lr"],
        }
        metric_logger.update(**loss_dict)
        metric_logger.update(**lr_dict)
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Training stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, iter_num


def parse_args():
    parser = argparse.ArgumentParser('Stage2: End-to-End Fine-tuning with Adaptive Prompts')
    
    # Adaptive Prompt Parameters
    parser.add_argument('--use_adaptive_prompt', action='store_true',
                       help='Use adaptive prompt generation')
    parser.add_argument('--adaptive_prompt_type', type=str, default='box',
                       choices=['point', 'box', 'both'],
                       help='Type of adaptive prompt')
    parser.add_argument('--use_attention', action='store_true',
                       help='Use attention mechanism in prompt generator')
    parser.add_argument('--classification_pretrained', type=str, default='',
                       help='Path to stage1 pretrained classification weights')
    parser.add_argument('--freeze_classification_epochs', type=int, default=3,
                       help='Number of epochs to freeze classification branch (warmup SAM first)')
    
    # SAM Parameters
    parser.add_argument('--prompt_mode', default=0, type=int, choices=[0, 1, 2, 3],
                       help="0=no prompt(adaptive only), 1=gt boxes, 2=gt points, 3=gt boxes&points")
    parser.add_argument('--warmup', action='store_true', help='Warmup at the beginning')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate for SAM')
    parser.add_argument('--lr_vit', default=5e-4, type=float, help='Learning rate for ViT')
    parser.add_argument('--lr_backbone', default=5e-4, type=float, help='Learning rate for backbone')
    parser.add_argument('--lr_classification', default=1e-5, type=float,
                       help='Learning rate for classification branch')
    parser.add_argument('--dice_weight', default=0.6, type=float)
    parser.add_argument('--heatmap_loss_weight', type=float, default=0.15,
                       help='Tumor heatmap auxiliary loss weight (prevents APG feature collapse)')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['poly', 'cosine'])
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--heatmap_source', type=str, default='mixed',
                       choices=['tumor', 'foreground', 'mixed'],
                       help='mixed: points from tumor heatmap, boxes from foreground heatmap')
    parser.add_argument('--rectum_heatmap_loss_weight', type=float, default=0.15,
                       help='Foreground heatmap auxiliary loss weight')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                       help='Gradient clipping max norm')
    
    # Prompt Generator Parameters
    parser.add_argument('--point_conf_threshold', type=float, default=0.5)
    parser.add_argument('--point_min_distance', type=int, default=20)
    parser.add_argument('--box_threshold', type=float, default=0.3)
    parser.add_argument('--box_margin_ratio', type=float, default=0.15)
    parser.add_argument('--num_points', type=int, default=5)
    parser.add_argument('--gaussian_sigma', type=float, default=2.0)
    parser.add_argument('--nms_window_size', type=int, default=30)
    
    # Dataset Parameters
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--dataset', type=str, choices=['rectum', 'word'], default='rectum')
    parser.add_argument('--root', type=str, default='', help='Dataset root path')
    
    # Runtime Configuration
    parser.add_argument('--output_dir', default='./exp/adaptive_prompt/stage2',
                       help='Path where to save')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', default=202312, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--resume_ignore_optimizer', action='store_true',
                       help='Ignore optimizer state when resuming to avoid param group or LR mismatch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                       help='Start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    
    # Distributed Training Parameters
    parser.add_argument('--world_size', default=1, type=int,
                       help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL for distributed training')
    
    return parser.parse_args()


def freeze_module(module):
    """Freeze module parameters."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze module parameters."""
    for param in module.parameters():
        param.requires_grad = True


def main(args):
    utils.init_distributed_mode(args)
    
    # Set prompt mode (compatible with original U-SAM)
    if not args.use_adaptive_prompt:
        # Use ground truth prompts
        if args.prompt_mode == 0:
            args.use_gt_box = False
            args.use_gt_pts = False
            args.use_psd_box = False
            args.use_psd_pts = False
            args.use_psd_mask = False
            args.use_text = False
            prompt = 'no_prompt'
        elif args.prompt_mode == 1:
            args.use_gt_box = True
            args.use_gt_pts = False
            args.use_psd_box = False
            args.use_psd_pts = False
            args.use_psd_mask = False
            args.use_text = False
            prompt = 'gt_boxes'
        elif args.prompt_mode == 2:
            args.use_gt_box = False
            args.use_gt_pts = True
            args.use_psd_box = False
            args.use_psd_pts = False
            args.use_psd_mask = False
            args.use_text = False
            prompt = 'gt_pts'
        elif args.prompt_mode == 3:
            args.use_gt_box = True
            args.use_gt_pts = True
            args.use_psd_box = False
            args.use_psd_pts = False
            args.use_psd_mask = False
            args.use_text = False
            prompt = 'gt_boxes_pts'
    else:
        # Use adaptive prompts
        args.use_gt_box = False
        args.use_gt_pts = False
        args.use_psd_box = False
        args.use_psd_pts = False
        args.use_psd_mask = False
        args.use_text = False
        attention_str = 'attn' if args.use_attention else 'noattn'
        prompt = f'adaptive_{args.adaptive_prompt_type}_{attention_str}'
    
    print(args)
    
    device = torch.device(args.device)
    
    # Fix random seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build dataset
    if args.dataset == 'rectum':
        args.sam_num_classes = 3
        if not args.root:
            args.root = './data/rectum'
        from dataset.rectum_dataloader import RectumDataloader
        dataset_train = RectumDataloader(args.root, mode='train', 
                                        imgsize=(args.img_size, args.img_size))
        dataset_val = RectumDataloader(args.root, mode='test', 
                                      imgsize=(args.img_size, args.img_size))
    elif args.dataset == 'word':
        args.sam_num_classes = 17
        if not args.root:
            args.root = './data/word'
        from dataset.word_dataloader import WordDataset
        dataset_train = WordDataset(args.root, mode='train', 
                                   imgsize=(args.img_size, args.img_size))
        dataset_val = WordDataset(args.root, mode='test', 
                                 imgsize=(args.img_size, args.img_size))
    
    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")
    
    # Build model
    print("Building model...")
    model = AdaptiveSAM(args)
    
    # Load stage 1 pre-trained classification weights
    if args.use_adaptive_prompt and args.classification_pretrained:
        model.load_classification_pretrained(args.classification_pretrained)
    
    # Evaluation mode: Load complete model
    if args.eval:
        print('Loading checkpoint for evaluation')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters:,}')
    
    # Build optimizer (multi-learning rate)
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                      if "image_encoder" not in n and "backbone" not in n 
                      and "adaptive_prompt_generator" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                      if "image_encoder" in n and p.requires_grad],
            "lr": args.lr_vit,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    # Add classification branch parameters (if using adaptive prompt)
    if args.use_adaptive_prompt:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters()
                      if "adaptive_prompt_generator" in n and p.requires_grad],
            "lr": args.lr_classification,
        })
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                 weight_decay=args.weight_decay)
    
    # Build data loader
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
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    
    data_loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size,
        sampler=sampler_val, drop_last=False, 
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )
    
    output_dir = Path(args.output_dir) / f'prompt={prompt}'
    
    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    model_without_ddp.pixel_mean = pixel_mean = (0.1364736, 0.1364736, 0.1364736)
    model_without_ddp.pixel_std = pixel_std = (0.23238614, 0.23238614, 0.23238614)
    print('mean: {}, std: {}'.format(pixel_mean, pixel_std))
    
    # Evaluation only
    if args.eval:
        print("Start evaluation")
        test_stats = evaluate(model, data_loader_val, device, visual=True)
        mean_dice = test_stats['mean_dice']
        miou = test_stats['miou']
        print('mean_dice: %.6f, miou: %.6f\n' % (mean_dice, miou))
        return
    
    # Resume training
    iter_num = 0
    best_dice = -1
    args.max_iter = args.epochs * len(dataset_train)
    
    if args.resume:
        print("Resume from checkpoint", args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if (not args.resume_ignore_optimizer) and ('optimizer' in checkpoint) and ('epoch' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            iter_num = args.start_epoch * len(dataset_train)
        else:
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                iter_num = args.start_epoch * len(dataset_train)
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
    
    # Start training
    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Branch training strategy: Freeze classification branch for first N epochs
        if args.use_adaptive_prompt:
            if epoch < args.freeze_classification_epochs:
                if epoch == 0:
                    print(f"Freezing classification branch for first {args.freeze_classification_epochs} epochs")
                    freeze_module(model_without_ddp.adaptive_prompt_generator)
            elif epoch == args.freeze_classification_epochs:
                print(f"Unfreezing classification branch at epoch {epoch}")
                unfreeze_module(model_without_ddp.adaptive_prompt_generator)
        
        # Train one epoch
        train_stats, iter_num = train_one_epoch_adaptive(
            args, model, data_loader_train, optimizer, device, epoch, iter_num
        )
        
        # Validation
        test_stats = evaluate(model, data_loader_val, device)
        mean_dice = test_stats['mean_dice']
        miou = test_stats['miou']
        print('mean_dice: %.6f, miou: %.6f\n' % (mean_dice, miou))
        
        # Save best model
        checkpoint_paths = []
        if mean_dice > best_dice:
            best_dice = mean_dice
            if args.output_dir:
                checkpoint_paths.append(output_dir / f'best_{mean_dice:.6f}_{miou:.6f}.pth')
        
        if args.output_dir:
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_dice': best_dice,
                    'args': args,
                }, checkpoint_path)
        
        # Logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best Dice: {:.6f}'.format(best_dice))


if __name__ == '__main__':
    args = parse_args()
    main(args)