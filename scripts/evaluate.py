"""
Evaluation Script: Adaptive Prompt Generation Module Evaluation

Features:
1. Calculate segmentation performance metrics (Dice, IoU, NSD)
2. Measure prompt generation time
3. Generate evaluation report
4. Visualize results (heatmaps, prompt points/boxes, segmentation results)

Author: U-SAM Adaptive Prompt Team
Date: 2025-12-14
"""

import os
import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List

from models.adaptive_sam import AdaptiveSAM
from utils import misc as utils


def compute_nsd(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 2.0) -> float:
    """
    Calculate Normalized Surface Distance (NSD)
    
    Args:
        pred_mask: Prediction mask [H, W]
        gt_mask: Ground truth mask [H, W]
        threshold: Distance threshold (pixels)
    Returns:
        nsd: NSD value
    """
    from scipy.ndimage import distance_transform_edt
    
    # Extract boundaries
    pred_boundary = np.logical_xor(pred_mask, 
                                   np.pad(pred_mask, 1, mode='constant')[1:-1, 1:-1])
    gt_boundary = np.logical_xor(gt_mask,
                                 np.pad(gt_mask, 1, mode='constant')[1:-1, 1:-1])
    
    if not (pred_boundary.any() and gt_boundary.any()):
        return 0.0
    
    # Calculate distance transform
    dt_pred = distance_transform_edt(~pred_boundary)
    dt_gt = distance_transform_edt(~gt_boundary)
    
    # Calculate surface distances
    pred_distances = dt_gt[pred_boundary]
    gt_distances = dt_pred[gt_boundary]
    
    # Calculate NSD
    pred_nsd = (pred_distances <= threshold).sum() / len(pred_distances)
    gt_nsd = (gt_distances <= threshold).sum() / len(gt_distances)
    
    nsd = (pred_nsd + gt_nsd) / 2.0
    return nsd


@torch.no_grad()
def evaluate_with_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    visualize: bool = False,
    save_dir: Path = None
) -> Dict:
    """
    Full evaluation including Dice, IoU, NSD metrics
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_dice_a, all_dice_b = [], []
    all_iou_a, all_iou_b = [], []
    all_nsd_scores = []
    all_prompt_times = []
    
    if visualize and save_dir:
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    print("Evaluating...")
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Measure prompt generation time (if using adaptive prompt)
        if args.use_adaptive_prompt:
            start_time = time.time()
        
        # Forward pass
        masks, dice_a, dice_b, iou_a, iou_b = model(samples, targets)
        
        if args.use_adaptive_prompt:
            prompt_time = time.time() - start_time
            all_prompt_times.append(prompt_time)
        
        # Collect Dice and IoU
        all_dice_a.append(dice_a)
        all_dice_b.append(dice_b)
        all_iou_a.append(iou_a)
        all_iou_b.append(iou_b)
        
        # Calculate NSD
        for b in range(len(targets)):
            pred_np = masks[b].cpu().numpy()
            gt_np = targets[b]['mask'].cpu().numpy()
            
            # Calculate NSD for each class
            nsd_per_class = []
            for cls in range(1, args.sam_num_classes):  # Ignore background
                pred_cls = (pred_np == cls).astype(np.uint8)
                gt_cls = (gt_np == cls).astype(np.uint8)
                
                if gt_cls.sum() > 0:  # Only calculate for existing classes
                    nsd = compute_nsd(pred_cls, gt_cls, threshold=2.0)
                    nsd_per_class.append(nsd)
            
            if nsd_per_class:
                all_nsd_scores.append(np.mean(nsd_per_class))
        
        # Visualization
        if visualize and idx < 20 and save_dir:  # Visualize only first 20 samples
            visualize_sample(
                samples[0], 
                targets[0], 
                masks[0],
                idx,
                vis_dir,
                args
            )
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(data_loader)} batches")
    
    # Aggregate metrics
    all_dice_a = torch.cat(all_dice_a, dim=0).sum(dim=0)
    all_dice_b = torch.cat(all_dice_b, dim=0).sum(dim=0)
    all_dice = all_dice_a / all_dice_b
    
    all_iou_a = torch.cat(all_iou_a, dim=0).sum(dim=0)
    all_iou_b = torch.cat(all_iou_b, dim=0).sum(dim=0)
    all_iou = all_iou_a / all_iou_b
    
    # Build metrics dictionary
    metrics = {
        'mean_dice': all_dice[1:].mean().item(),  # Ignore background
        'miou': all_iou[1:].mean().item(),
        'mean_nsd': np.mean(all_nsd_scores) if all_nsd_scores else 0.0,
    }
    
    # Per-class metrics
    for i in range(1, args.sam_num_classes):
        metrics[f'class{i}_dice'] = all_dice[i].item()
        metrics[f'class{i}_iou'] = all_iou[i].item()
    
    # Prompt generation time
    if all_prompt_times:
        metrics['prompt_time_mean'] = np.mean(all_prompt_times)
        metrics['prompt_time_std'] = np.std(all_prompt_times)
    
    return metrics


def visualize_sample(
    image: torch.Tensor,
    target: Dict,
    pred_mask: torch.Tensor,
    idx: int,
    save_dir: Path,
    args: argparse.Namespace
):
    """
    Visualize a single sample
    
    Args:
        image: Input image [3, H, W]
        target: Target annotations
        pred_mask: Prediction mask [H, W]
        idx: Sample index
        save_dir: Directory to save visualization
        args: Arguments
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show input image
    img_np = image[0].cpu().numpy()  # Take first channel
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Show Ground Truth
    gt_mask = target['mask'].cpu().numpy()
    axes[1].imshow(gt_mask, cmap='jet', vmin=0, vmax=args.sam_num_classes-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Show Prediction
    pred_np = pred_mask.cpu().numpy()
    axes[2].imshow(pred_np, cmap='jet', vmin=0, vmax=args.sam_num_classes-1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(
    metrics: Dict,
    save_path: Path,
    args: argparse.Namespace
):
    """
    Generate evaluation report
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save report
        args: Arguments
    """
    report = []
    report.append("=" * 80)
    report.append("Adaptive Prompt Generation Module Evaluation Report")
    report.append("=" * 80)
    report.append("")
    
    # Model Configuration
    report.append("Model Configuration:")
    report.append(f"  - Use Adaptive Prompt: {args.use_adaptive_prompt}")
    if args.use_adaptive_prompt:
        report.append(f"  - Prompt Type: {args.adaptive_prompt_type}")
        report.append(f"  - Attention Mechanism: {args.use_attention}")
    report.append(f"  - Dataset: {args.dataset}")
    report.append("")
    
    # Segmentation Performance Metrics
    report.append("Segmentation Performance Metrics:")
    report.append(f"  - Mean Dice: {metrics['mean_dice']:.4f}")
    report.append(f"  - Mean IoU: {metrics['miou']:.4f}")
    report.append(f"  - Mean NSD: {metrics['mean_nsd']:.4f}")
    report.append("")
    
    # Per-class Metrics
    report.append("Per-class Metrics:")
    for i in range(1, args.sam_num_classes):
        dice = metrics.get(f'class{i}_dice', 0.0)
        iou = metrics.get(f'class{i}_iou', 0.0)
        report.append(f"  - Class {i}: Dice={dice:.4f}, IoU={iou:.4f}")
    report.append("")
    
    # Efficiency Metrics
    if 'prompt_time_mean' in metrics:
        report.append("Efficiency Metrics:")
        report.append(f"  - Prompt Generation Time (Mean): {metrics['prompt_time_mean']:.4f} s")
        report.append(f"  - Prompt Generation Time (Std): {metrics['prompt_time_std']:.4f} s")
        report.append("")
    
    report.append("=" * 80)
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Print to console
    for line in report:
        print(line)


def parse_args():
    parser = argparse.ArgumentParser('Evaluation Script for Adaptive Prompt Generation')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--use_adaptive_prompt', action='store_true',
                       help='Use adaptive prompt generation')
    parser.add_argument('--adaptive_prompt_type', type=str, default='box',
                       choices=['point', 'box', 'both'])
    parser.add_argument('--use_attention', action='store_true')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='rectum',
                       choices=['rectum', 'word'])
    parser.add_argument('--root', type=str, default='./data/rectum')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # Evaluation parameters
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--output_dir', type=str, 
                       default='./exp/adaptive_prompt/evaluation',
                       help='Output directory')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=202312)
    
    # Distributed (compatibility)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str, default='env://')
    
    return parser.parse_args()


def main(args):
    # Set prompt_mode to 0 (compatibility)
    args.prompt_mode = 0
    args.use_gt_box = False
    args.use_gt_pts = False
    args.use_psd_box = False
    args.use_psd_pts = False
    args.use_psd_mask = False
    
    utils.init_distributed_mode(args)
    
    print(args)
    
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    if args.dataset == 'rectum':
        args.sam_num_classes = 3
        from dataset.rectum_dataloader import RectumDataloader
        dataset_val = RectumDataloader(args.root, mode='test',
                                      imgsize=(args.img_size, args.img_size))
    elif args.dataset == 'word':
        args.sam_num_classes = 17
        from dataset.word_dataloader import WordDataset
        dataset_val = WordDataset(args.root, mode='test',
                                 imgsize=(args.img_size, args.img_size))
    
    print(f"Evaluation samples: {len(dataset_val)}")
    
    # Create data loader
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn
    )
    
    # Load model
    print("Loading model...")
    model = AdaptiveSAM(args)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Set pixel normalization parameters
    model.pixel_mean = (0.1364736, 0.1364736, 0.1364736)
    model.pixel_std = (0.23238614, 0.23238614, 0.23238614)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation
    print("Starting evaluation...")
    metrics = evaluate_with_metrics(
        model, 
        data_loader_val, 
        device, 
        args,
        visualize=args.visualize,
        save_dir=output_dir
    )
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Generate report
    report_path = output_dir / 'evaluation_report.txt'
    generate_report(metrics, report_path, args)
    print(f"Report saved to: {report_path}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
