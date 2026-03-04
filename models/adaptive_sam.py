"""
U-SAM with Adaptive Prompt Generation

Extends the U-SAM class to integrate the Adaptive Prompt Generator module.

Author: U-SAM Adaptive Prompt Team
Date: 2025-12-14
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.u_sam import SAM, DiceLoss, DiceIndexAB, mIoUAB
from models.adaptive_prompt_generator import AdaptivePromptGenerator
from torch.nn import CrossEntropyLoss
from models.backbone import UNet as downsample


class AdaptiveSAM(SAM):
    """
    U-SAM with Adaptive Prompt Generation
    
    Extends the original SAM class to add adaptive prompt generation capabilities.
    """
    def __init__(self, args):
        # Initialize parent class first
        super(AdaptiveSAM, self).__init__(args)
        
        # Override parent's num_pts to match training arguments
        if hasattr(args, 'num_points') and args.num_points in [1, 3, 5]:
            self.num_pts = args.num_points
        
        # Segmentation loss weight (consistent with baseline: 0.6*Dice + 0.4*CE)
        self.dice_weight = getattr(args, 'dice_weight', 0.6)
        
        # Add adaptive prompt generator module parameters
        self.use_adaptive_prompt = getattr(args, 'use_adaptive_prompt', False)
        self.adaptive_prompt_type = getattr(args, 'adaptive_prompt_type', 'box')  # 'point' or 'box'
        self.use_attention = getattr(args, 'use_attention', True)
        
        # Create Adaptive Prompt Generator module
        if self.use_adaptive_prompt:
            self.adaptive_prompt_generator = AdaptivePromptGenerator(
                use_attention=self.use_attention,
                pretrained_backbone=False,
                prompt_type=self.adaptive_prompt_type,
                num_points=getattr(args, 'num_points', 5),
                point_conf_threshold=getattr(args, 'point_conf_threshold', 0.5),
                point_min_distance=getattr(args, 'point_min_distance', 20),
                box_threshold=getattr(args, 'box_threshold', 0.4),
                box_margin_ratio=getattr(args, 'box_margin_ratio', 0.1),
                gaussian_sigma=getattr(args, 'gaussian_sigma', 2.0),
                nms_window_size=getattr(args, 'nms_window_size', 30)
            )
            print(f"Adaptive Prompt Generator initialized with type='{self.adaptive_prompt_type}', "
                  f"attention={'enabled' if self.use_attention else 'disabled'}")
            self.heatmap_loss_weight = getattr(args, 'heatmap_loss_weight', 0.15)
            self.heatmap_source = getattr(args, 'heatmap_source', 'tumor')
            self.rectum_heatmap_loss_weight = getattr(args, 'rectum_heatmap_loss_weight', 0.05)
    
    def load_classification_pretrained(self, checkpoint_path: str):
        """
        Load pretrained weights for the classification branch.
        
        Args:
            checkpoint_path: Path to the Stage 1 training checkpoint.
        """
        if not self.use_adaptive_prompt:
            print("Warning: Adaptive prompt is not enabled, skipping loading classification weights")
            return
        
        print(f"Loading classification branch weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state_dict for the classification branch
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load into the classification branch
        self.adaptive_prompt_generator.classification_branch.load_state_dict(state_dict, strict=True)
        print("Classification branch weights loaded successfully")
    
    def forward(self, samples, targets=None):
        """
        Forward pass with support for adaptive prompt generation.
        
        Args:
            samples: Input image batch.
            targets: Target annotations (used during training).
        Returns:
            Training: masks, sam_losses, loss_dice, loss_ce
            Evaluation: masks, dice_a, dice_b, iou_a, iou_b
        """
        device = samples.tensors.device
        boxes = pts = None
        
        # Classification branch uses original [0,1] range images (consistent with Stage 1 training)
        input_images = samples.tensors.clone()
        
        # Restore SAM normalization: backbone and image_encoder require normalized input
        pixel_mean = torch.tensor(self.pixel_mean).float().to(device).reshape(1, 3, 1, 1)
        pixel_std = torch.tensor(self.pixel_std).float().to(device).reshape(1, 3, 1, 1)
        samples.tensors = (samples.tensors - pixel_mean) / pixel_std
        
        # ========== Adaptive Prompt Generation Logic ==========
        if self.use_adaptive_prompt:
            heatmap_all = self.adaptive_prompt_generator.classification_branch(input_images)
            heatmap_tumor = heatmap_all[:, 1:2]
            heatmap_foreground = heatmap_all[:, 0:1]
            
            # heatmap_tumor and heatmap_foreground are Sigmoid outputs [0, 1]
            if self.use_attention:
                # Even with the same attention module, ensure it processes single-channel heatmaps
                heatmap_tumor = self.adaptive_prompt_generator.attention_module(heatmap_tumor)
                heatmap_foreground = self.adaptive_prompt_generator.attention_module(heatmap_foreground)
            
            heatmap_tumor = heatmap_tumor.clamp(0.0, 1.0)
            heatmap_foreground = heatmap_foreground.clamp(0.0, 1.0)
            
            if self.adaptive_prompt_type == 'box':
                sel = heatmap_tumor if self.heatmap_source != 'foreground' else heatmap_foreground
                boxes_gen = self.adaptive_prompt_generator.prompt_generator.generate_box_prompts(sel)
                boxes = boxes_gen.to(device)
                pts = None
                heatmap = sel
            elif self.adaptive_prompt_type == 'point':
                sel = heatmap_tumor if self.heatmap_source != 'foreground' else heatmap_foreground
                points_gen, labels_gen = self.adaptive_prompt_generator.prompt_generator.generate_point_prompts(
                    sel, num_points=self.num_pts
                )
                pts = (points_gen.to(device), labels_gen.to(device))
                boxes = None
                heatmap = sel
            elif self.adaptive_prompt_type == 'both':
                if self.heatmap_source == 'mixed':
                    points_gen, labels_gen = self.adaptive_prompt_generator.prompt_generator.generate_point_prompts(
                        heatmap_tumor, num_points=self.num_pts
                    )
                    boxes_gen = self.adaptive_prompt_generator.prompt_generator.generate_box_prompts(heatmap_foreground)
                    pts = (points_gen.to(device), labels_gen.to(device))
                    boxes = boxes_gen.to(device)
                else:
                    sel = heatmap_tumor if self.heatmap_source != 'foreground' else heatmap_foreground
                    points_gen, labels_gen = self.adaptive_prompt_generator.prompt_generator.generate_point_prompts(
                        sel, num_points=self.num_pts
                    )
                    boxes_gen = self.adaptive_prompt_generator.prompt_generator.generate_box_prompts(sel)
                    pts = (points_gen.to(device), labels_gen.to(device))
                    boxes = boxes_gen.to(device)
                heatmap = heatmap_tumor if self.heatmap_source != 'foreground' else heatmap_foreground
        
        # ========== Original Manual Prompt Logic (Compatible) ==========
        else:
            # use groundtruth boxes as prompt
            if self.use_gt_box or (self.use_psd_box and self.training):
                boxes = torch.vstack([targets[i]['orig_boxes'] for i in range(len(targets))])
            
            # use groundtruth points as prompt
            if self.use_gt_pts or (self.use_psd_pts and self.training):
                if self.num_pts == 1:
                    pts_idx = torch.tensor([2]).long()
                elif self.num_pts == 3:
                    pts_idx = torch.tensor([0, 2, 4]).long()
                elif self.num_pts == 5:
                    pts_idx = torch.tensor([0, 1, 2, 3, 4]).long()
                else:
                    pts_idx = slice(None)
                
                pts = torch.vstack([targets[i]['points'][:, :, pts_idx].reshape(1, -1, 2)
                                    for i in range(len(targets))])
                pts_lbs = torch.ones([pts.size(0), pts.size(1)]).long().to(device)
                pts = (pts, pts_lbs)
        
        # ========== SAM Inference Process ==========
        # Note: For adaptive prompts, do NOT wrap prompt_encoder with no_grad!
        # Gradient flow is required: Heatmap -> Prompt Coords -> prompt_encoder -> SAM -> loss
        # This allows the classification branch to learn from SAM's feedback to generate better prompts.
        
        if self.use_adaptive_prompt:
            # Adaptive prompt: requires gradients
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=pts,
                boxes=boxes,
                masks=None,
            )
        else:
            # GT prompt: no gradients needed (prompts are fixed constants)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=pts,
                    boxes=boxes,
                    masks=None,
                )
        
        bt_feature, skip_feature = self.backbone(samples.tensors)
        image_embedding = self.sam.image_encoder(bt_feature)
        masks, low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            skip=skip_feature,
        )
        
        masks = self.sam.postprocess_masks(
            masks=masks,
            input_size=masks.shape[-2:],
            original_size=[self.img_size, self.img_size]
        )
        
        # ========== Loss Calculation and Return ==========
        if self.training:
            # calc_loss is consistent with baseline (u-sam.py), no extra class_weight introduced
            def calc_loss(logits, labels, ce_loss, dice_loss, dice_weight: float = 0.6):
                loss_ce = ce_loss(logits, labels)
                loss_dice = dice_loss(logits, labels, softmax=True)
                loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
                return loss, loss_ce, loss_dice
            
            gt = torch.stack([targets[i]['mask'] for i in range(len(targets))], dim=0)
            sam_losses, loss_ce, loss_dice = calc_loss(masks, gt, self.ce_loss, self.dice_loss, self.dice_weight)
            
            if self.use_adaptive_prompt and hasattr(self, 'adaptive_prompt_generator'):
                if self.adaptive_prompt_type == 'both' and getattr(self, 'heatmap_source', 'tumor') == 'mixed':
                    # Tumor heatmap supervision (Channel 1 -> Tumor)
                    gt_mask_tumor = (gt == 2).float().unsqueeze(1)
                    pt_tumor = torch.where(gt_mask_tumor == 1, heatmap_tumor, 1 - heatmap_tumor)
                    focal_weight_tumor = (1 - pt_tumor) ** 2
                    focal_weight_tumor = focal_weight_tumor.detach()
                    heatmap_tumor = heatmap_tumor.clamp(1e-6, 1 - 1e-6)
                    self.heatmap_focal_tumor = torch.nn.functional.binary_cross_entropy(
                        heatmap_tumor, gt_mask_tumor, weight=focal_weight_tumor, reduction='mean'
                    )
                    
                    # Foreground heatmap supervision (Channel 0 -> Foreground: Rectum + Tumor)
                    # Note: gt > 0 is used here to match Stage 1 targets, ensuring box prompts cover the full area
                    gt_mask_foreground = (gt > 0).float().unsqueeze(1)
                    pt_foreground = torch.where(gt_mask_foreground == 1, heatmap_foreground, 1 - heatmap_foreground)
                    focal_weight_foreground = (1 - pt_foreground) ** 2
                    focal_weight_foreground = focal_weight_foreground.detach()
                    heatmap_foreground = heatmap_foreground.clamp(1e-6, 1 - 1e-6)
                    self.heatmap_focal_foreground = torch.nn.functional.binary_cross_entropy(
                        heatmap_foreground, gt_mask_foreground, weight=focal_weight_foreground, reduction='mean'
                    )
                    
                    sam_losses = sam_losses + self.heatmap_loss_weight * self.heatmap_focal_tumor + self.rectum_heatmap_loss_weight * self.heatmap_focal_foreground
                else:
                    cls_id = 1 if getattr(self, 'heatmap_source', 'tumor') == 'foreground' else 2
                    gt_mask_binary = (gt == cls_id).float().unsqueeze(1)
                    sel = heatmap if 'heatmap' in locals() else heatmap_tumor
                    pt = torch.where(gt_mask_binary == 1, sel, 1 - sel)
                    focal_weight = (1 - pt) ** 2
                    focal_weight = focal_weight.detach()
                    sel = sel.clamp(1e-6, 1 - 1e-6)
                    self.heatmap_focal_tumor = torch.nn.functional.binary_cross_entropy(
                        sel, gt_mask_binary, weight=focal_weight, reduction='mean'
                    )
                    sam_losses = sam_losses + self.heatmap_loss_weight * self.heatmap_focal_tumor
            
            return masks, sam_losses, loss_dice, loss_ce
        else:
            masks = torch.argmax(masks, dim=1, keepdim=False)
            gt = torch.stack([targets[i]['mask'] for i in range(len(targets))], dim=0)
            dice_a, dice_b = self.dice_index(masks, gt)
            iou_a, iou_b = self.iou_index(masks, gt)
            return masks, dice_a, dice_b, iou_a, iou_b
