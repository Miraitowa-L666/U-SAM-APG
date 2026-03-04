"""
Adaptive Prompt Generator (APG) Module

This module implements automated prompt generation, including:
1. Classification Branch: ResNet18 + Upsampling layers to generate lesion heatmaps.
2. Attention Mechanism: Non-local Attention to refine heatmaps.
3. Prompt Generation Strategy: Algorithms for generating point and box prompts.

Author: U-SAM Adaptive Prompt Team
Date: 2025-12-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label

class DifferentiablePointExtractor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heatmap: torch.Tensor, points_np: np.ndarray):
        device = heatmap.device
        H, W = heatmap.shape
        points = torch.from_numpy(points_np[:, :2]).float().to(device)
        ctx.save_for_backward(heatmap)
        ctx.points_indices = [(int(x), int(y)) for x, y in points_np[:, :2]]
        return points

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        heatmap = ctx.saved_tensors[0]
        H, W = heatmap.shape
        grad_heatmap = torch.zeros_like(heatmap)
        for i, (x, y) in enumerate(ctx.points_indices):
            g = grad_output[i].norm()
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        w = torch.exp(torch.tensor(-(dx * dx + dy * dy) / (2 * 1.5 * 1.5), device=grad_heatmap.device))
                        grad_heatmap[ny, nx] += g * w
        return grad_heatmap, None

class DifferentiableBoxExtractor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, heatmap: torch.Tensor, box_np: np.ndarray):
        device = heatmap.device
        box = torch.from_numpy(box_np).float().to(device)
        ctx.save_for_backward(heatmap)
        x0, y0, x1, y1 = [int(v) for v in box_np.tolist()]
        ctx.box = (x0, y0, x1, y1)
        return box

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        heatmap = ctx.saved_tensors[0]
        H, W = heatmap.shape
        grad_heatmap = torch.zeros_like(heatmap)
        x0, y0, x1, y1 = ctx.box
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
        g = grad_output.norm()
        grad_heatmap[y0:y1 + 1, x0:x1 + 1] += g / area
        return grad_heatmap, None

class NonLocalAttention(nn.Module):
    """
    Non-local Attention Module
    Dynamically refines heatmap features to suppress noise and enhance lesion regions.
    """
    def __init__(self, in_channels: int, inter_channels: Optional[int] = None):
        super(NonLocalAttention, self).__init__()
        
        self.in_channels = in_channels
        # Ensure inter_channels is at least 1
        if inter_channels is None:
            self.inter_channels = max(1, in_channels // 2)
        else:
            self.inter_channels = max(1, inter_channels)
        
        # Query, Key, Value transformations
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # Output transformation
        self.output_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
        # Scale factor for residual connection
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Enhanced feature map [B, C, H, W]
        """
        batch_size, C, H, W = x.size()
        
        # Memory optimization: Downsample to 1/4 resolution to avoid OOM
        # 224x224 -> 56x56
        if H > 64 or W > 64:
            downsample_ratio = 4
            x_down = F.interpolate(x, scale_factor=1/downsample_ratio, mode='bilinear', align_corners=False)
            H_down, W_down = x_down.size(2), x_down.size(3)
        else:
            x_down = x
            H_down, W_down = H, W
            downsample_ratio = 1
        
        # Generate Query, Key, Value (on downsampled features)
        query = self.query_conv(x_down).view(batch_size, self.inter_channels, -1)
        key = self.key_conv(x_down).view(batch_size, self.inter_channels, -1)
        value = self.value_conv(x_down).view(batch_size, self.inter_channels, -1)
        
        # Compute attention weights
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention weights
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.inter_channels, H_down, W_down)
        
        # Upsample back to original resolution
        if downsample_ratio > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Output transformation
        out = self.output_conv(out)
        
        # Residual connection
        out = self.gamma * out + x
        
        return out


class ClassificationBranch(nn.Module):
    """
    Classification Branch Network (Improved: U-Net style decoder)
    Uses ResNet18 as backbone, adds U-Net style decoder with skip connections.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super(ClassificationBranch, self).__init__()
        
        self.dropout = dropout
        
        # Load ResNet18 backbone
        resnet18 = models.resnet18(pretrained=pretrained)
        
        # Extract feature extraction layers
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        
        self.layer1 = resnet18.layer1  # 64x56x56
        self.layer2 = resnet18.layer2  # 128x28x28
        self.layer3 = resnet18.layer3  # 256x14x14
        self.layer4 = resnet18.layer4  # 512x7x7
        
        # U-Net style decoder with skip connections
        # Decoder Layer 1: 512+256 -> 256
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = self._make_conv_block(256 + 256, 256, dropout)
        
        # Decoder Layer 2: 256+128 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self._make_conv_block(128 + 128, 128, dropout)
        
        # Decoder Layer 3: 128+64 -> 64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self._make_conv_block(64 + 64, 64, dropout)
        
        # Decoder Layer 4: 64 -> 32
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = self._make_conv_block(32, 32, dropout)
        
        # Decoder Layer 5: 32 -> 16
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        
        # Output Layer: 2 channels (Foreground + Tumor)
        self.output_conv = nn.Conv2d(16, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_conv_block(self, in_channels, out_channels, dropout):
        """Create conv block: two 3x3 convs + BN + ReLU + Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 3, 224, 224]
        Returns:
            Heatmap [B, 2, 224, 224], range [0,1]
        """
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        feat1 = self.layer1(x)  # [B, 64, 56, 56]
        feat2 = self.layer2(feat1)  # [B, 128, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 256, 14, 14]
        feat4 = self.layer4(feat3)  # [B, 512, 7, 7]
        
        # Decoder + Skip Connections
        x = self.up1(feat4)
        x = self.conv_up1(torch.cat([x, feat3], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, feat2], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, feat1], dim=1))
        
        x = self.conv_up4(self.up4(x))
        x = self.up5(x)
        
        heatmap = self.sigmoid(self.output_conv(x))
        return heatmap


class PromptGenerator:
    """
    Prompt Generator
    Generates point or box prompts based on heatmaps.
    """
    def __init__(
        self,
        point_conf_threshold: float = 0.5,
        point_min_distance: int = 20,
        box_threshold: float = 0.4,
        box_margin_ratio: float = 0.1,
        use_gaussian_filter: bool = True,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 2.0,
        nms_window_size: int = 30
    ):
        """
        Args:
            point_conf_threshold: Minimum confidence threshold for point prompts
            point_min_distance: Minimum distance between points (pixels)
            box_threshold: Binarization threshold for box prompts
            box_margin_ratio: Margin ratio for box expansion
            use_gaussian_filter: Whether to use Gaussian filtering
            gaussian_kernel_size: Gaussian kernel size
            gaussian_sigma: Gaussian standard deviation
            nms_window_size: NMS window size
        """
        self.point_conf_threshold = point_conf_threshold
        self.point_min_distance = point_min_distance
        self.box_threshold = box_threshold
        self.box_margin_ratio = box_margin_ratio
        self.use_gaussian_filter = use_gaussian_filter
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.nms_window_size = nms_window_size
        
    def generate_point_prompts(
        self, 
        heatmap: torch.Tensor, 
        num_points: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate point prompts.
        
        Args:
            heatmap: Heatmap [B, 1, H, W]
            num_points: Number of points to generate
        Returns:
            points: Point coordinates [B, num_points, 2] (x, y)
            labels: Point labels [B, num_points] (all 1s)
        """
        batch_size = heatmap.size(0)
        H, W = heatmap.shape[2], heatmap.shape[3]
        
        points_list = []
        labels_list = []
        
        for b in range(batch_size):
            heatmap_single = heatmap[b, 0]
            heatmap_np = heatmap_single.detach().cpu().numpy()
            
            # Apply Gaussian filter
            if self.use_gaussian_filter:
                heatmap_np = gaussian_filter(heatmap_np, sigma=self.gaussian_sigma)
            
            # Apply NMS
            peaks = self._non_maximum_suppression(heatmap_np, self.nms_window_size)
            
            # Get valid peaks
            valid_peaks = peaks[peaks[:, 2] >= self.point_conf_threshold]
            
            if len(valid_peaks) == 0:
                # Fallback: Use center point
                center_points = np.array([[W // 2, H // 2, 1.0]] * num_points)
                valid_peaks = center_points
            
            # Sort by confidence
            valid_peaks = valid_peaks[np.argsort(-valid_peaks[:, 2])]
            
            # Apply distance constraint
            selected_points = self._apply_distance_constraint(
                valid_peaks[:, :2], 
                self.point_min_distance
            )
            
            # Pad to num_points
            if len(selected_points) < num_points:
                last_point = selected_points[-1] if len(selected_points) > 0 else [W // 2, H // 2]
                while len(selected_points) < num_points:
                    selected_points = np.vstack([selected_points, last_point])
            else:
                selected_points = selected_points[:num_points]
            
            points_with_grad = DifferentiablePointExtractor.apply(
                heatmap_single, selected_points.astype(np.float32)
            )
            points_list.append(points_with_grad)
            labels_list.append(torch.ones(num_points, dtype=torch.long, device=heatmap.device))
        
        points = torch.stack(points_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        
        return points, labels
    
    def generate_box_prompts(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Generate box prompts.
        
        Args:
            heatmap: Heatmap [B, 1, H, W]
        Returns:
            boxes: Bounding boxes [B, 4] (x_min, y_min, x_max, y_max)
        """
        batch_size = heatmap.size(0)
        H, W = heatmap.shape[2], heatmap.shape[3]
        
        boxes_list = []
        
        for b in range(batch_size):
            heatmap_single = heatmap[b, 0]
            heatmap_np = heatmap_single.detach().cpu().numpy()
            
            # Binarize
            binary_mask = (heatmap_np >= self.box_threshold).astype(np.uint8)
            
            # Fallback if no valid region
            if binary_mask.sum() == 0:
                fallback_thresh = max(0.001, float(heatmap_np.max()) * 0.5)
                binary_mask = (heatmap_np >= fallback_thresh).astype(np.uint8)
            
            if binary_mask.sum() == 0:
                box_np = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
                box_with_grad = DifferentiableBoxExtractor.apply(heatmap_single, box_np)
                boxes_list.append(box_with_grad)
                continue
            
            # Connected components
            labels, num_labels = scipy_label(binary_mask)
            
            if num_labels == 0:
                box = torch.tensor([0, 0, W, H], dtype=torch.float32)
                boxes_list.append(box)
                continue
            
            # Select largest component
            largest_component = 1
            max_area = 0
            for i in range(1, num_labels + 1):
                area = (labels == i).sum()
                if area > max_area:
                    max_area = area
                    largest_component = i
            
            # Compute bounding box
            component_mask = (labels == largest_component).astype(np.uint8)
            y_indices, x_indices = np.where(component_mask > 0)
            
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Expand margin
            width = x_max - x_min
            height = y_max - y_min
            
            x_min = max(0, x_min - int(width * self.box_margin_ratio))
            x_max = min(W, x_max + int(width * self.box_margin_ratio))
            y_min = max(0, y_min - int(height * self.box_margin_ratio))
            y_max = min(H, y_max + int(height * self.box_margin_ratio))
            
            # Minimum size constraint
            min_box_size = 20
            if (x_max - x_min) < min_box_size:
                center_x = (x_min + x_max) // 2
                x_min = max(0, center_x - min_box_size // 2)
                x_max = min(W, center_x + min_box_size // 2)
            if (y_max - y_min) < min_box_size:
                center_y = (y_min + y_max) // 2
                y_min = max(0, center_y - min_box_size // 2)
                y_max = min(H, center_y + min_box_size // 2)
            
            box_np = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            box_with_grad = DifferentiableBoxExtractor.apply(heatmap_single, box_np)
            boxes_list.append(box_with_grad)
        
        boxes = torch.stack(boxes_list, dim=0)
        
        return boxes
    
    def _non_maximum_suppression(
        self, 
        heatmap: np.ndarray, 
        window_size: int
    ) -> np.ndarray:
        """
        Non-Maximum Suppression (Optimized).
        
        Args:
            heatmap: Heatmap [H, W]
            window_size: NMS window size
        Returns:
            peaks: [N, 3] (x, y, confidence)
        """
        from scipy.ndimage import maximum_filter
        
        H, W = heatmap.shape
        
        local_max = maximum_filter(heatmap, size=window_size, mode='constant')
        peaks_mask = (heatmap == local_max) & (heatmap > 0)
        y_indices, x_indices = np.where(peaks_mask)
        
        if len(x_indices) == 0:
            return np.array([[W // 2, H // 2, 0.0]])
        
        peaks = np.stack([
            x_indices,
            y_indices,
            heatmap[y_indices, x_indices]
        ], axis=1)
        
        return peaks
    
    def _apply_distance_constraint(
        self, 
        points: np.ndarray, 
        min_distance: int
    ) -> np.ndarray:
        """
        Apply distance constraint.
        """
        if len(points) <= 1:
            return points
        
        selected = [points[0]]
        
        for point in points[1:]:
            distances = np.sqrt(
                np.sum((np.array(selected) - point) ** 2, axis=1)
            )
            if np.all(distances >= min_distance):
                selected.append(point)
        
        return np.array(selected)


class AdaptivePromptGenerator(nn.Module):
    """
    Adaptive Prompt Generator (Full Module)
    Integrates classification branch, attention mechanism, and prompt generator.
    """
    def __init__(
        self,
        use_attention: bool = True,
        pretrained_backbone: bool = True,
        prompt_type: str = 'box',
        num_points: int = 3,
        **prompt_gen_kwargs
    ):
        """
        Args:
            use_attention: Whether to use attention mechanism
            pretrained_backbone: Whether to use pretrained ResNet18
            prompt_type: 'point' or 'box'
            num_points: Number of point prompts
            **prompt_gen_kwargs: Extra args for PromptGenerator
        """
        super(AdaptivePromptGenerator, self).__init__()
        
        self.use_attention = use_attention
        self.prompt_type = prompt_type
        self.num_points = num_points
        
        self.classification_branch = ClassificationBranch(pretrained=pretrained_backbone)
        
        if self.use_attention:
            self.attention_module = NonLocalAttention(in_channels=1)
        
        self.prompt_generator = PromptGenerator(**prompt_gen_kwargs)
        
    def forward(
        self, 
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, 3, 224, 224]
        Returns:
            prompts, labels/None, heatmap
        """
        heatmap_all = self.classification_branch(images)
        heatmap = heatmap_all[:, 1:2]
        if self.use_attention:
            heatmap = self.attention_module(heatmap)
        
        if self.prompt_type == 'point':
            points, labels = self.prompt_generator.generate_point_prompts(
                heatmap, 
                num_points=self.num_points
            )
            return points, labels, heatmap
        elif self.prompt_type == 'box':
            boxes = self.prompt_generator.generate_box_prompts(heatmap)
            return boxes, None, heatmap
        else:
            raise ValueError(f"Invalid prompt_type: {self.prompt_type}")
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # Test code
    print("=" * 50)
    print("Adaptive Prompt Generator Test")
    print("=" * 50)
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    print("\nTesting Point Prompt Mode:")
    model_point = AdaptivePromptGenerator(
        use_attention=True,
        pretrained_backbone=False,
        prompt_type='point',
        num_points=3
    )
    points, labels, heatmap_point = model_point(images)
    print(f"  Input shape: {images.shape}")
    print(f"  Points shape: {points.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Heatmap shape: {heatmap_point.shape}")
    
    print("\nTesting Box Prompt Mode:")
    model_box = AdaptivePromptGenerator(
        use_attention=True,
        pretrained_backbone=False,
        prompt_type='box'
    )
    boxes, _, heatmap_box = model_box(images)
    print(f"  Input shape: {images.shape}")
    print(f"  Boxes shape: {boxes.shape}")
    print(f"  Heatmap shape: {heatmap_box.shape}")
    
    print("\n" + "=" * 50)
    print("Test Completed!")
    print("=" * 50)