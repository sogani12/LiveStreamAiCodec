"""
Perceptual loss using VGG features.

Based on "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
by Johnson et al. (2016).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG feature maps.
    
    Computes L2 loss between feature representations of predicted and target images
    extracted from intermediate layers of a pre-trained VGG network.
    """
    
    def __init__(
        self,
        feature_layers: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        use_normalized_features: bool = True,
    ):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: List of VGG layer indices to extract features from.
                          Default: [4, 9, 16, 23, 30] (relu1_2, relu2_2, relu3_3, relu4_3, relu5_3)
            weights: Weights for each layer in the loss calculation.
                    Default: Equal weights for all layers.
            use_normalized_features: If True, normalize feature maps before computing loss.
        """
        super().__init__()
        
        # Default layers: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        # These are good for perceptual quality
        if feature_layers is None:
            feature_layers = [4, 9, 16, 23, 30]
        
        if weights is None:
            # Default: equal weights, but can be adjusted
            weights = [1.0 / len(feature_layers)] * len(feature_layers)
        
        self.feature_layers = feature_layers
        self.weights = weights
        self.use_normalized_features = use_normalized_features
        
        # Load pre-trained VGG19 (or VGG16)
        vgg = models.vgg19(pretrained=True)
        
        # Extract features up to the specified layers
        self.feature_extractor = nn.ModuleList()
        
        # VGG19 feature layers (up to relu5_3)
        layers = list(vgg.features.children())
        
        for i, layer_idx in enumerate(feature_layers):
            # Extract all layers up to and including this layer
            self.feature_extractor.append(
                nn.Sequential(*layers[:layer_idx+1])
            )
        
        # Set to evaluation mode (no gradients for VGG)
        for module in self.feature_extractor:
            module.eval()
            # Freeze VGG weights (we don't train VGG)
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted image tensor (B, C, H, W), values in [0, 1]
            target: Target image tensor (B, C, H, W), values in [0, 1]
        
        Returns:
            Perceptual loss scalar tensor
        """
        # VGG expects RGB images in [0, 1] range
        # Assume input is already in correct format
        
        # Normalize to ImageNet statistics (VGG was trained on ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_normalized = (pred - mean) / std
        target_normalized = (target - mean) / std
        
        loss = 0.0
        
        # Extract features from each layer
        for i, extractor in enumerate(self.feature_extractor):
            # Extract features (VGG is frozen but gradients flow through pred)
            pred_features = extractor(pred_normalized)
            # Target features don't need gradients (detached)
            with torch.no_grad():
                target_features = extractor(target_normalized)
            
            # Normalize features if requested
            if self.use_normalized_features:
                # L2 normalize feature maps (channel-wise)
                pred_features = nn.functional.normalize(pred_features, p=2, dim=1)
                target_features = nn.functional.normalize(target_features, p=2, dim=1)
            
            # Compute MSE loss (L2 loss)
            # This will have gradients w.r.t. pred_features
            layer_loss = nn.functional.mse_loss(pred_features, target_features)
            
            # Weight and accumulate
            loss += self.weights[i] * layer_loss
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function: Pixel loss (L1 or MSE) + optional Perceptual loss.
    
    Supports different loss configurations:
    - L1 only: pixel_loss_type='l1', perceptual_weight=0.0
    - L1 + Perceptual: pixel_loss_type='l1', perceptual_weight>0.0
    - MSE + Perceptual: pixel_loss_type='mse', perceptual_weight>0.0 (default)
    """
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        pixel_loss_type: str = "mse",
        feature_layers: Optional[List[int]] = None,
        perceptual_weights: Optional[List[float]] = None,
    ):
        """
        Initialize combined loss.
        
        Args:
            pixel_weight: Weight for pixel-wise loss (default: 1.0)
            perceptual_weight: Weight for perceptual loss (default: 0.1, small part)
                              Set to 0.0 to use only pixel loss
            pixel_loss_type: Type of pixel loss - 'l1' or 'mse' (default: 'mse')
            feature_layers: VGG layers for perceptual loss (default: [4, 9, 16, 23, 30])
            perceptual_weights: Weights for each VGG layer (default: equal weights)
        """
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.pixel_loss_type = pixel_loss_type.lower()
        
        # Validate pixel loss type
        if self.pixel_loss_type not in ['l1', 'mse']:
            raise ValueError(f"pixel_loss_type must be 'l1' or 'mse', got '{pixel_loss_type}'")
        
        # Initialize pixel loss
        if self.pixel_loss_type == 'l1':
            self.pixel_loss = nn.L1Loss()
        else:  # mse
            self.pixel_loss = nn.MSELoss()
        
        # Initialize perceptual loss only if weight > 0
        if self.perceptual_weight > 0.0:
            self.perceptual_loss = PerceptualLoss(
                feature_layers=feature_layers,
                weights=perceptual_weights,
            )
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image tensor (B, C, H, W), values in [0, 1]
            target: Target image tensor (B, C, H, W), values in [0, 1]
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Pixel-wise loss (L1 or MSE)
        pixel_loss = self.pixel_loss(pred, target)
        
        # Perceptual loss (if enabled)
        if self.perceptual_loss is not None:
            perceptual_loss_val = self.perceptual_loss(pred, target)
        else:
            perceptual_loss_val = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = (
            self.pixel_weight * pixel_loss +
            self.perceptual_weight * perceptual_loss_val
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'pixel': pixel_loss.item(),
            'perceptual': perceptual_loss_val.item() if self.perceptual_loss is not None else 0.0,
        }
        
        return total_loss, loss_dict

