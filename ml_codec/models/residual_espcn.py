"""
Residual ESPCN (Efficient Sub-Pixel Convolutional Network) for video enhancement.

Based on "Real-Time Single Image and Video Super-Resolution" by Shi et al. (2016)
Extended with residual connections for better training and quality.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .neural_base import NeuralNetworkDecoder


class ResidualESPCN(nn.Module):
    """
    Residual ESPCN architecture for image/video enhancement.
    
    Architecture:
    - Feature extraction layers (conv layers)
    - Residual blocks for deeper feature learning
    - Sub-pixel convolution for efficient upsampling
    
    For codec enhancement (not super-resolution), input and output are same resolution,
    but we still use ESPCN's efficient feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_residual_blocks: int = 2,
        upscale_factor: int = 1,  # 1 for same resolution (codec enhancement)
    ):
        """
        Initialize Residual ESPCN model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            num_features: Number of feature maps in hidden layers
            num_residual_blocks: Number of residual blocks
            upscale_factor: Upscaling factor (1 = no upscaling, 2 = 2x, etc.)
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)
        self.conv_input_act = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Feature reconstruction
        # If upscale_factor > 1, use sub-pixel convolution
        # If upscale_factor == 1, use regular convolution (no upscaling)
        if upscale_factor > 1:
            # Sub-pixel convolution for upscaling
            self.conv_up = nn.Conv2d(
                num_features,
                out_channels * (upscale_factor ** 2),
                kernel_size=9,
                padding=4
            )
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        else:
            # Regular convolution for same resolution
            self.conv_up = nn.Conv2d(num_features, out_channels, kernel_size=9, padding=4)
            self.pixel_shuffle = None
        
        # Residual connection: add input to output if same resolution
        self.use_residual = (upscale_factor == 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Enhanced tensor (B, C, H*upscale, W*upscale)
        """
        # Store input for residual connection
        if self.use_residual:
            identity = x
        
        # Initial feature extraction
        out = self.conv_input_act(self.conv_input(x))
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
        
        # Feature reconstruction
        out = self.conv_up(out)
        
        # Sub-pixel shuffle if upscaling
        if self.pixel_shuffle is not None:
            out = self.pixel_shuffle(out)
        
        # Residual connection (for same-resolution enhancement)
        if self.use_residual:
            out = out + identity
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity  # Residual connection
        out = self.relu(out)
        return out


class ResidualESPCNEnhancer(NeuralNetworkDecoder):
    """
    Residual ESPCN enhancer for decoder-side video quality improvement.
    
    Integrates Residual ESPCN model with the ML decoder interface.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        num_features: int = 64,
        num_residual_blocks: int = 2,
        upscale_factor: int = 1,
        **config: Any
    ):
        """
        Initialize Residual ESPCN enhancer.
        
        Args:
            model_path: Path to trained model (.pth file). If None, creates untrained model.
            device: Device to run inference on ('cpu' or 'cuda')
            num_features: Number of feature maps (64 is typical)
            num_residual_blocks: Number of residual blocks (2-4 is typical)
            upscale_factor: Upscaling factor (1 for same-resolution enhancement)
            **config: Additional configuration
        """
        super().__init__(model_path=model_path, device=device, **config)
        self.num_features = num_features
        self.num_residual_blocks = num_residual_blocks
        self.upscale_factor = upscale_factor
    
    def _load_pytorch_model(self, model_path) -> None:
        """Load PyTorch model with architecture specification."""
        import torch
        
        self.model_type = 'pytorch'
        
        # Create model architecture
        model = ResidualESPCN(
            in_channels=3,
            out_channels=3,
            num_features=self.num_features,
            num_residual_blocks=self.num_residual_blocks,
            upscale_factor=self.upscale_factor,
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model = checkpoint['model']
            else:
                # Assume it's a state dict directly
                model.load_state_dict(checkpoint)
        else:
            # Assume it's a model object
            model = checkpoint
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        self.model = model
    
    def setup(self) -> None:
        """Setup model (load from file or create new)."""
        super().setup()
        
        # If no model path, create untrained model for inference
        # (useful for testing architecture)
        if not self.model_path or not self.model:
            import torch
            self.model_type = 'pytorch'
            self.model = ResidualESPCN(
                in_channels=3,
                out_channels=3,
                num_features=self.num_features,
                num_residual_blocks=self.num_residual_blocks,
                upscale_factor=self.upscale_factor,
            ).to(self.device)
            self.model.eval()
            print("[ResidualESPCN] Created untrained model (no checkpoint provided)")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model architecture
    model = ResidualESPCN(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_residual_blocks=2,
        upscale_factor=1,
    )
    
    num_params = count_parameters(model)
    print(f"Residual ESPCN Parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 360, 640)  # (B, C, H, W)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Input and output should match for upscale_factor=1"

