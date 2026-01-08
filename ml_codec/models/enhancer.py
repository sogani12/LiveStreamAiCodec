"""
Lightweight decoder-side enhancement models using OpenCV operations.

These are fast, deterministic filters that demonstrate the ML decoder interface
and can be used as baselines before integrating neural networks.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..base import BaseMLDecoder


class BilateralEnhancer(BaseMLDecoder):
    """
    Joint bilateral filter for artifact reduction and denoising.
    
    Good for reducing compression artifacts while preserving edges.
    Fast (<5ms per frame typically).
    
    Config:
        d: Diameter of pixel neighborhood (default: 5)
        sigma_color: Filter sigma in color space (default: 50)
        sigma_space: Filter sigma in coordinate space (default: 50)
    """
    
    def __init__(self, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to reduce artifacts."""
        self.ensure_setup()
        return cv2.bilateralFilter(frame, self.d, self.sigma_color, self.sigma_space)


class SharpenEnhancer(BaseMLDecoder):
    """
    Unsharp mask for sharpness enhancement.
    
    Improves perceived sharpness by subtracting a blurred version from the original.
    Very fast (<1ms per frame).
    
    Config:
        strength: Sharpening strength (0.0-2.0, default: 1.0)
        kernel_size: Blur kernel size (must be odd, default: 5)
    """
    
    def __init__(self, strength: float = 1.0, kernel_size: int = 5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.strength = strength
        self.kernel_size = kernel_size
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply unsharp mask sharpening."""
        self.ensure_setup()
        # Create blurred version
        blurred = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(frame, 1.0 + self.strength, blurred, -self.strength, 0)
        # Clip to valid range
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class CombinedEnhancer(BaseMLDecoder):
    """
    Combined bilateral + sharpen enhancement.
    
    Applies bilateral filter first (denoise), then sharpening (enhance edges).
    Good balance for codec artifact reduction.
    
    Config:
        bilateral_d: Bilateral filter diameter (default: 5)
        bilateral_sigma_color: Bilateral color sigma (default: 50)
        bilateral_sigma_space: Bilateral space sigma (default: 50)
        sharpen_strength: Sharpening strength (default: 0.5)
        sharpen_kernel_size: Sharpening kernel size (default: 5)
    """
    
    def __init__(
        self,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        sharpen_strength: float = 0.5,
        sharpen_kernel_size: int = 5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.bilateral = BilateralEnhancer(
            d=bilateral_d,
            sigma_color=bilateral_sigma_color,
            sigma_space=bilateral_sigma_space
        )
        self.sharpen = SharpenEnhancer(
            strength=sharpen_strength,
            kernel_size=sharpen_kernel_size
        )
        # Setup sub-components
        self.bilateral.setup()
        self.sharpen.setup()
    
    def setup(self) -> None:
        """Setup sub-components."""
        super().setup()
        self.bilateral.setup()
        self.sharpen.setup()
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply bilateral filter, then sharpening."""
        self.ensure_setup()
        # First denoise with bilateral
        denoised = self.bilateral.enhance(frame)
        # Then sharpen
        return self.sharpen.enhance(denoised)


class NoOpEnhancer(BaseMLDecoder):
    """
    No-op enhancer that passes frames through unchanged.
    
    Useful for baseline comparisons and testing the pipeline.
    """
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Return frame unchanged."""
        self.ensure_setup()
        return frame.copy()



