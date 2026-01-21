"""
Compression augmentation functions for simulating codec artifacts.

These augmentations are used during dataset generation to create
realistic low-quality (LR) images from high-quality (HR) images.
"""

from .compression_augmentations import (
    add_blocking_artifacts,
    add_ringing_artifacts,
    apply_compression_blur,
    simulate_frame_drops,
)

__all__ = [
    'add_blocking_artifacts',
    'add_ringing_artifacts',
    'apply_compression_blur',
    'simulate_frame_drops',
]
