"""
Compression augmentation functions for simulating codec artifacts.

These functions add realistic compression artifacts to images to simulate
poor network conditions and low bitrate encoding.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


def add_blocking_artifacts(
    image: np.ndarray,
    block_size: int = 8,
    intensity: float = 0.3,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate block-based compression artifacts (e.g., JPEG, H.264).
    
    Adds blocky artifacts by quantizing image blocks, simulating
    DCT-based compression where high frequencies are discarded.
    
    Args:
        image: Input image (BGR format, uint8)
        block_size: Size of compression blocks (typically 8x8)
        intensity: Strength of blocking (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        Image with blocking artifacts (same shape and dtype)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if intensity <= 0.0:
        return image.copy()
    
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Process each channel separately
    for c in range(image.shape[2]):
        channel = result[:, :, c]
        
        # Divide into blocks
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block_h = min(block_size, h - y)
                block_w = min(block_size, w - x)
                
                block = channel[y:y+block_h, x:x+block_w]
                
                # Quantize block (simulate DCT quantization)
                # Higher intensity = more quantization = more blocking
                quant_step = 1.0 + intensity * 20.0  # Quantization step size
                quantized = np.round(block / quant_step) * quant_step
                
                # Blend with original based on intensity
                block_result = (1 - intensity) * block + intensity * quantized
                channel[y:y+block_h, x:x+block_w] = block_result
        
        result[:, :, c] = channel
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_ringing_artifacts(
    image: np.ndarray,
    intensity: float = 0.2,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Simulate ringing artifacts from DCT-based compression.
    
    Ringing appears as oscillations near sharp edges, common in
    JPEG and H.264 compression at low bitrates.
    
    Args:
        image: Input image (BGR format, uint8)
        intensity: Strength of ringing (0.0 to 1.0)
        kernel_size: Size of edge detection kernel
        
    Returns:
        Image with ringing artifacts (same shape and dtype)
    """
    if intensity <= 0.0:
        return image.copy()
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Laplacian (captures high-frequency details)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    edges = np.abs(laplacian)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create ringing pattern (oscillations near edges)
    # Use a high-frequency pattern that follows edges
    h, w = image.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Create oscillating pattern
    freq = 0.1  # Frequency of oscillations
    pattern = np.sin(X * freq) * np.sin(Y * freq) * 255
    
    # Apply pattern only near edges
    edge_mask = (edges > 30).astype(np.float32)
    ringing_pattern = pattern.astype(np.float32) * edge_mask[:, :, np.newaxis]
    
    # Blend with original image
    result = image.astype(np.float32)
    ringing_effect = intensity * ringing_pattern * 0.1  # Scale down intensity
    result = result + np.stack([ringing_effect] * 3, axis=2)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_compression_blur(
    image: np.ndarray,
    intensity: float = 0.3,
    kernel_size_range: Tuple[int, int] = (3, 7)
) -> np.ndarray:
    """
    Apply Gaussian blur to simulate motion blur or low-quality encoding.
    
    Args:
        image: Input image (BGR format, uint8)
        intensity: Strength of blur (0.0 to 1.0)
        kernel_size_range: Range of kernel sizes (min, max)
        
    Returns:
        Blurred image (same shape and dtype)
    """
    if intensity <= 0.0:
        return image.copy()
    
    # Calculate kernel size based on intensity
    min_k, max_k = kernel_size_range
    kernel_size = int(min_k + intensity * (max_k - min_k))
    # Ensure odd kernel size
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Calculate sigma (standard deviation) for Gaussian blur
    sigma = intensity * 2.0
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Blend with original based on intensity
    result = (1 - intensity) * image.astype(np.float32) + intensity * blurred.astype(np.float32)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_frame_drops(
    frames: list,
    drop_probability: float = 0.1,
    random_seed: Optional[int] = None
) -> list:
    """
    Simulate frame drops by randomly removing frames from a sequence.
    
    This is used during dataset generation, not during training.
    Frame drops simulate packet loss in network transmission.
    
    Args:
        frames: List of frames (images as numpy arrays)
        drop_probability: Probability of dropping each frame (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        List of frames with some frames removed
    """
    if drop_probability <= 0.0:
        return frames.copy()
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    result = []
    for frame in frames:
        if np.random.random() > drop_probability:
            result.append(frame)
    
    return result
