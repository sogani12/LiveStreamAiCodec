"""
Quality and performance metrics for video frames.
"""

import cv2
import numpy as np


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Implements the reference formula using Gaussian blur.
    Images are compared in grayscale (Y channel) for perceptual relevance.
    """
    gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY).astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel_size = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(gray1, kernel_size, sigma)
    mu2 = cv2.GaussianBlur(gray2, kernel_size, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 * gray1, kernel_size, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 * gray2, kernel_size, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, kernel_size, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())



