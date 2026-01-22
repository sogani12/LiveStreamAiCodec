"""
Training script for Residual ESPCN model with combined loss (MSE + Perceptual).

Usage:
    python -m ml_codec.tools.train_residual_espcn \
        --lr-dir /path/to/lr/frames \
        --hr-dir /path/to/hr/frames \
        --output checkpoints/residual_espcn.pth \
        --epochs 50 \
        --batch-size 8 \
        --lr 1e-4
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ml_codec.models.residual_espcn import ResidualESPCN
from ml_codec.losses.perceptual_loss import CombinedLoss


class LRHRDataset(Dataset):
    """Dataset for LR-HR image pairs."""
    
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        patch_size: int = 128,
        augment: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            lr_dir: Directory containing LR images
            hr_dir: Directory containing HR images
            patch_size: Size of patches to extract (None = use full images)
            augment: Whether to apply data augmentation
        """
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # Get all image files
        lr_all_files = list(self.lr_dir.glob("*.png")) + list(self.lr_dir.glob("*.jpg"))
        hr_all_files = list(self.hr_dir.glob("*.png")) + list(self.hr_dir.glob("*.jpg"))
        
        # Create dictionaries mapping filename stem (without extension) to file path
        lr_dict = {f.stem: f for f in lr_all_files}
        hr_dict = {f.stem: f for f in hr_all_files}
        
        # Find matching pairs by filename stem
        lr_stems = set(lr_dict.keys())
        hr_stems = set(hr_dict.keys())
        
        # Find stems that exist in both directories
        matching_stems = lr_stems & hr_stems
        
        # Check for mismatches
        lr_only = lr_stems - hr_stems
        hr_only = hr_stems - lr_stems
        
        if lr_only:
            print(f"[Dataset] WARNING: {len(lr_only)} LR files without HR pairs: {sorted(list(lr_only))[:5]}{'...' if len(lr_only) > 5 else ''}")
        
        if hr_only:
            print(f"[Dataset] WARNING: {len(hr_only)} HR files without LR pairs: {sorted(list(hr_only))[:5]}{'...' if len(hr_only) > 5 else ''}")
        
        # Assert that we have at least some matching pairs
        if not matching_stems:
            raise ValueError(
                f"No matching LR-HR pairs found! "
                f"LR files: {len(lr_all_files)}, HR files: {len(hr_all_files)}. "
                f"Check that filenames (without extensions) match between directories."
            )
        
        # Build matched file pairs (sorted by stem for reproducibility)
        self.lr_files = [lr_dict[stem] for stem in sorted(matching_stems)]
        self.hr_files = [hr_dict[stem] for stem in sorted(matching_stems)]
        
        # Final assertion that pairs are correctly matched
        assert len(self.lr_files) == len(self.hr_files), \
            f"Internal error: matched file counts don't match ({len(self.lr_files)} vs {len(self.hr_files)})"
        
        # Verify each pair has matching stems
        for lr_file, hr_file in zip(self.lr_files, self.hr_files):
            assert lr_file.stem == hr_file.stem, \
                f"Pair mismatch: {lr_file.name} <-> {hr_file.name}"
        
        print(f"[Dataset] Loaded {len(self.lr_files)} matched image pairs")
        if lr_only or hr_only:
            print(f"[Dataset] Skipped {len(lr_only)} LR-only and {len(hr_only)} HR-only files")
    
    def __len__(self) -> int:
        return len(self.lr_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LR-HR pair."""
        # Load images
        lr_path = self.lr_files[idx]
        hr_path = self.hr_files[idx]
        
        lr_img = cv2.imread(str(lr_path))
        hr_img = cv2.imread(str(hr_path))
        
        if lr_img is None or hr_img is None:
            raise ValueError(f"Failed to load image pair: {lr_path}, {hr_path}")
        
        # Convert BGR to RGB
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        # Extract patches if requested
        if self.patch_size:
            lr_img, hr_img = self._extract_patch(lr_img, hr_img)
        
        # Data augmentation
        if self.augment:
            lr_img, hr_img = self._augment(lr_img, hr_img)
        
        # Convert to tensors and normalize to [0, 1]
        # Use .copy() to ensure contiguous arrays (fixes negative stride issues from augmentation)
        lr_tensor = torch.from_numpy(lr_img.copy()).permute(2, 0, 1).float() / 255.0
        hr_tensor = torch.from_numpy(hr_img.copy()).permute(2, 0, 1).float() / 255.0
        
        return lr_tensor, hr_tensor
    
    def _extract_patch(self, lr_img: np.ndarray, hr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random patch from images."""
        h, w = lr_img.shape[:2]
        
        # Check if image is smaller than patch size
        if h < self.patch_size or w < self.patch_size:
            # Pad the image to at least patch_size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            
            # Use reflection padding to avoid edge artifacts
            lr_img = np.pad(lr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            hr_img = np.pad(hr_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            # Update dimensions after padding
            h, w = lr_img.shape[:2]
            assert h >= self.patch_size and w >= self.patch_size, \
                f"After padding, image size ({h}x{w}) must be >= patch_size ({self.patch_size})"
        
        # Random crop
        top = np.random.randint(0, max(1, h - self.patch_size + 1))
        left = np.random.randint(0, max(1, w - self.patch_size + 1))
        
        lr_patch = lr_img[top:top+self.patch_size, left:left+self.patch_size]
        hr_patch = hr_img[top:top+self.patch_size, left:left+self.patch_size]
        
        return lr_patch, hr_patch
    
    def _augment(self, lr_img: np.ndarray, hr_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            lr_img = cv2.flip(lr_img, 1)
            hr_img = cv2.flip(hr_img, 1)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            lr_img = cv2.flip(lr_img, 0)
            hr_img = cv2.flip(hr_img, 0)
        
        # Random rotation (90, 180, 270 degrees)
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            lr_img = np.rot90(lr_img, k)
            hr_img = np.rot90(hr_img, k)
        
        return lr_img, hr_img


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 100,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_perceptual_loss = 0.0
    num_batches = 0
    
    for batch_idx, (lr_images, hr_images) in enumerate(dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_images = model(lr_images)
        
        # Compute loss
        loss, loss_dict = criterion(pred_images, hr_images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_pixel_loss += loss_dict['pixel']
        total_perceptual_loss += loss_dict['perceptual']
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % print_freq == 0:
            print(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.6f} | "
                f"Pixel: {loss_dict['pixel']:.6f} | "
                f"Perceptual: {loss_dict['perceptual']:.6f}"
            )
    
    return {
        'total_loss': total_loss / num_batches,
        'pixel_loss': total_pixel_loss / num_batches,
        'perceptual_loss': total_perceptual_loss / num_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_perceptual_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Forward pass
            pred_images = model(lr_images)
            
            # Compute loss
            loss, loss_dict = criterion(pred_images, hr_images)
            
            # Accumulate losses
            total_loss += loss_dict['total']
            total_pixel_loss += loss_dict['pixel']
            total_perceptual_loss += loss_dict['perceptual']
            num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'pixel_loss': total_pixel_loss / num_batches,
        'perceptual_loss': total_perceptual_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Residual ESPCN model")
    parser.add_argument("--lr-dir", required=True, help="Directory containing LR images")
    parser.add_argument("--hr-dir", required=True, help="Directory containing HR images")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size (None = full images)")
    parser.add_argument("--num-features", type=int, default=64, help="Number of feature maps")
    parser.add_argument("--num-residual-blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--pixel-weight", type=float, default=1.0, help="Weight for pixel loss")
    parser.add_argument("--perceptual-weight", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--pixel-loss-type", type=str, default="mse", choices=["l1", "mse"], 
                        help="Type of pixel loss: 'l1' or 'mse' (default: 'mse')")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[Training] Set random seed: {args.seed}")
    
    # Setup device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[Training] MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[Training] Using device: {device}")
    
    # Create dataset
    dataset = LRHRDataset(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        augment=True,
    )
    
    # Split into train/val with generator for reproducibility
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # Create generator with seed for reproducibility
    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"[Training] Train: {train_size} samples, Val: {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    
    # Create model
    model = ResidualESPCN(
        in_channels=3,
        out_channels=3,
        num_features=args.num_features,
        num_residual_blocks=args.num_residual_blocks,
        upscale_factor=1,  # Same resolution for codec enhancement
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Training] Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = CombinedLoss(
        pixel_weight=args.pixel_weight,
        perceptual_weight=args.perceptual_weight,
        pixel_loss_type=args.pixel_loss_type,
    ).to(device)
    
    loss_desc = f"Pixel ({args.pixel_loss_type.upper()})"
    if args.perceptual_weight > 0.0:
        loss_desc += f" + Perceptual"
    print(f"[Training] Loss: {loss_desc} | Weights - Pixel: {args.pixel_weight}, Perceptual: {args.perceptual_weight}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        print(f"[Training] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print(f"\n[Training] Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_metrics['total_loss']:.6f} | "
              f"Pixel: {train_metrics['pixel_loss']:.6f} | "
              f"Perceptual: {train_metrics['perceptual_loss']:.6f}")
        print(f"Val Loss:   {val_metrics['total_loss']:.6f} | "
              f"Pixel: {val_metrics['pixel_loss']:.6f} | "
              f"Perceptual: {val_metrics['perceptual_loss']:.6f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics,
            'val_loss': val_metrics,
            'config': {
                'num_features': args.num_features,
                'num_residual_blocks': args.num_residual_blocks,
                'upscale_factor': 1,
            },
        }
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_path = args.output.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"[Training] Saved best model: {best_path} (val_loss: {best_val_loss:.6f})")
        
        # Save latest checkpoint
        torch.save(checkpoint, args.output)
    
    print(f"[Training] Training complete! Best model: {args.output.replace('.pth', '_best.pth')}")


if __name__ == "__main__":
    main()

