"""
Quick test script to validate the training pipeline with a small dataset.

This script:
1. Generates a test video (100 frames)
2. Creates LR-HR pairs from it
3. Runs a short training session to verify everything works

Usage:
    python -m ml_codec.tools.test_training_pipeline
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        sys.exit(1)
    
    print(f"\n[SUCCESS] {description} completed!")


def main():
    # Setup paths
    test_dir = Path("datasets/test_pipeline")
    test_video = test_dir / "test_video.mp4"
    dataset_dir = test_dir / "dataset"
    checkpoint_dir = test_dir / "checkpoints"
    
    # Create directories
    test_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Training Pipeline Test")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Test video: {test_video}")
    print(f"Dataset output: {dataset_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    # Step 1: Generate test video
    if not test_video.exists():
        run_command(
            [
                sys.executable, "-m", "ml_codec.tools.generate_test_video",
                "--output", str(test_video),
                "--frames", "100"
            ],
            "Generate test video (100 frames)"
        )
    else:
        print(f"\n[SKIP] Test video already exists: {test_video}")
    
    # Step 2: Generate LR-HR dataset
    lr_dir = dataset_dir / "lr"
    hr_dir = dataset_dir / "hr"
    
    if not lr_dir.exists() or not any(lr_dir.glob("*.png")):
        run_command(
            [
                sys.executable, "-m", "ml_codec.tools.generate_training_dataset",
                "--input", str(test_video),
                "--output-dir", str(dataset_dir),
                "--codec", "h264",
                "--bitrates", "100,200",
                "--augment", "blocking,ringing,blur",
                "--augmentation-intensity", "0.3",
                "--max-frames", "100"
            ],
            "Generate LR-HR dataset from test video"
        )
    else:
        print(f"\n[SKIP] Dataset already exists: {dataset_dir}")
        print(f"Found {len(list(lr_dir.glob('*.png')))} LR images")
    
    # Step 3: Run training (short test run)
    print(f"\n{'='*60}")
    print("Step: Run training test (2 epochs)")
    print(f"{'='*60}\n")
    
    run_command(
        [
            sys.executable, "-m", "ml_codec.tools.train_residual_espcn",
            "--lr-dir", str(lr_dir),
            "--hr-dir", str(hr_dir),
            "--output", str(checkpoint_dir / "test_model.pth"),
            "--epochs", "2",
            "--batch-size", "4",
            "--lr", "1e-4",
            "--patch-size", "128",
            "--pixel-loss-type", "l1",
            "--perceptual-weight", "0.0",  # Disable perceptual for faster test
            "--val-split", "0.2",
            "--seed", "42",
            "--device", "mps"  # Use CPU for compatibility, change to mps/cuda if available
        ],
        "Train model (2 epochs test)"
    )
    
    print(f"\n{'='*60}")
    print("Training Pipeline Test Complete!")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(f"  - Test video: {test_video}")
    print(f"  - Dataset: {dataset_dir}")
    print(f"  - Checkpoint: {checkpoint_dir / 'test_model.pth'}")
    print(f"\nIf this worked, you're ready to train on the full dataset!")
    print(f"\nNext steps:")
    print(f"  1. Wait for Vimeo-90K download to complete")
    print(f"  2. Generate full dataset:")
    print(f"     python -m ml_codec.tools.generate_training_dataset \\")
    print(f"         --input <vimeo_videos> \\")
    print(f"         --output-dir datasets/train \\")
    print(f"         --codec h264 --bitrates 50,100,200,300")
    print(f"  3. Train on full dataset with more epochs")


if __name__ == "__main__":
    main()
