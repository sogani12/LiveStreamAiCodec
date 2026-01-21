"""
Dataset generation tool for training Residual ESPCN.

Extracts frames from videos, encodes/decodes them at various bitrates to create
LR images, applies augmentations, and saves LR-HR pairs for training.

Usage:
    python -m ml_codec.tools.generate_training_dataset \
        --input video.mp4 \
        --output-dir datasets/train \
        --codec h264 \
        --bitrates 50,100,200,300 \
        --augment blocking,ringing,blur \
        --max-frames 1000
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ml_codec.tools.video_codec_tester import VideoCodecTester
from ml_codec.augmentations import (
    add_blocking_artifacts,
    add_ringing_artifacts,
    apply_compression_blur,
)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        if desc:
            print(f"[{desc}]")
        return iterable


def process_frame_batch(
    frames: List[np.ndarray],
    bitrate: int,
    codec: str,
    fps: int,
    augmentations: List[str],
    augmentation_intensity: float,
    frame_start_idx: int,
    output_lr_dir: Path,
    output_hr_dir: Path,
    seed: Optional[int] = None
) -> Tuple[int, int]:
    """
    Process a batch of frames: encode/decode and apply augmentations.
    
    Args:
        frames: List of frames to process
        bitrate: Bitrate in bps for encoding
        codec: Codec name (h264, vp8, etc.)
        fps: Frames per second
        augmentations: List of augmentation names to apply
        augmentation_intensity: Intensity of augmentations (0.0 to 1.0)
        frame_start_idx: Starting index for frame numbering
        output_lr_dir: Directory to save LR images
        output_hr_dir: Directory to save HR images
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (frames_processed, frames_saved)
    """
    if not frames:
        return 0, 0
    
    # Create codec tester for this batch
    tester = VideoCodecTester(codec=codec, bitrate=bitrate, fps=fps)
    
    # Encode and decode frames to create LR images
    try:
        decoded_frames, _ = tester.encode_decode_frames(frames)
    except Exception as e:
        print(f"[ERROR] Failed to encode/decode batch: {e}")
        return 0, 0
    
    frames_saved = 0
    
    # Process each frame pair
    for i, (hr_frame, lr_frame) in enumerate(zip(frames, decoded_frames)):
        frame_idx = frame_start_idx + i
        
        # Apply augmentations to LR frame
        augmented_lr = lr_frame.copy()
        
        if seed is not None:
            np.random.seed(seed + frame_idx)
        
        for aug_name in augmentations:
            if aug_name == "blocking":
                augmented_lr = add_blocking_artifacts(
                    augmented_lr,
                    intensity=augmentation_intensity,
                    random_seed=seed + frame_idx if seed is not None else None
                )
            elif aug_name == "ringing":
                augmented_lr = add_ringing_artifacts(
                    augmented_lr,
                    intensity=augmentation_intensity
                )
            elif aug_name == "blur":
                augmented_lr = apply_compression_blur(
                    augmented_lr,
                    intensity=augmentation_intensity
                )
        
        # Save LR-HR pair with matching filename stem
        filename_stem = f"frame_{frame_idx:06d}"
        
        # Save HR (original) frame
        hr_path = output_hr_dir / f"{filename_stem}.png"
        cv2.imwrite(str(hr_path), hr_frame)
        
        # Save LR (decoded + augmented) frame
        lr_path = output_lr_dir / f"{filename_stem}.png"
        cv2.imwrite(str(lr_path), augmented_lr)
        
        frames_saved += 1
    
    return len(frames), frames_saved


def generate_dataset(
    input_video: str,
    output_dir: str,
    codec: str = "h264",
    bitrates: List[int] = [100, 200, 300],
    fps: int = 20,
    max_frames: Optional[int] = None,
    augmentations: List[str] = [],
    augmentation_intensity: float = 0.3,
    batch_size: int = 50,
    num_workers: int = 4,
    seed: Optional[int] = None
):
    """
    Generate training dataset from video file.
    
    Args:
        input_video: Path to input video file
        output_dir: Output directory for LR/HR pairs
        codec: Codec to use for encoding (h264, vp8, vp9)
        bitrates: List of bitrates in kbps
        fps: Frames per second
        max_frames: Maximum number of frames to process (None = all)
        augmentations: List of augmentation names (blocking, ringing, blur)
        augmentation_intensity: Intensity of augmentations (0.0 to 1.0)
        batch_size: Number of frames to process per batch
        num_workers: Number of parallel workers
        seed: Random seed for reproducibility
    """
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create LR and HR directories
    lr_dir = output_path / "lr"
    hr_dir = output_path / "hr"
    lr_dir.mkdir(exist_ok=True)
    hr_dir.mkdir(exist_ok=True)
    
    print(f"[Dataset Generation] Input: {input_video}")
    print(f"[Dataset Generation] Output: {output_dir}")
    print(f"[Dataset Generation] Codec: {codec}")
    print(f"[Dataset Generation] Bitrates: {bitrates} kbps")
    print(f"[Dataset Generation] Augmentations: {augmentations}")
    print(f"[Dataset Generation] Augmentation intensity: {augmentation_intensity}")
    
    # Load all frames from video
    print(f"\n[Dataset Generation] Loading frames from video...")
    tester = VideoCodecTester(codec=codec, bitrate=300000, fps=fps)
    all_frames = tester.load_video_frames(str(input_path), max_frames=max_frames)
    
    if not all_frames:
        raise ValueError("No frames loaded from video!")
    
    print(f"[Dataset Generation] Loaded {len(all_frames)} frames")
    
    total_pairs = 0
    
    # Process each bitrate
    for bitrate_kbps in bitrates:
        bitrate_bps = bitrate_kbps * 1000
        print(f"\n[Dataset Generation] Processing bitrate: {bitrate_kbps} kbps")
        
        # Split frames into batches
        num_batches = (len(all_frames) + batch_size - 1) // batch_size
        batches = [
            all_frames[i:i+batch_size]
            for i in range(0, len(all_frames), batch_size)
        ]
        
        # Process batches (sequentially for now to avoid codec conflicts)
        # Parallel processing can be added later if needed
        frames_processed = 0
        
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Bitrate {bitrate_kbps}kbps")):
            frame_start_idx = total_pairs
            
            # Process batch
            processed, saved = process_frame_batch(
                frames=batch,
                bitrate=bitrate_bps,
                codec=codec,
                fps=fps,
                augmentations=augmentations,
                augmentation_intensity=augmentation_intensity,
                frame_start_idx=frame_start_idx,
                output_lr_dir=lr_dir,
                output_hr_dir=hr_dir,
                seed=seed
            )
            
            frames_processed += saved
            total_pairs += saved
        
        print(f"[Dataset Generation] Bitrate {bitrate_kbps} kbps: {frames_processed} pairs generated")
    
    print(f"\n[Dataset Generation] Complete!")
    print(f"[Dataset Generation] Total LR-HR pairs: {total_pairs}")
    print(f"[Dataset Generation] LR images: {lr_dir}")
    print(f"[Dataset Generation] HR images: {hr_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training dataset from video files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video file or directory containing videos"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for LR/HR image pairs"
    )
    parser.add_argument(
        "--codec",
        default="h264",
        choices=["h264", "h265", "hevc", "vp8", "vp9"],
        help="Video codec to use for encoding"
    )
    parser.add_argument(
        "--bitrates",
        type=str,
        default="50,100,200,300",
        help="Comma-separated list of bitrates in kbps (e.g., '50,100,200')"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (None = all)"
    )
    parser.add_argument(
        "--augment",
        type=str,
        default="",
        help="Comma-separated list of augmentations: blocking,ringing,blur"
    )
    parser.add_argument(
        "--augmentation-intensity",
        type=float,
        default=0.3,
        help="Intensity of augmentations (0.0 to 1.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of frames to process per batch"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Parse bitrates
    bitrates = [int(b.strip()) for b in args.bitrates.split(",") if b.strip()]
    if not bitrates:
        raise ValueError("At least one bitrate must be specified")
    
    # Parse augmentations
    augmentations = []
    if args.augment:
        augmentations = [a.strip() for a in args.augment.split(",") if a.strip()]
        valid_augs = {"blocking", "ringing", "blur"}
        invalid_augs = set(augmentations) - valid_augs
        if invalid_augs:
            raise ValueError(f"Invalid augmentations: {invalid_augs}. Valid: {valid_augs}")
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    if input_path.is_file():
        # Single video file
        generate_dataset(
            input_video=str(input_path),
            output_dir=args.output_dir,
            codec=args.codec,
            bitrates=bitrates,
            fps=args.fps,
            max_frames=args.max_frames,
            augmentations=augmentations,
            augmentation_intensity=args.augmentation_intensity,
            batch_size=args.batch_size,
            seed=args.seed
        )
    elif input_path.is_dir():
        # Directory of videos - process each video
        video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.webm"))
        if not video_files:
            raise ValueError(f"No video files found in {input_path}")
        
        print(f"[Dataset Generation] Found {len(video_files)} video files")
        
        for video_file in video_files:
            print(f"\n{'='*60}")
            print(f"Processing: {video_file.name}")
            print(f"{'='*60}")
            
            # Create subdirectory for this video
            video_stem = video_file.stem
            video_output_dir = Path(args.output_dir) / video_stem
            
            generate_dataset(
                input_video=str(video_file),
                output_dir=str(video_output_dir),
                codec=args.codec,
                bitrates=bitrates,
                fps=args.fps,
                max_frames=args.max_frames,
                augmentations=augmentations,
                augmentation_intensity=args.augmentation_intensity,
                batch_size=args.batch_size,
                seed=args.seed
            )
    else:
        raise ValueError(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
