"""
Helper script to download and prepare Vimeo-90K dataset for training.

Vimeo-90K is a high-quality video dataset commonly used for video enhancement tasks.
It contains 90,000+ video sequences with 7 frames each.

Dataset info: http://toflow.csail.mit.edu/
Download: https://github.com/anchen1011/toflow/issues/5 (see comments for download links)

Usage:
    # Download and prepare dataset
    python -m ml_codec.tools.download_vimeo90k \
        --download-dir datasets/vimeo90k \
        --prepare-videos

    # Or just prepare if already downloaded
    python -m ml_codec.tools.download_vimeo90k \
        --download-dir datasets/vimeo90k \
        --prepare-videos \
        --skip-download
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(f"[{desc}]")
        return iterable


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def download_vimeo90k(download_dir: Path, train_only: bool = True):
    """
    Download Vimeo-90K dataset.
    
    Note: The official download requires manual steps. This function provides
    instructions and can download if wget/curl is available.
    
    Args:
        download_dir: Directory to download dataset to
        train_only: If True, only download training set (smaller)
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("[Vimeo-90K] Download Instructions:")
    print("=" * 60)
    print("The Vimeo-90K dataset is large (~90GB for full, ~30GB for train-only)")
    print("Official download links are available at:")
    print("  https://github.com/anchen1011/toflow/issues/5")
    print("\nRecommended download method:")
    print("1. Visit the GitHub issue above")
    print("2. Find the download links in the comments")
    print("3. Download 'vimeo_triplet.tar' (training set, ~30GB)")
    print("4. Extract to:", download_dir)
    print("=" * 60)
    
    # Check if dataset already exists
    sequences_dir = download_dir / "sequences"
    if sequences_dir.exists() and any(sequences_dir.iterdir()):
        print(f"\n[Vimeo-90K] Dataset appears to already exist at {sequences_dir}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Try to download using wget if available
    print("\n[Vimeo-90K] Attempting automatic download...")
    print("Note: You may need to download manually if this fails.")
    
    # Common download URLs (these may change, check GitHub issue)
    base_urls = [
        "https://github.com/anchen1011/toflow/releases/download/v1.0.0/vimeo_triplet.tar",
        # Add other mirrors if available
    ]
    
    for url in base_urls:
        try:
            print(f"[Vimeo-90K] Trying to download from: {url}")
            tar_path = download_dir / "vimeo_triplet.tar"
            
            # Try wget first
            try:
                subprocess.run(
                    ["wget", "-c", url, "-O", str(tar_path)],
                    check=True,
                    timeout=10
                )
                print(f"[Vimeo-90K] Download complete: {tar_path}")
                return extract_vimeo90k(download_dir)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Try curl
            try:
                subprocess.run(
                    ["curl", "-L", "-o", str(tar_path), url],
                    check=True,
                    timeout=10
                )
                print(f"[Vimeo-90K] Download complete: {tar_path}")
                return extract_vimeo90k(download_dir)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
                
        except Exception as e:
            print(f"[Vimeo-90K] Download failed: {e}")
            continue
    
    print("\n[Vimeo-90K] Automatic download failed.")
    print("Please download manually and extract to:", download_dir)
    return False


def extract_vimeo90k(download_dir: Path) -> bool:
    """
    Extract Vimeo-90K tar file.
    
    Args:
        download_dir: Directory containing the tar file
        
    Returns:
        True if extraction successful
    """
    tar_path = download_dir / "vimeo_triplet.tar"
    
    if not tar_path.exists():
        print(f"[Vimeo-90K] Tar file not found: {tar_path}")
        return False
    
    print(f"[Vimeo-90K] Extracting {tar_path}...")
    print("This may take a while (30GB+ to extract)...")
    
    try:
        import tarfile
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(download_dir)
        print("[Vimeo-90K] Extraction complete!")
        
        # Clean up tar file to save space
        response = input("Delete tar file to save space? (y/n): ")
        if response.lower() == 'y':
            tar_path.unlink()
            print(f"[Vimeo-90K] Deleted {tar_path}")
        
        return True
    except Exception as e:
        print(f"[Vimeo-90K] Extraction failed: {e}")
        return False


def convert_sequences_to_videos(
    sequences_dir: Path,
    output_dir: Path,
    fps: int = 24,
    max_sequences: Optional[int] = None
):
    """
    Convert Vimeo-90K frame sequences to video files.
    
    Vimeo-90K stores sequences as folders with 7 PNG frames each.
    This function converts them to MP4 videos for easier processing.
    
    Args:
        sequences_dir: Directory containing sequence folders
        output_dir: Directory to save video files
        fps: Frames per second for output videos
        max_sequences: Maximum number of sequences to process (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sequence folders
    # Vimeo-90K structure: sequences/00001/0001/im1.png, im2.png, ..., im7.png
    sequence_folders = []
    
    for folder in sequences_dir.rglob("*/"):
        # Check if folder contains frame images
        frame_files = sorted(list(folder.glob("im*.png")))
        if len(frame_files) >= 7:  # Vimeo-90K has 7 frames per sequence
            sequence_folders.append((folder, frame_files))
    
    if not sequence_folders:
        print(f"[Vimeo-90K] No sequence folders found in {sequences_dir}")
        print("Expected structure: sequences/XXXXX/XXXX/im1.png ... im7.png")
        return
    
    print(f"[Vimeo-90K] Found {len(sequence_folders)} sequence folders")
    
    if max_sequences:
        sequence_folders = sequence_folders[:max_sequences]
        print(f"[Vimeo-90K] Processing first {max_sequences} sequences")
    
    # Convert each sequence to video
    videos_created = 0
    
    for seq_folder, frame_files in tqdm(sequence_folders, desc="Converting sequences"):
        # Create unique video filename from folder path
        # e.g., sequences/00001/0001 -> 00001_0001.mp4
        rel_path = seq_folder.relative_to(sequences_dir)
        video_name = "_".join(rel_path.parts) + ".mp4"
        video_path = output_dir / video_name
        
        if video_path.exists():
            continue  # Skip if already converted
        
        # Load frames
        frames = []
        for frame_file in frame_files[:7]:  # Only first 7 frames
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 7:
            continue  # Skip incomplete sequences
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        videos_created += 1
    
    print(f"[Vimeo-90K] Created {videos_created} video files in {output_dir}")


def prepare_vimeo90k(
    download_dir: Path,
    output_videos_dir: Optional[Path] = None,
    skip_download: bool = False,
    skip_conversion: bool = False,
    max_sequences: Optional[int] = None,
    fps: int = 24
):
    """
    Complete preparation pipeline for Vimeo-90K dataset.
    
    Args:
        download_dir: Directory containing or to contain Vimeo-90K dataset
        output_videos_dir: Directory to save converted videos (None = download_dir/videos)
        skip_download: Skip download step (assume dataset already exists)
        skip_conversion: Skip video conversion (use frame sequences directly)
        max_sequences: Maximum sequences to process (None = all)
        fps: Frames per second for video conversion
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    sequences_dir = download_dir / "sequences"
    
    # Step 1: Download (if needed)
    if not skip_download:
        if not sequences_dir.exists() or not any(sequences_dir.iterdir()):
            print("[Vimeo-90K] Step 1: Downloading dataset...")
            download_vimeo90k(download_dir)
        else:
            print("[Vimeo-90K] Dataset already exists, skipping download")
    else:
        print("[Vimeo-90K] Skipping download step")
    
    # Check if sequences exist
    if not sequences_dir.exists():
        print(f"[ERROR] Sequences directory not found: {sequences_dir}")
        print("Please download the dataset first or check the path.")
        return False
    
    # Step 2: Convert to videos (if needed)
    if not skip_conversion:
        if output_videos_dir is None:
            output_videos_dir = download_dir / "videos"
        
        print(f"\n[Vimeo-90K] Step 2: Converting sequences to videos...")
        print(f"Output directory: {output_videos_dir}")
        
        convert_sequences_to_videos(
            sequences_dir=sequences_dir,
            output_dir=output_videos_dir,
            fps=fps,
            max_sequences=max_sequences
        )
        
        print(f"\n[Vimeo-90K] Videos ready at: {output_videos_dir}")
        print("You can now use generate_training_dataset.py on this directory:")
        print(f"  python -m ml_codec.tools.generate_training_dataset \\")
        print(f"      --input {output_videos_dir} \\")
        print(f"      --output-dir datasets/train \\")
        print(f"      --codec h264 \\")
        print(f"      --bitrates 50,100,200,300")
    else:
        print("[Vimeo-90K] Skipping video conversion")
        print(f"Use sequences directly at: {sequences_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Vimeo-90K dataset"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="datasets/vimeo90k",
        help="Directory to download/extract dataset"
    )
    parser.add_argument(
        "--output-videos-dir",
        type=str,
        default=None,
        help="Directory to save converted videos (default: download_dir/videos)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (assume dataset already exists)"
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip video conversion step"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (for testing)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for video conversion"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare (convert to videos), skip download"
    )
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    # Handle prepare-only mode
    if args.prepare_only:
        args.skip_download = True
    
    # Run preparation pipeline
    success = prepare_vimeo90k(
        download_dir=Path(args.download_dir),
        output_videos_dir=Path(args.output_videos_dir) if args.output_videos_dir else None,
        skip_download=args.skip_download,
        skip_conversion=args.skip_conversion,
        max_sequences=args.max_sequences,
        fps=args.fps
    )
    
    if success:
        print("\n[Vimeo-90K] Preparation complete!")
    else:
        print("\n[Vimeo-90K] Preparation failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
