"""
Video Codec Tester

A tool for testing video codecs with different compression settings.
Measures quality (PSNR, SSIM) and performance (encoding/decoding time).

Usage:
    python -m ml_codec.tools.video_codec_tester --input video.mp4 --codec h264 --bitrate 300
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import av

from ml_codec.metrics import calculate_psnr, calculate_ssim


class VideoCodecTester:
    """Test video codecs with quality and performance metrics."""
    
    def __init__(self, codec: str = "h264", bitrate: int = 300000, fps: int = 20):
        """
        Initialize codec tester.
        
        Args:
            codec: Codec name (h264, vp8, vp9, hevc)
            bitrate: Target bitrate in bits per second
            fps: Frames per second
        """
        self.codec = codec
        self.bitrate = bitrate
        self.fps = fps
        
        # Map codec names to PyAV codec strings
        self.codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "hevc": "libx265",
            "vp8": "libvpx",
            "vp9": "libvpx-vp9",
        }
    
    def load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to input video
            max_frames: Maximum number of frames to load (None = all)
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        print(f"[VideoCodecTester] Loading frames from {video_path}...")
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        print(f"[VideoCodecTester] Loaded {len(frames)} frames")
        return frames
    
    def encode_decode_frames(
        self, 
        frames: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Encode and decode frames using the specified codec.
        
        Args:
            frames: List of input frames (BGR format)
            
        Returns:
            Tuple of (decoded_frames, stats_dict)
        """
        if not frames:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        codec_name = self.codec_map.get(self.codec, self.codec)
        
        print(f"[VideoCodecTester] Encoding {len(frames)} frames...")
        print(f"  Codec: {codec_name}")
        print(f"  Bitrate: {self.bitrate/1000:.1f} kbps")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.fps}")
        
        # Create temporary file for encoded video
        # Use appropriate container for codec
        if self.codec in ["vp8", "vp9"]:
            temp_path = "/tmp/codec_test_temp.webm"
        else:
            temp_path = "/tmp/codec_test_temp.mp4"
        
        # Encoding
        encode_start = time.perf_counter()
        output = av.open(temp_path, mode='w')
        stream = output.add_stream(codec_name, rate=self.fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = self.bitrate
        
        # Codec-specific options for faster encoding
        if self.codec in ["h264", "h265", "hevc"]:
            stream.options = {
                'preset': 'ultrafast',  # Fast encoding
                'tune': 'zerolatency',  # Low latency
            }
        
        total_encode_time = 0
        for frame in frames:
            frame_start = time.perf_counter()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb_frame, format='rgb24')
            
            # Encode frame
            for packet in stream.encode(av_frame):
                output.mux(packet)
            
            total_encode_time += time.perf_counter() - frame_start
        
        # Flush encoder
        for packet in stream.encode():
            output.mux(packet)
        
        output.close()
        encode_total_time = time.perf_counter() - encode_start
        
        print(f"[VideoCodecTester] Encoding complete: {encode_total_time:.2f}s")
        print(f"  Average: {total_encode_time/len(frames)*1000:.2f}ms per frame")
        
        # Get file size
        file_size = Path(temp_path).stat().st_size
        
        # Decoding
        print(f"[VideoCodecTester] Decoding {len(frames)} frames...")
        decode_start = time.perf_counter()
        
        decoded_frames = []
        container = av.open(temp_path)
        video_stream = container.streams.video[0]
        
        total_decode_time = 0
        for packet in container.demux(video_stream):
            frame_start = time.perf_counter()
            
            for av_frame in packet.decode():
                # Convert to numpy array (RGB)
                rgb_array = av_frame.to_ndarray(format='rgb24')
                # Convert RGB to BGR for OpenCV
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                decoded_frames.append(bgr_array)
                
                total_decode_time += time.perf_counter() - frame_start
        
        container.close()
        decode_total_time = time.perf_counter() - decode_start
        
        print(f"[VideoCodecTester] Decoding complete: {decode_total_time:.2f}s")
        print(f"  Average: {total_decode_time/len(decoded_frames)*1000:.2f}ms per frame")
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
        
        # Statistics
        stats = {
            'encode_time_total': encode_total_time,
            'encode_time_per_frame': total_encode_time / len(frames),
            'decode_time_total': decode_total_time,
            'decode_time_per_frame': total_decode_time / len(decoded_frames),
            'file_size_bytes': file_size,
            'bitrate_actual': (file_size * 8 * self.fps) / len(frames),
            'compression_ratio': (len(frames) * width * height * 3) / file_size,
        }
        
        return decoded_frames, stats


def main():
    parser = argparse.ArgumentParser(description="Test video codecs with quality metrics")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--codec", default="h264", choices=["h264", "h265", "hevc", "vp8", "vp9"],
                       help="Video codec to use")
    parser.add_argument("--bitrate", type=int, default=300, help="Target bitrate in kbps")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    parser.add_argument("--output", help="Save decoded video to file (optional)")
    
    args = parser.parse_args()
    
    # Convert bitrate to bps
    bitrate_bps = args.bitrate * 1000
    
    # Initialize tester
    tester = VideoCodecTester(codec=args.codec, bitrate=bitrate_bps, fps=args.fps)
    
    # Load frames
    original_frames = tester.load_video_frames(args.input, max_frames=args.max_frames)
    
    if not original_frames:
        print("[ERROR] No frames loaded!")
        return
    
    # Encode and decode
    decoded_frames, stats = tester.encode_decode_frames(original_frames)
    
    # Calculate quality metrics
    print(f"\n[VideoCodecTester] Calculating quality metrics...")
    psnr_values = []
    ssim_values = []
    
    for i, (orig, decoded) in enumerate(zip(original_frames, decoded_frames)):
        psnr_values.append(calculate_psnr(orig, decoded))
        ssim_values.append(calculate_ssim(orig, decoded))
    
    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    
    # Print results
    print(f"\n{'='*60}")
    print(f"CODEC TEST RESULTS")
    print(f"{'='*60}")
    print(f"Codec:              {args.codec.upper()}")
    print(f"Target Bitrate:     {args.bitrate} kbps")
    print(f"Actual Bitrate:     {stats['bitrate_actual']/1000:.1f} kbps")
    print(f"Frames Processed:   {len(original_frames)}")
    print(f"File Size:          {stats['file_size_bytes']/1024:.1f} KB")
    print(f"Compression Ratio:  {stats['compression_ratio']:.1f}x")
    print(f"\nPerformance:")
    print(f"  Encode time:      {stats['encode_time_per_frame']*1000:.2f} ms/frame")
    print(f"  Decode time:      {stats['decode_time_per_frame']*1000:.2f} ms/frame")
    print(f"  Total time:       {stats['encode_time_total']+stats['decode_time_total']:.2f}s")
    print(f"\nQuality:")
    print(f"  Average PSNR:     {avg_psnr:.2f} dB")
    print(f"  Min PSNR:         {min(psnr_values):.2f} dB")
    print(f"  Max PSNR:         {max(psnr_values):.2f} dB")
    print(f"  Average SSIM:     {avg_ssim:.4f}")
    print(f"  Min SSIM:         {min(ssim_values):.4f}")
    print(f"  Max SSIM:         {max(ssim_values):.4f}")
    print(f"{'='*60}\n")
    
    # Save output video if requested
    if args.output:
        output_path = Path(args.output)
        
        # If a directory is provided, generate filename automatically.
        if output_path.is_dir():
            filename = f"{Path(args.input).stem}_{args.codec}_{args.bitrate}kbps_decoded.mp4"
            output_path = output_path / filename
        else:
            # Ensure .mp4 extension
            if output_path.suffix == "":
                output_path = output_path.with_suffix(".mp4")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[VideoCodecTester] Saving decoded video to {output_path}...")
        height, width = decoded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
        
        for frame in decoded_frames:
            out.write(frame)
        
        out.release()
        print(f"[VideoCodecTester] Saved!")


if __name__ == "__main__":
    main()

