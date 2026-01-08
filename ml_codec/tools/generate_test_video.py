"""
Generate Test Video

Creates a simple test video with moving patterns for codec testing.

Usage:
    python -m ml_codec.tools.generate_test_video --output test.mp4 --frames 100
"""

import argparse
import cv2
import numpy as np


def generate_test_frames(num_frames: int, width: int = 640, height: int = 360) -> list:
    """
    Generate test frames with moving patterns.
    
    Creates frames with:
    - Moving gradient background
    - Bouncing circle
    - Frame counter text
    
    Args:
        num_frames: Number of frames to generate
        width: Frame width
        height: Frame height
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    
    # Circle properties
    circle_radius = 30
    circle_x = circle_radius
    circle_y = height // 2
    dx = 3  # Horizontal velocity
    dy = 2  # Vertical velocity
    
    print(f"Generating {num_frames} test frames ({width}x{height})...")
    
    for i in range(num_frames):
        # Create gradient background that shifts over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Horizontal gradient that moves
        offset = (i * 2) % 255
        for x in range(width):
            color_value = int((x / width * 255 + offset) % 255)
            frame[:, x] = [color_value, 100, 255 - color_value]
        
        # Draw bouncing circle
        cv2.circle(frame, (int(circle_x), int(circle_y)), circle_radius, (0, 255, 0), -1)
        
        # Update circle position (bounce off walls)
        circle_x += dx
        circle_y += dy
        
        if circle_x - circle_radius <= 0 or circle_x + circle_radius >= width:
            dx = -dx
        if circle_y - circle_radius <= 0 or circle_y + circle_radius >= height:
            dy = -dy
        
        # Add frame number
        text = f"Frame {i+1}/{num_frames}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_frames} frames...")
    
    print(f"Done! Generated {len(frames)} frames")
    return frames


def save_video(frames: list, output_path: str, fps: int = 20):
    """
    Save frames to video file.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    
    # Use H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving video to {output_path}...")
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved: {output_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")


def main():
    parser = argparse.ArgumentParser(description="Generate test video for codec testing")
    parser.add_argument("--output", default="test_video.mp4", help="Output video file")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=360, help="Frame height")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    
    args = parser.parse_args()
    
    # Generate frames
    frames = generate_test_frames(args.frames, args.width, args.height)
    
    # Save to video
    save_video(frames, args.output, args.fps)
    
    print(f"\nTest video created successfully!")
    print(f"You can now test it with:")
    print(f"  python -m ml_codec.tools.video_codec_tester --input {args.output} --codec h264 --bitrate 300")


if __name__ == "__main__":
    main()



