"""
Side-by-side video comparison utility.

Supports 2-way (original vs decoded) or 3-way (original vs decoded vs enhanced) comparison.

Usage:
    # 2-way comparison
    python -m ml_codec.tools.compare_videos \
        --original original.mp4 \
        --test decoded.mp4 \
        --show --diff --save comparisons.mp4
    
    # 3-way comparison with ML enhanced output
    python -m ml_codec.tools.compare_videos \
        --original original.mp4 \
        --test decoded.mp4 \
        --enhanced enhanced.mp4 \
        --show --diff --save comparisons.mp4
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ml_codec.metrics import calculate_psnr, calculate_ssim


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def resize_to_match(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    if (w, h) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def annotate(image: np.ndarray, text: str, align: str = "right") -> np.ndarray:
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    color = (255, 255, 255)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    if align == "right":
        x = image.shape[1] - text_size[0] - 10
    else:
        x = 10
    y = 30
    cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness)
    return annotated


def build_composite_view(
    original: np.ndarray,
    test: np.ndarray,
    diff_map: Optional[np.ndarray],
    psnr: float,
    ssim: float,
    frame_idx: int,
    enhanced: Optional[np.ndarray] = None,
    enhanced_psnr: Optional[float] = None,
    enhanced_ssim: Optional[float] = None,
) -> np.ndarray:
    height, width = original.shape[:2]

    orig_label = annotate(original, "Original", align="right")
    test_label = annotate(test, "Compressed", align="right")

    # Build top row: 2-way or 3-way comparison
    if enhanced is not None:
        enhanced_label = annotate(enhanced, "Enhanced", align="right")
        top_row = np.hstack([orig_label, test_label, enhanced_label])
        
        # Show metrics for both decoded and enhanced
        text1 = f"Frame {frame_idx}  Decoded: PSNR={psnr:.2f}dB SSIM={ssim:.4f}"
        text2 = f"Enhanced: PSNR={enhanced_psnr:.2f}dB SSIM={enhanced_ssim:.4f} (+{enhanced_psnr-psnr:+.2f}dB)"
        cv2.putText(top_row, text1, (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(top_row, text2, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        top_row = np.hstack([orig_label, test_label])
        text = f"Frame {frame_idx}  PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}"
        cv2.putText(top_row, text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if diff_map is None:
        return top_row

    diff_label = annotate(diff_map, "Difference (Heatmap)", align="left")
    pad_width = top_row.shape[1] - diff_label.shape[1]
    if pad_width > 0:
        padding = np.zeros((diff_label.shape[0], pad_width, 3), dtype=diff_label.dtype)
        diff_row = np.hstack([diff_label, padding])
    else:
        diff_row = diff_label[:, : top_row.shape[1]]

    combined = np.vstack([top_row, diff_row])
    return combined


def create_diff_heatmap(original: np.ndarray, test: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(original, test)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def compare_videos(
    original_path: str,
    test_path: str,
    max_frames: Optional[int],
    show: bool,
    save_path: Optional[str],
    show_diff: bool,
    fps: Optional[int],
    enhanced_path: Optional[str] = None,
):
    cap_orig = open_video(original_path)
    cap_test = open_video(test_path)
    cap_enhanced = open_video(enhanced_path) if enhanced_path else None

    frame_idx = 0
    psnr_values = []
    ssim_values = []
    enhanced_psnr_values = []
    enhanced_ssim_values = []

    writer = None
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    window_name = "Codec Comparison" + (" (3-way)" if cap_enhanced else "")
    if show:
        # Adjust window size for 3-way comparison
        width = 1920 if cap_enhanced else 1280
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, 720)

    try:
        while True:
            if max_frames and frame_idx >= max_frames:
                break

            orig_frame = read_frame(cap_orig)
            test_frame = read_frame(cap_test)
            enhanced_frame = read_frame(cap_enhanced) if cap_enhanced else None

            if orig_frame is None or test_frame is None:
                break
            if cap_enhanced and enhanced_frame is None:
                break

            frame_idx += 1

            # Resize to match original
            test_frame = resize_to_match(test_frame, (orig_frame.shape[1], orig_frame.shape[0]))
            if enhanced_frame is not None:
                enhanced_frame = resize_to_match(enhanced_frame, (orig_frame.shape[1], orig_frame.shape[0]))

            # Calculate metrics for decoded
            psnr = calculate_psnr(orig_frame, test_frame)
            ssim = calculate_ssim(orig_frame, test_frame)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # Calculate metrics for enhanced if available
            enhanced_psnr = None
            enhanced_ssim = None
            if enhanced_frame is not None:
                enhanced_psnr = calculate_psnr(orig_frame, enhanced_frame)
                enhanced_ssim = calculate_ssim(orig_frame, enhanced_frame)
                enhanced_psnr_values.append(enhanced_psnr)
                enhanced_ssim_values.append(enhanced_ssim)

            # Create difference heatmap (original vs decoded by default, or vs enhanced if showing 3-way)
            diff_target = enhanced_frame if enhanced_frame is not None else test_frame
            diff_heat = create_diff_heatmap(orig_frame, diff_target) if show_diff else None
            
            composite = build_composite_view(
                orig_frame, 
                test_frame, 
                diff_heat, 
                psnr, 
                ssim, 
                frame_idx,
                enhanced=enhanced_frame,
                enhanced_psnr=enhanced_psnr,
                enhanced_ssim=enhanced_ssim,
            )

            if show:
                cv2.imshow(window_name, composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(-1)

            if save_path:
                if writer is None:
                    height, width = composite.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(save_path), fourcc, fps or 30, (width, height))
                writer.write(composite)

    finally:
        cap_orig.release()
        cap_test.release()
        if cap_enhanced:
            cap_enhanced.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if not psnr_values:
        print("[compare_videos] No overlapping frames to compare.")
        return

    print("\nComparison Summary")
    print("==================")
    print(f"Frames compared:   {len(psnr_values)}")
    print(f"\nDecoded vs Original:")
    print(f"  PSNR - avg/min/max: {np.mean(psnr_values):.2f} / {np.min(psnr_values):.2f} / {np.max(psnr_values):.2f} dB")
    print(f"  SSIM - avg/min/max: {np.mean(ssim_values):.4f} / {np.min(ssim_values):.4f} / {np.max(ssim_values):.4f}")
    
    if enhanced_psnr_values:
        psnr_improvement = np.mean(enhanced_psnr_values) - np.mean(psnr_values)
        ssim_improvement = np.mean(enhanced_ssim_values) - np.mean(ssim_values)
        print(f"\nEnhanced vs Original:")
        print(f"  PSNR - avg/min/max: {np.mean(enhanced_psnr_values):.2f} / {np.min(enhanced_psnr_values):.2f} / {np.max(enhanced_psnr_values):.2f} dB")
        print(f"  SSIM - avg/min/max: {np.mean(enhanced_ssim_values):.4f} / {np.min(enhanced_ssim_values):.4f} / {np.max(enhanced_ssim_values):.4f}")
        print(f"\nImprovement:")
        print(f"  PSNR: {psnr_improvement:+.2f} dB")
        print(f"  SSIM: {ssim_improvement:+.4f}")
    
    if save_path:
        print(f"\nSaved comparison video to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side video comparison tool")
    parser.add_argument("--original", required=True, help="Original (reference) video file")
    parser.add_argument("--test", required=True, help="Test (compressed/decoded) video file")
    parser.add_argument("--enhanced", help="Enhanced (ML-processed) video file (optional, enables 3-way comparison)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to compare")
    parser.add_argument("--show", action="store_true", help="Display comparison window")
    parser.add_argument("--diff", action="store_true", help="Show difference heatmap")
    parser.add_argument("--save", help="Path to save composite video (optional)")
    parser.add_argument("--fps", type=int, help="FPS for saved video")

    args = parser.parse_args()

    compare_videos(
        original_path=args.original,
        test_path=args.test,
        enhanced_path=args.enhanced,
        max_frames=args.max_frames,
        show=args.show,
        save_path=args.save,
        show_diff=args.diff,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

