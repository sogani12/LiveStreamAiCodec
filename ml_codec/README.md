# ML Codec Research Framework

Fast, file-based testing framework for ML-enhanced video codecs.

## Why File-Based Testing?

- **10-50x faster** than WebRTC (hardware-accelerated H.264 vs slow VP8)
- **No network overhead** - focus on codec quality
- **Easy comparison** - test multiple codecs, bitrates, and ML models
- **Reproducible** - same input every time

## Quick Start

### Step 1: Generate Test Video

```bash
cd /Users/namans/live_stream_AI_codec_2
source venv/bin/activate
python -m ml_codec.tools.generate_test_video --output test_video.mp4 --frames 100
```

This creates a 100-frame test video with moving patterns.

### Step 2: Test Different Codecs

**H.264 (Hardware Accelerated - FAST!):**
```bash
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec h264 --bitrate 300
```

**VP8 (Software - for comparison):**
```bash
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec vp8 --bitrate 300
```

**VP9 (Better compression):**
```bash
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec vp9 --bitrate 300
```

### Step 3: Compare Bitrates

```bash
# High quality (1000 kbps)
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec h264 --bitrate 1000

# Medium quality (500 kbps)
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec h264 --bitrate 500

# Low quality (100 kbps) - good for ML enhancement testing
python -m ml_codec.tools.video_codec_tester --input test_video.mp4 --codec h264 --bitrate 100
```

### Step 4: Test ML Enhancement

Apply decoder-side ML enhancement to improve compressed video quality:

```bash
# Test with bilateral filter (artifact reduction)
python -m ml_codec.tools.video_codec_tester \
    --input test_video.mp4 \
    --codec h264 --bitrate 100 \
    --ml-decoder bilateral \
    --output .

# Test with combined enhancer (bilateral + sharpen)
python -m ml_codec.tools.video_codec_tester \
    --input test_video.mp4 \
    --codec h264 --bitrate 100 \
    --ml-decoder combined \
    --output .
```

**Available ML Decoders:**
- `noop` - No enhancement (baseline)
- `bilateral` - Joint bilateral filter for artifact reduction
- `sharpen` - Unsharp mask for sharpness enhancement
- `combined` - Bilateral + sharpen (recommended)

### Step 5: Visualize Differences (Side-by-side)

```bash
# 2-way: Display original vs decoded with heatmap
python -m ml_codec.tools.compare_videos \
    --original test_video.mp4 \
    --test decoded.mp4 \
    --show --diff

# 3-way: Compare original vs decoded vs enhanced
python -m ml_codec.tools.compare_videos \
    --original test_video.mp4 \
    --test decoded.mp4 \
    --enhanced enhanced.mp4 \
    --show --diff --save comparison_3way.mp4
```

## What You'll See

The codec tester shows:
- **Encode/decode speed** (ms per frame)
- **File size** and actual bitrate
- **Compression ratio**
- **Quality metrics** (PSNR + SSIM)
- **ML enhancement latency** (when `--ml-decoder` is used)
- **Quality improvement** (decoded vs enhanced)

### Example: Without ML Enhancement
```
============================================================
CODEC TEST RESULTS
============================================================
Codec:              H264
Target Bitrate:     300 kbps
Actual Bitrate:     285.3 kbps
Frames Processed:   100
File Size:          178.3 KB
Compression Ratio:  298.1x

Performance:
  Encode time:      1.70 ms/frame
  Decode time:      0.51 ms/frame
  Total time:       0.22s

Quality (Decoded):
  Average PSNR:     39.27 dB
  Average SSIM:     0.9933
============================================================
```

### Example: With ML Enhancement
```
============================================================
CODEC TEST RESULTS
============================================================
Codec:              H264
ML Enhancer:        bilateral
Target Bitrate:     100 kbps
Actual Bitrate:     157.8 kbps
Frames Processed:   20
File Size:          19.3 KB
Compression Ratio:  700.7x

Performance:
  Encode time:      1.03 ms/frame
  Decode time:      0.45 ms/frame
  ML enhance time:  0.86 ms/frame
  Total time:       0.06s

Quality (Decoded):
  Average PSNR:     32.13 dB
  Average SSIM:     0.9840

Quality (Enhanced):
  Average PSNR:     32.54 dB (+0.41 dB)
  Average SSIM:     0.9858 (+0.0018)
============================================================
```

**Expected Improvements:**
- **Bilateral filter**: +0.3-0.8 dB PSNR improvement, reduces compression artifacts
- **Sharpen**: +0.1-0.3 dB PSNR, improves perceived sharpness
- **Combined**: +0.4-1.0 dB PSNR, best overall quality improvement
- **Enhancement speed**: 0.5-2 ms/frame (fast enough for real-time)

## ML Enhancement Guide

### How It Works

ML enhancement runs **after** video decoding to improve quality:
1. Video is compressed with traditional codec (H.264, VP8, etc.)
2. Decoded frames may have artifacts (blocking, blurring, ringing)
3. ML enhancer processes each frame to reduce artifacts
4. Quality metrics compare original vs enhanced output

### When to Use ML Enhancement

**Best for:**
- Low bitrate compression (50-200 kbps) where artifacts are visible
- Real-time applications where <5ms latency is acceptable
- Research into ML-assisted codec improvements

**Not ideal for:**
- High bitrate (>500 kbps) where quality is already excellent
- Ultra-low latency requirements (<1ms per frame)
- Applications where deterministic output is critical

### Adding Your Own ML Models

Create a new enhancer by subclassing `BaseMLDecoder`:

```python
from ml_codec.base import BaseMLDecoder
import numpy as np

class MyEnhancer(BaseMLDecoder):
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        # Your enhancement logic here
        return enhanced_frame
```

Register it in `ml_codec/models/__init__.py`:
```python
from ..registry import register_decoder
register_decoder("my_model", MyEnhancer)
```

## Status

1. ✅ **Basic codec testing** (done!)
2. ✅ **SSIM quality metric** (done)
3. ✅ **ML enhancement integration** (done)
4. ✅ **Side-by-side comparison** (done)
5. 🔄 **Neural network models** (coming soon)

## File Structure

```
ml_codec/
├── __init__.py
├── README.md (this file)
├── base.py                      - BaseMLDecoder interface
├── registry.py                  - Model registry system
├── metrics.py                   - PSNR/SSIM calculations
├── tools/
│   ├── generate_test_video.py   - Create test videos
│   ├── video_codec_tester.py    - Test codecs with ML enhancement
│   └── compare_videos.py        - 2-way/3-way visualization
└── models/
    ├── __init__.py              - Model registration
    └── enhancer.py              - Built-in enhancement models
```

## Advantages Over WebRTC

| Feature | WebRTC (VP8) | File-Based (H.264) |
|---------|--------------|-------------------|
| Decode Speed | 540ms/frame | 2-10ms/frame |
| Encode Speed | ~50ms/frame | 5-15ms/frame |
| Hardware Accel | No | Yes |
| Reproducible | No (network) | Yes |
| Easy Testing | No | Yes |

## Complete Workflow Example

Here's a complete workflow from test video to ML-enhanced comparison:

```bash
# 1. Generate test video
python -m ml_codec.tools.generate_test_video --output test.mp4 --frames 100

# 2. Test codec at low bitrate WITHOUT ML
python -m ml_codec.tools.video_codec_tester \
    --input test.mp4 \
    --codec h264 --bitrate 100 \
    --output decoded.mp4

# 3. Test codec WITH ML enhancement
python -m ml_codec.tools.video_codec_tester \
    --input test.mp4 \
    --codec h264 --bitrate 100 \
    --ml-decoder combined \
    --output enhanced.mp4

# 4. Visualize 3-way comparison
python -m ml_codec.tools.compare_videos \
    --original test.mp4 \
    --test decoded.mp4 \
    --enhanced enhanced.mp4 \
    --show --diff --save comparison.mp4
```

## Using Your Own Videos

```bash
# Test your own video with ML enhancement
python -m ml_codec.tools.video_codec_tester \
    --input my_video.mp4 \
    --codec h264 \
    --bitrate 200 \
    --ml-decoder bilateral \
    --max-frames 200 \
    --output enhanced_output.mp4
```

The `--output` flag saves the enhanced video (or decoded if no ML enhancer is used) for visual inspection.

