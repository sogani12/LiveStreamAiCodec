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

## What You'll See

The codec tester shows:
- **Encode/decode speed** (ms per frame)
- **File size** and actual bitrate
- **Compression ratio**
- **Quality metrics** (PSNR + SSIM)

Example output:
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
  Encode time:      8.45 ms/frame
  Decode time:      2.31 ms/frame
  Total time:       1.08s

Quality:
  Average PSNR:     38.45 dB
  Min PSNR:         36.12 dB
  Max PSNR:         40.89 dB
  Average SSIM:     0.9621
  Min SSIM:         0.9450
  Max SSIM:         0.9783
============================================================
```

## Next Steps

1. ✅ **Basic codec testing** (done!)
2. ✅ **Add SSIM quality metric** (done)
3. 🔄 **ML enhancement integration** (coming soon)
4. 🔄 **Side-by-side comparison** (coming soon)

## File Structure

```
ml_codec/
├── __init__.py
├── README.md (this file)
├── tools/
│   ├── generate_test_video.py  - Create test videos
│   └── video_codec_tester.py   - Test codecs with metrics
└── models/
    └── (ML models coming soon)
```

## Advantages Over WebRTC

| Feature | WebRTC (VP8) | File-Based (H.264) |
|---------|--------------|-------------------|
| Decode Speed | 540ms/frame | 2-10ms/frame |
| Encode Speed | ~50ms/frame | 5-15ms/frame |
| Hardware Accel | No | Yes |
| Reproducible | No (network) | Yes |
| Easy Testing | No | Yes |

## Using Your Own Videos

```bash
# Use your webcam recording or any video
python -m ml_codec.tools.video_codec_tester \
    --input my_video.mp4 \
    --codec h264 \
    --bitrate 300 \
    --max-frames 200 \
    --output decoded_output.mp4
```

The `--output` flag saves the decoded video so you can visually inspect quality.

