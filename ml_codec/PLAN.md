# Decoder-Side Enhancement: Dataset Generation, QAT, and Temporal Smoothing

## Overview

This plan implements the complete pipeline for training and deploying decoder-side ML enhancement for live streaming. The focus is on creating realistic training data from video/webcam recordings, training with quantization-aware training (QAT) for low latency, and adding temporal smoothing for stable inference.

## Components

### 1. Dataset Generation Tool

**File**: `ml_codec/tools/generate_training_dataset.py`

Create a new tool that:

- Extracts frames from video files or webcam recordings
- Generates LR images by encoding/decoding at various bitrates (50-500 kbps range)
- Applies augmentations to simulate poor network conditions:
  - **Blocking artifacts**: Add block-based compression artifacts
  - **Ringing artifacts**: Add edge ringing from compression
  - **Blur**: Gaussian blur to simulate motion blur or low quality
  - **Frame drops**: Optionally skip frames to simulate packet loss
- Saves LR-HR pairs with matching filenames (stem-based matching)

**Key features**:

- Reuse `VideoCodecTester.encode_decode_frames()` logic for encoding/decoding
- Support multiple codecs (H.264, VP8, VP9) and bitrate ranges
- Configurable augmentation intensity
- Parallel processing for large video files
- Progress tracking and statistics

**Usage**:

```bash
python -m ml_codec.tools.generate_training_dataset \
    --input video.mp4 \
    --output-dir datasets/train \
    --codec h264 \
    --bitrates 50,100,200,300 \
    --augment blocking,ringing,blur \
    --max-frames 1000
```

### 2. Augmentation Module

**File**: `ml_codec/augmentations/compression_augmentations.py`

Create augmentation functions:

- `add_blocking_artifacts()`: Simulate block-based compression artifacts
- `add_ringing_artifacts()`: Add edge ringing from DCT-based compression
- `apply_compression_blur()`: Gaussian blur with configurable kernel size
- `simulate_frame_drops()`: Optionally skip frames (for dataset generation, not training)

These augmentations can be applied:

1. During dataset generation (to LR images)
2. As data augmentation during training (optional, in `LRHRDataset`)

### 3. PyTorch QAT Integration

**File**: `ml_codec/tools/train_residual_espcn.py` (modify)

Add QAT support:

- Import `torch.quantization` modules
- Add `--use-qat` flag to enable quantization-aware training
- Wrap model with `torch.quantization.QuantStub()` and `torch.quantization.DeQuantStub()`
- Use `torch.quantization.prepare_qat()` to prepare model for QAT
- Convert to quantized model after training with `torch.quantization.convert()`
- Save both FP32 and INT8 checkpoints

**QAT workflow**:

1. Start with FP32 training (baseline)
2. Enable QAT with `--use-qat` flag
3. Fine-tune with QAT (typically fewer epochs)
4. Export quantized model checkpoint

**File**: `ml_codec/models/residual_espcn.py` (modify)

Update `ResidualESPCN` to support QAT:

- Add `QuantStub` and `DeQuantStub` layers
- Make architecture QAT-compatible (avoid operations that don't quantize well)
- Add `fuse_model()` method for layer fusion optimization

### 4. Temporal Smoothing (EMA Frame Buffering)

**File**: `ml_codec/models/residual_espcn.py` (modify `ResidualESPCNEnhancer`)

Add EMA-based temporal smoothing:

- Add `temporal_smoothing` parameter (boolean, default False)
- Add `ema_alpha` parameter (default 0.7) for exponential moving average
- Maintain internal frame buffer (previous enhanced frame)
- Apply EMA: `output = alpha * current_enhanced + (1 - alpha) * previous_enhanced`
- Initialize buffer on first frame

**Implementation**:

```python
def enhance(self, frame: np.ndarray) -> np.ndarray:
    enhanced = self._enhance_frame(frame)  # Model inference
    
    if self.temporal_smoothing:
        if self._prev_frame is None:
            self._prev_frame = enhanced
        else:
            enhanced = self.ema_alpha * enhanced + (1 - self.ema_alpha) * self._prev_frame
            self._prev_frame = enhanced
    
    return enhanced
```

### 5. Training Script Updates

**File**: `ml_codec/tools/train_residual_espcn.py` (modify)

Add QAT support:

- Add `--use-qat` argument
- Add QAT preparation after model creation
- Handle QAT-specific optimizer settings (may need different learning rate)
- Save quantized model state in checkpoint
- Add validation to compare FP32 vs INT8 accuracy

**Optional: Temporal Consistency Loss** (for future if EMA isn't sufficient)

Add `--temporal-loss-weight` parameter:

- Load consecutive frames during training
- Compute loss between current and previous frame predictions
- Encourage smooth temporal transitions
- Only add if EMA smoothing proves insufficient

### 6. Model Export (Future: ONNX INT8)

**File**: `ml_codec/tools/export_model.py` (new, for later)

Create export tool for ONNX INT8:

- Load PyTorch QAT model
- Export to ONNX format
- Apply ONNX Runtime quantization (INT8)
- Validate accuracy preservation
- Save optimized ONNX model

## Implementation Order

1. **Dataset generation tool** - Critical for training
2. **Augmentation module** - Needed for dataset generation
3. **Temporal smoothing (EMA)** - Inference-time improvement
4. **PyTorch QAT integration** - Training optimization
5. **ONNX export** - Future optimization (after QAT validation)

## Key Design Decisions

- **Dataset**: LR = decoded frames at various bitrates, HR = original frames
- **Augmentations**: Applied during dataset generation, not training (to keep training deterministic)
- **QAT**: PyTorch native first, ONNX later for cross-platform
- **Temporal smoothing**: EMA at inference time, avoid recurrent architecture initially
- **Frame matching**: Use filename stems (already implemented in `LRHRDataset`)

## Files to Create/Modify

**New files**:

- `ml_codec/tools/generate_training_dataset.py`
- `ml_codec/augmentations/__init__.py`
- `ml_codec/augmentations/compression_augmentations.py`

**Modified files**:

- `ml_codec/tools/train_residual_espcn.py` - Add QAT support
- `ml_codec/models/residual_espcn.py` - Add QAT compatibility and temporal smoothing
- `requirements.txt` - Add any QAT dependencies if needed

## Testing Strategy

1. Generate small test dataset (100 frames) to validate pipeline
2. Train small model with QAT to verify quantization works
3. Compare FP32 vs INT8 model quality (PSNR/SSIM)
4. Test temporal smoothing on video sequences
5. Measure inference latency improvements from quantization
