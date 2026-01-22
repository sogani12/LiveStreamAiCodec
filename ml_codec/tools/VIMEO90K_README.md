# Vimeo-90K Dataset Download and Preparation

This guide explains how to download and prepare the Vimeo-90K dataset for training.

## Quick Start

```bash
# Download and prepare the dataset
python -m ml_codec.tools.download_vimeo90k \
    --download-dir datasets/vimeo90k \
    --prepare-videos

# Or if you already have the dataset downloaded
python -m ml_codec.tools.download_vimeo90k \
    --download-dir datasets/vimeo90k \
    --prepare-videos \
    --skip-download
```

## Manual Download

The Vimeo-90K dataset is large (~30GB for training set). The script will attempt automatic download, but you may need to download manually:

1. Visit: https://github.com/anchen1011/toflow/issues/5
2. Find download links in the comments
3. Download `vimeo_triplet.tar` (training set, ~30GB)
4. Extract to `datasets/vimeo90k/`
5. Run the preparation script with `--skip-download`

## Dataset Structure

Vimeo-90K contains:
- **90,000+ video sequences**
- **7 frames per sequence**
- **High quality** (448x256 to 1280x720)
- **Diverse content** (people, scenes, objects)

Original structure:
```
sequences/
  00001/
    0001/
      im1.png, im2.png, ..., im7.png
    0002/
      ...
  ...
```

After conversion to videos:
```
videos/
  00001_0001.mp4
  00001_0002.mp4
  ...
```

## Usage Options

### Download Only
```bash
python -m ml_codec.tools.download_vimeo90k \
    --download-dir datasets/vimeo90k
```

### Convert to Videos Only (if already downloaded)
```bash
python -m ml_codec.tools.download_vimeo90k \
    --download-dir datasets/vimeo90k \
    --skip-download \
    --output-videos-dir datasets/vimeo90k/videos
```

### Test with Small Subset
```bash
python -m ml_codec.tools.download_vimeo90k \
    --download-dir datasets/vimeo90k \
    --max-sequences 100 \
    --prepare-videos
```

## Generate Training Dataset

After preparing videos, generate LR-HR pairs:

```bash
python -m ml_codec.tools.generate_training_dataset \
    --input datasets/vimeo90k/videos \
    --output-dir datasets/train \
    --codec h264 \
    --bitrates 50,100,200,300 \
    --augment blocking,ringing,blur \
    --augmentation-intensity 0.3
```

## Storage Requirements

- **Downloaded dataset**: ~30GB (training set)
- **Converted videos**: ~10-15GB (MP4 format)
- **Training dataset (LR-HR pairs)**: ~50-100GB (depends on bitrates and augmentations)

Total: ~100-150GB for complete pipeline

## Tips

1. **Start small**: Use `--max-sequences 1000` to test the pipeline first
2. **Disk space**: Ensure you have enough space before downloading
3. **Time**: Video conversion can take several hours for full dataset
4. **Quality**: Vimeo-90K provides excellent HR targets for training

## Alternative Datasets

If Vimeo-90K is too large, consider:
- **REDS dataset**: Smaller, also high quality
- **Your own recordings**: Webcam videos for domain-specific training
- **YouTube videos**: Download high-quality videos (respect copyright)
