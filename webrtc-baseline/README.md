# WebRTC Baseline Streaming

This is a baseline WebRTC video streaming setup for ML codec research.

## Setup

Make sure you have the virtual environment activated:
```bash
source venv/bin/activate  # On macOS/Linux
```

## Running

**IMPORTANT:** Do NOT run `signaling_server.py` separately! `TcpSocketSignaling` handles server/client roles automatically.

### Step 1: Start the Sender (in terminal 1)
```bash
python webrtc-baseline/sender.py --host 127.0.0.1 --port 9999 --cam 0 --fps 20 --bitrate 300
```

### Step 2: Start the Receiver (in terminal 2)
```bash
python webrtc-baseline/receiver.py --host 127.0.0.1 --port 9999
```

**Order matters:** Start sender first (it becomes the signaling server), then receiver (connects as client).

## Arguments

### sender.py
- `--host`: Signaling server host (default: 127.0.0.1)
- `--port`: Signaling server port (default: 9999)
- `--cam`: Camera index (default: 0)
- `--fps`: Target frames per second (default: 20)
- `--bitrate`: Target bitrate in kbps (default: 300)

### receiver.py
- `--host`: Signaling server host (default: 127.0.0.1)
- `--port`: Signaling server port (default: 9999)
- `--jitter`: Placeholder parameter (default: 60)

## Controls

In the receiver window:
- Press `o` to toggle stats overlay
- Press `q` to quit

## Performance Notes

**VP8 Codec Performance:**
- VP8 software decoding is slow (~540ms per frame on macOS)
- Results in ~1.8 fps instead of target 20 fps
- This is acceptable for ML codec research as we'll replace it with custom codecs
- H.264 hardware acceleration has frame transmission issues with aiortc

**For Production:**
- Use `--use-h264` flag (experimental, may not work)
- Or implement custom codec with hardware acceleration
- Or test on separate machines to reduce CPU contention

## Troubleshooting

1. **Camera not opening**: Try `--cam 1` and grant camera permissions to Terminal
2. **Connection errors**: Make sure receiver is started before sender  
3. **Port conflicts**: Change the port if 9999 is already in use
4. **Signaling errors**: Make sure `signaling_server.py` is NOT running
5. **Low FPS (~1.8)**: Normal for VP8 software decoder. Will improve with ML codec implementation

