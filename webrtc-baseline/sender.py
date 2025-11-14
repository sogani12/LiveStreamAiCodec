import argparse
import asyncio
import time
import struct
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCRtpSender, MediaStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from aiortc.contrib.signaling import CopyAndPasteSignaling
from aiortc.codecs import vpx
from av import VideoFrame

AUX_STRUCT = struct.Struct("<I Q H H")  # frame_id, sender_ns, schema_id, count

def build_aux_packet(frame_id: int, sender_ns: int) -> bytes:
    # schema_id = 0 (mock), count = 0 (no points)
    return AUX_STRUCT.pack(frame_id, sender_ns, 0, 0)

class CameraTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, src=0, fps=20, width=640, height=360):
        super().__init__()  # don't forget this!
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.last_ts = time.perf_counter()
        self.start_ts = time.perf_counter()
        # Try to open camera
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open. Try --cam 1 and grant Camera access to your Terminal.")
        
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame_id = 0

    async def recv(self) -> VideoFrame:
        # pacing to target fps
        now = time.perf_counter()
        wait = self.frame_time - (now - self.last_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_ts = time.perf_counter()

        ok, frame = self.cap.read()
        if not ok:
            # If camera fails, send a blank frame to keep pipeline alive
            img = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            img = frame

        # BGR (OpenCV) -> VideoFrame
        video_frame = VideoFrame.from_ndarray(img, format="bgr24")
        video_frame.pts = None  # aiortc will set timestamps

        self.frame_id += 1
        
        # Debug: Print every 60 frames (every ~3 seconds at 20fps)
        if self.frame_id % 60 == 0:
            elapsed = time.perf_counter() - self.start_ts
            actual_fps = self.frame_id / elapsed if elapsed > 0 else 0
            print(f"[sender] frame_id={self.frame_id}, target_fps={self.fps}, actual_fps={actual_fps:.2f}")
        
        return video_frame

async def run_sender(host: str, port: int, cam: int, fps: int, bitrate_kbps: int, use_h264: bool = False):

    MAX_BPS = int(bitrate_kbps * 1000)
    
    vpx.DEFAULT_BITRATE = int(0.75*MAX_BPS)
    vpx.MIN_BITRATE = int(0.4*MAX_BPS) # Setting MIN high might force it up
    vpx.MAX_BITRATE = MAX_BPS

    # Basic ICE config (pure host candidates on loopback are fine)
    cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(cfg)

    # Signaling (TCP on localhost)
    # Note: TcpSocketSignaling automatically acts as server when sending
    # Do NOT run signaling_server.py separately - it will conflict
    signaling = TcpSocketSignaling(host, port)
    #signaling = CopyAndPasteSignaling()
    await signaling.connect()  # This is a no-op for TcpSocketSignaling

    # Create camera track
    track = CameraTrack(src=cam, fps=fps, width=640, height=360)
    #sender = pc.addTrack(track)

    # Aux data channel
    dc = pc.createDataChannel("aux")

    tx = pc.addTransceiver("video", direction="sendonly")

    # Codec selection: VP8 by default (H.264 has frame transmission issues with aiortc)
    caps = RTCRtpSender.getCapabilities("video")
    
    if use_h264:
        # Explicitly use H.264 (experimental - may not transmit frames properly)
        h264_codecs = [c for c in caps.codecs if c.mimeType.lower() == "video/h264"]
        if h264_codecs:
            print(f"[sender] Using H.264 codec (EXPERIMENTAL - may not work)")
            tx.setCodecPreferences(h264_codecs)
        else:
            print(f"[sender] H.264 not available, using VP8")
            vp8_only = [c for c in caps.codecs if c.mimeType.lower() == "video/vp8"]
            if vp8_only:
                tx.setCodecPreferences(vp8_only)
    else:
        # Default: VP8 (known working)
        print(f"[sender] Using VP8 codec (default)")
        vp8_only = [c for c in caps.codecs if c.mimeType.lower() == "video/vp8"]
        if vp8_only:
            tx.setCodecPreferences(vp8_only)

    # Attach the camera track using the documented API
    pc.addTrack(track)

    # Create & send offer
    await pc.setLocalDescription(await pc.createOffer())
    print(f"[sender] sending offer, waiting for receiver...")
    await signaling.send(pc.localDescription)

    # Wait for answer (this will connect as server if not already connected)
    print(f"[sender] waiting for answer...")
    answer = await signaling.receive()
    if answer is None:
        raise RuntimeError("Failed to receive answer from receiver")
    await pc.setRemoteDescription(answer)

    # Wait for data channel to be ready
    print(f"[sender] waiting for connection...")
    await asyncio.sleep(1)  # Give time for connection to establish
    timeout = 0
    while dc.readyState != "open" and timeout < 50:
        await asyncio.sleep(0.1)
        timeout += 1
    
    if dc.readyState != "open":
        print(f"[sender] WARNING: data channel did not open (state={dc.readyState})")
    else:
        print(f"[sender] data channel ready!")

    # Per-frame aux sender loop
    async def aux_loop():
        frame_id_snapshot = 0
        while True:
            await asyncio.sleep(0)  # yield to event loop
            try:
                # Grab frame_id from track; we don't want to block capture
                current_id = track.frame_id
                if current_id != frame_id_snapshot:
                    frame_id_snapshot = current_id
                    pkt = build_aux_packet(frame_id_snapshot, time.monotonic_ns())
                    if dc.readyState == "open":
                        dc.send(pkt)
            except Exception as e:
                print(f"[sender] aux_loop error: {e}")
                await asyncio.sleep(0.1)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[sender] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            print(f"[sender] Connection failed/closed, cleaning up...")
            await signaling.close()
            await pc.close()

    print(f"[sender] connecting to {host}:{port}, target ~{bitrate_kbps} kbps, {fps} fps at 360p")
    await asyncio.gather(aux_loop())  # runs until you Ctrl+C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--cam", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--bitrate", type=int, default=300, help="kbps")
    parser.add_argument("--use-h264", action="store_true", help="Use H.264 codec (EXPERIMENTAL, may not work)")
    args = parser.parse_args()
    try:
        asyncio.run(run_sender(args.host, args.port, args.cam, args.fps, args.bitrate, args.use_h264))
    except KeyboardInterrupt:
        pass
