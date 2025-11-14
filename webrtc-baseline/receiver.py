import argparse
import asyncio
import time
import struct
import statistics
import cv2
import numpy as np
from collections import deque

from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.contrib.signaling import TcpSocketSignaling
from aiortc.contrib.signaling import CopyAndPasteSignaling

AUX_STRUCT = struct.Struct("<I Q H H")  # frame_id, sender_ns, schema_id, count

class Stats:
    def __init__(self, window=120):
        self.lats = deque(maxlen=window)
        self.frame_times = deque(maxlen=window)
        self.last_frame_ts = None
        self.overlay_on = True
        self.last_frame_id = 0

    def update_latency(self, lat_ms: float):
        self.lats.append(lat_ms)

    def tick_fps(self):
        now = time.perf_counter()
        if self.last_frame_ts is not None:
            self.frame_times.append(now - self.last_frame_ts)
        self.last_frame_ts = now

    def fps(self):
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def p50(self):
        if not self.lats:
            return 0.0
        return statistics.median(self.lats)

    def p95(self):
        if not self.lats:
            return 0.0
        return float(np.percentile(self.lats, 95))
    
def parse_aux_packet(data: bytes):
    # Returns dict with frame_id, sender_ns
    frame_id, sender_ns, schema_id, count = AUX_STRUCT.unpack(data[:AUX_STRUCT.size])
    return {"frame_id": frame_id, "sender_ns": sender_ns, "schema_id": schema_id, "count": count}

async def run_receiver(host: str, port: int, jitter_ms: int):
    cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(cfg)
    # Signaling (TCP connection)
    # Note: TcpSocketSignaling automatically acts as client when receiving
    # Do NOT run signaling_server.py separately - it will conflict
    signaling = TcpSocketSignaling(host, port)
    #signaling = CopyAndPasteSignaling()
    await signaling.connect()  # This is a no-op for TcpSocketSignaling

    stats = Stats()
    aux_latest = {"frame_id": -1, "sender_ns": 0}

    @pc.on("datachannel")
    def on_dc(dc):
        if dc.label == "aux":
            @dc.on("message")
            def on_msg(data):
                try:
                    if isinstance(data, (bytes, bytearray)):
                        pkt = parse_aux_packet(data)
                        aux_latest.update(pkt)
                        now = time.monotonic_ns()
                        lat_ms = (now - pkt["sender_ns"]) / 1e6
                        stats.update_latency(lat_ms)
                except Exception as e:
                    print(f"[receiver] aux packet parse error: {e}")

    @pc.on("track")
    def on_track(track):
        print(f"[receiver] Track received: kind={track.kind}, id={track.id}")
        if track.kind == "video":
            print(f"[receiver] Starting video frame reader...")
            async def reader():
                cv2.namedWindow("receiver", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("receiver", 960, 540)
                
                last_frame_time = None
                frame_count = 0
                dropped_frames = 0
                start_time = time.perf_counter()
                frame_receive_times = []
                decode_times = []
                display_times = []
                
                while True:
                    try:
                        if frame_count == 0:
                            print(f"[receiver] Waiting for first frame...")
                        frame_start = time.perf_counter()
                        frame = await track.recv()
                        if frame is None:
                            print(f"[receiver] received None frame, skipping")
                            continue
                        if frame_count == 0:
                            print(f"[receiver] First frame received! Starting display...")
                        frame_receive_time = time.perf_counter() - frame_start
                        frame_receive_times.append(frame_receive_time)
                        if len(frame_receive_times) > 60:
                            frame_receive_times.pop(0)
                        
                        # If we're falling behind, drop frames to catch up
                        # Allow frames up to 40fps (25ms between frames), drop if too fast
                        if last_frame_time is not None:
                            elapsed = time.perf_counter() - last_frame_time
                            if elapsed < 0.025:  # Less than 25ms between frames (40fps threshold)
                                dropped_frames += 1
                                continue  # Skip this frame
                        
                        last_frame_time = time.perf_counter()
                        frame_count += 1
                        
                        # PROFILING: Measure decode time (PyAV frame to numpy array)
                        decode_start = time.perf_counter()
                        img = frame.to_ndarray(format="bgr24")
                        decode_time = time.perf_counter() - decode_start

                        # Update FPS stats
                        stats.tick_fps()

                        # PROFILING: Measure display/rendering time
                        display_start = time.perf_counter()
                        
                        # Overlay metrics (toggle with 'o')
                        if stats.overlay_on:
                            avg_receive_time = sum(frame_receive_times) / len(frame_receive_times) * 1000 if frame_receive_times else 0
                            text1 = f"FPS: {stats.fps():.1f}  p50: {stats.p50():.0f} ms  p95: {stats.p95():.0f} ms"
                            text2 = f"Frame: {frame_count}  Dropped: {dropped_frames}  Recv: {avg_receive_time:.1f}ms"
                            text3 = f"Aux frame_id: {aux_latest['frame_id']}"
                            text4 = f"Decode: {decode_time*1000:.1f}ms  Display: (measuring...)"
                            cv2.putText(img, text1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(img, text2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(img, text3, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(img, text4, (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.imshow("receiver", img)
                        # Non-blocking UI & hotkeys
                        key = cv2.waitKey(1) & 0xFF
                        
                        display_time = time.perf_counter() - display_start
                        
                        # Track profiling data
                        decode_times.append(decode_time * 1000)
                        display_times.append(display_time * 1000)
                        if len(decode_times) > 60:
                            decode_times.pop(0)
                            display_times.pop(0)
                        
                        if key == ord('q'):
                            break
                        elif key == ord('o'):
                            stats.overlay_on = not stats.overlay_on
                        
                        # Debug every 60 frames
                        if frame_count % 60 == 0:
                            elapsed_total = time.perf_counter() - start_time
                            overall_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                            avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
                            avg_display = sum(display_times) / len(display_times) if display_times else 0
                            print(f"[receiver] frames: {frame_count}, dropped: {dropped_frames}, overall_fps: {overall_fps:.2f}")
                            print(f"[receiver] PROFILING: receive={avg_receive_time:.2f}ms, decode={avg_decode:.2f}ms, display={avg_display:.2f}ms, total={(avg_receive_time+avg_decode+avg_display):.2f}ms")
                        
                    except Exception as e:
                        import traceback
                        print(f"[receiver] frame receive error: {e}")
                        print(f"[receiver] error type: {type(e).__name__}")
                        traceback.print_exc()
                        break

                cv2.destroyAllWindows()

            asyncio.ensure_future(reader())

    # As answerer: wait for offer, return answer
    # Note: Sender must start first and send offer (sender becomes server)
    print(f"[receiver] waiting for offer from sender...")
    max_retries = 30
    retry_count = 0
    offer = None
    while offer is None and retry_count < max_retries:
        try:
            offer = await asyncio.wait_for(signaling.receive(), timeout=1.0)
            break
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"[receiver] waiting for sender... (retry {retry_count}/{max_retries})")
                await asyncio.sleep(1)
            else:
                raise RuntimeError(f"Failed to connect to sender after {max_retries} retries. Make sure sender is running first.")
    
    if offer is None:
        raise RuntimeError("Failed to receive offer from sender. Make sure sender is running first.")
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(await pc.createAnswer())
    print(f"[receiver] sending answer...")
    await signaling.send(pc.localDescription)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[receiver] Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            print(f"[receiver] Connection failed/closed, cleaning up...")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            await signaling.close()
            await pc.close()

    print(f"[receiver] listening on {host}:{port}. Press 'o' to toggle overlay, 'q' to quit.")
    # keep running
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--jitter", type=int, default=60, help="(placeholder) ms")
    args = parser.parse_args()
    try:
        asyncio.run(run_receiver(args.host, args.port, args.jitter))
    except KeyboardInterrupt:
        pass