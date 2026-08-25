"""Headless WebRTC client: synthetic video in, annotated-frame stats out."""

import argparse
import asyncio
import contextlib
import time
import urllib.request
from pathlib import Path

import cv2
import msgpack
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
from websockets.asyncio.client import connect

SAMPLE_IMAGE = "https://ultralytics.com/images/bus.jpg"


class PanningImageTrack(VideoStreamTrack):
    """A camera stand-in: pans a crop window across a real photo at ~30 fps,
    so the returned stream shows boxes tracking moving objects."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self.width, self.height = width, height
        raw = urllib.request.urlopen(SAMPLE_IMAGE, timeout=30).read()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        scale = (height * 2) / image.shape[0]
        self.canvas = cv2.resize(image, None, fx=scale, fy=scale)
        self.step = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        max_x = self.canvas.shape[1] - self.width
        max_y = self.canvas.shape[0] - self.height
        phase = (self.step % 200) / 200.0
        bounce = 2 * phase if phase < 0.5 else 2 - 2 * phase
        x = int(max_x * bounce)
        y = int(max_y * bounce)
        self.step += 1
        frame = VideoFrame.from_ndarray(
            self.canvas[y : y + self.height, x : x + self.width], format="bgr24"
        )
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def run(url: str, seconds: float, width: int, height: int, save_dir: str | None):
    from fal.sdk import get_credentials

    ws_url = url.rstrip("/")
    if not ws_url.endswith("/realtime"):
        ws_url += "/realtime"
    ws_url = ws_url.replace("https://", "wss://", 1)

    received = 0
    first_frame_at: float | None = None
    connected_at: float | None = None
    done = asyncio.Event()
    background: list[asyncio.Task] = []
    pc: RTCPeerConnection | None = None

    async def read_frames(track) -> None:
        nonlocal received, first_frame_at
        while True:
            frame = await track.recv()
            received += 1
            if first_frame_at is None:
                first_frame_at = time.monotonic()
                since = first_frame_at - (connected_at or first_frame_at)
                print(f"first annotated frame after {since:.2f}s")
            if save_dir and (received == 1 or received % 30 == 0) and received <= 300:
                path = Path(save_dir) / f"frame_{received:04d}.png"
                cv2.imwrite(str(path), frame.to_ndarray(format="bgr24"))

    def create_peer(ice_servers: list[dict]) -> RTCPeerConnection:
        peer = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=server["urls"],
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                    for server in ice_servers
                ]
            )
        )

        @peer.on("track")
        def on_track(track) -> None:
            if track.kind == "video":
                background.append(asyncio.ensure_future(read_frames(track)))

        @peer.on("connectionstatechange")
        async def on_state() -> None:
            nonlocal connected_at
            print(f"peer connection: {peer.connectionState}")
            if peer.connectionState == "connected":
                connected_at = time.monotonic()
            if peer.connectionState in ("failed", "closed"):
                done.set()

        peer.addTrack(PanningImageTrack(width, height))
        return peer

    print(f"connecting to {ws_url}")
    async with connect(
        ws_url, additional_headers=get_credentials().to_headers(), open_timeout=60
    ) as ws:

        async def session() -> None:
            nonlocal pc
            async for raw in ws:
                if not isinstance(raw, bytes):
                    continue  # str frames are x-fal metadata
                msg = msgpack.unpackb(raw, raw=False)
                if msg["type"] == "iceservers":
                    print(f"ICE servers: {[s['urls'] for s in msg['iceservers']]}")
                    pc = create_peer(msg["iceservers"])
                    await pc.setLocalDescription(await pc.createOffer())
                    # aiortc bundles ICE candidates into the offer; no trickle.
                    await ws.send(
                        msgpack.packb(
                            {"type": "offer", "sdp": pc.localDescription.sdp},
                            use_bin_type=True,
                        )
                    )
                elif msg["type"] == "answer" and pc is not None:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg["sdp"], type="answer")
                    )
                elif msg["type"] == "error":
                    print(f"server error: {msg['error']}")
                    done.set()

        background.append(asyncio.create_task(session()))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=seconds)
        for task in background:
            task.cancel()
        if pc is not None:
            await pc.close()

    if received and first_frame_at is not None:
        elapsed = time.monotonic() - first_frame_at
        fps = (received - 1) / elapsed if elapsed > 0 else 0.0
        print(f"received {received} annotated frames, {fps:.1f} fps average")
        return 0
    print(
        "no annotated frames received - signaling worked but media did not flow; "
        "this usually means the NATs need a TURN relay (see README)"
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=True,
        help="app URL, e.g. https://fal.run/you/realtime-video-to-video",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="session length")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save-dir", default=None, help="dump annotated frames here")
    args = parser.parse_args()
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        run(args.url, args.seconds, args.width, args.height, args.save_dir)
    )


if __name__ == "__main__":
    raise SystemExit(main())
