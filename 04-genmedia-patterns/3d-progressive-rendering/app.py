"""3D progressive rendering: stream SAM-3D reconstruction as it happens.

Catalog slot 04.7 (SERV-1319). One H100 app, two endpoints: `/` returns the
final GLB + Gaussian splat as URLs; `/stream` emits SSE while the two
diffusion stages run - voxels appear live, then the final assets arrive by
URL. The pattern to steal: model progress -> bounded queue -> SSE generator,
where progress frames are DISPOSABLE (drop-oldest) and the single terminal
event is GUARANTEED.
"""
import base64
import json
import queue
import secrets
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import FAL_PERSISTENT_DIR, File
from fal.toolkit import Image as FalImage
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Stage-1 voxel coords are indices in a fixed 64-cube grid, so uint8 is
# LOSSLESS - no per-frame renormalization, no jitter between frames.
GRID_SIZE = 64
HEARTBEAT_SECONDS = 15.0  # well under typical 60-100s proxy idle timeouts
SSE_HEARTBEAT = ": keepalive\n\n"


class GenerationCancelled(Exception):
    """Raised inside the worker when the client has disconnected."""


def encode_voxels(coords, colors=None) -> str:
    """Pack voxels as base64 of N x [x,y,z,r,g,b] uint8 rows (6 bytes/voxel).

    coords: (N,3) integer grid indices in [0, GRID_SIZE).
    colors: (N,3) floats in [0,1], or None for gray. The float->uint8 scaling
    happens HERE and only here - one encoder, one color contract.
    """
    import numpy as np

    coords = np.asarray(coords)
    packed = np.empty((len(coords), 6), dtype=np.uint8)
    packed[:, :3] = coords.astype(np.uint8)
    if colors is None:
        packed[:, 3:] = 128
    else:
        colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
        packed[:, 3:] = np.round(colors * 255).astype(np.uint8)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_voxels(payload: str) -> list[tuple[int, int, int, int, int, int]]:
    """Reference decoder (pure stdlib) - what a client does with `voxels`."""
    raw = base64.b64decode(payload)
    if len(raw) % 6:
        raise ValueError(f"voxel payload length {len(raw)} is not a multiple of 6")
    return [tuple(raw[i : i + 6]) for i in range(0, len(raw), 6)]


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def put_progress(q: queue.Queue, event: dict) -> None:
    """Enqueue a disposable progress frame. A slow client must never stall
    the GPU: when the queue is full, drop the OLDEST frame and move on."""
    while True:
        try:
            q.put_nowait(event)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def put_terminal(q: queue.Queue, event: dict | None) -> None:
    """Enqueue a terminal event (or the None end-of-stream sentinel) with
    guaranteed delivery, evicting progress frames if the queue is full."""
    while True:
        try:
            q.put(event, timeout=0.1)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def stream_frames(q: queue.Queue, worker_done: threading.Event,
                  heartbeat_s: float = HEARTBEAT_SECONDS):
    """Drain the queue into SSE frames. Heartbeats are SSE comments (clients'
    event parsers never see them). Ends at the None sentinel; if the worker
    died without one, emit a terminal error so 'finished' is never ambiguous
    with 'connection died'."""
    while True:
        try:
            event = q.get(timeout=heartbeat_s)
        except queue.Empty:
            if worker_done.is_set() and q.empty():
                yield sse_format({
                    "type": "error", "code": "worker_died",
                    "message": "generation worker exited unexpectedly",
                })
                return
            yield SSE_HEARTBEAT
            continue
        if event is None:
            return
        yield sse_format(event)
