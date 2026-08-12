"""Tests for 04.7 - 3D progressive rendering.

Two kinds, both run by CI:
  * unit tests over the pure encoding/SSE helpers - free, no deploy, run on
    every PR by the static job (numpy-dependent ones skip when numpy is
    absent);
  * live tests that consume the real SSE stream on an ephemeral deploy
    (harness/check.py).
"""
import base64
import importlib.util
import json
import queue
import threading
import time
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location(
    "sam3d_stream_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


# --- decode_voxels: pure stdlib, runs everywhere -------------------------

def test_decode_voxels_roundtrips_handbuilt_payload():
    raw = bytes([1, 2, 3, 10, 20, 30, 63, 0, 63, 255, 128, 0])
    rows = app_module.decode_voxels(base64.b64encode(raw).decode())
    assert rows == [(1, 2, 3, 10, 20, 30), (63, 0, 63, 255, 128, 0)]


def test_decode_voxels_rejects_ragged_payload():
    with pytest.raises(ValueError):
        app_module.decode_voxels(base64.b64encode(b"\x01\x02\x03\x04").decode())


# --- encode_voxels: numpy path (skips on static CI without numpy) --------

def test_encode_decode_roundtrip_with_colors():
    np = pytest.importorskip("numpy")
    coords = np.array([[0, 0, 0], [63, 63, 63]], dtype=np.int64)
    colors = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=np.float32)
    rows = app_module.decode_voxels(app_module.encode_voxels(coords, colors))
    assert rows[0] == (0, 0, 0, 0, 128, 255)
    assert rows[1] == (63, 63, 63, 255, 0, 64)


def test_encode_voxels_gray_fill_without_colors():
    np = pytest.importorskip("numpy")
    rows = app_module.decode_voxels(
        app_module.encode_voxels(np.array([[5, 6, 7]]), None)
    )
    assert rows == [(5, 6, 7, 128, 128, 128)]


def test_encode_voxels_clips_out_of_range_colors():
    np = pytest.importorskip("numpy")
    rows = app_module.decode_voxels(
        app_module.encode_voxels(np.array([[1, 1, 1]]), np.array([[-0.5, 2.0, 1.0]]))
    )
    assert rows == [(1, 1, 1, 0, 255, 255)]


# --- SSE framing -----------------------------------------------------------

def test_sse_format_is_one_data_frame():
    frame = app_module.sse_format({"type": "status", "progress": 0.5})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):]) == {"type": "status", "progress": 0.5}


def test_heartbeat_is_a_comment_line():
    assert app_module.SSE_HEARTBEAT.startswith(":")
    assert app_module.SSE_HEARTBEAT.endswith("\n\n")


# --- queue semantics: progress disposable, terminal guaranteed -------------

def test_put_progress_drops_oldest_when_full():
    q = queue.Queue(maxsize=2)
    for i in range(4):
        app_module.put_progress(q, {"type": "geometry", "step": i})
    steps = [q.get_nowait()["step"] for _ in range(2)]
    assert steps == [2, 3]  # oldest frames were dropped, newest survive


def test_put_terminal_evicts_progress_to_guarantee_delivery():
    q = queue.Queue(maxsize=1)
    app_module.put_progress(q, {"type": "geometry", "step": 0})
    app_module.put_terminal(q, {"type": "complete"})
    assert q.get_nowait() == {"type": "complete"}


# --- stream_frames: heartbeat, sentinel, dead-worker taxonomy --------------

def test_stream_frames_yields_events_then_stops_at_sentinel():
    q = queue.Queue()
    done = threading.Event()
    q.put({"type": "status"})
    q.put({"type": "complete"})
    q.put(None)
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    assert [json.loads(f[6:]) for f in frames] == [
        {"type": "status"}, {"type": "complete"},
    ]


def test_stream_frames_emits_heartbeat_on_silence():
    q = queue.Queue()
    done = threading.Event()

    def finish_later():
        time.sleep(0.15)
        q.put(None)

    threading.Thread(target=finish_later).start()
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    assert app_module.SSE_HEARTBEAT in frames


def test_stream_frames_reports_dead_worker_without_sentinel():
    q = queue.Queue()
    done = threading.Event()
    done.set()  # worker exited without ever enqueueing the sentinel
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    last = json.loads(frames[-1][6:])
    assert last["type"] == "error"
    assert last["code"] == "worker_died"


# --- live tests (harness only: app_client skips without HARNESS_APP_URL) ---

SAMPLE = app_module.SAMPLE_IMAGE


def _fal_headers():
    from fal.sdk import get_credentials
    return get_credentials().to_headers()


def test_generate_returns_valid_assets(app_client):
    out = app_client.run("/", {"image_url": SAMPLE, "seed": 42})
    assert out["seed"] == 42
    glb_head = b""
    with httpx.stream("GET", out["model_glb"]["url"], timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            glb_head += chunk
            if len(glb_head) >= 4:
                break
    assert glb_head[:4] == b"glTF"
    assert httpx.head(out["gaussian_splat"]["url"], timeout=60).status_code < 400


def test_stream_event_contract(app_client):
    """The streaming contract: monotonic progress, geometry strictly before
    appearance, decodable voxel frames, exactly one terminal event carrying
    asset URLs."""
    events = []
    with httpx.stream(
        "POST", app_client.url + "/stream",
        json={"image_url": SAMPLE, "seed": 42, "geometry_every": 5, "colors_every": 10},
        headers=_fal_headers(), timeout=600,
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    terminals = [t for t in types if t in ("complete", "error")]
    assert terminals == ["complete"], f"stream ended badly: {types}"
    assert types[-1] == "complete"

    geometry = [e for e in events if e["type"] == "geometry"]
    appearance = [e for e in events if e["type"] == "appearance"]
    assert geometry and appearance
    last_geometry = max(i for i, t in enumerate(types) if t == "geometry")
    assert last_geometry < types.index("appearance")

    progresses = [e["progress"] for e in events if "progress" in e]
    assert progresses == sorted(progresses), "progress went backwards"

    frame = geometry[-1]
    rows = app_module.decode_voxels(frame["voxels"])
    assert len(rows) == frame["count"] > 0
    assert all(0 <= r[i] < frame["grid_size"] for r in rows[:100] for i in range(3))

    final = events[-1]
    with httpx.stream("GET", final["model_glb"]["url"], timeout=60) as r:
        r.raise_for_status()
        assert next(r.iter_bytes(4))[:4] == b"glTF"


def test_stream_rejects_bad_image_before_streaming(app_client):
    resp = httpx.post(
        app_client.url + "/stream",
        json={"image_url": "https://storage.googleapis.com/falserverless/does-not-exist.png"},
        headers=_fal_headers(), timeout=120,
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "image_url"]
