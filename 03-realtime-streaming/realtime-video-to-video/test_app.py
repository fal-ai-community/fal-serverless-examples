"""Tests for the realtime video-to-video example."""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

_spec = importlib.util.spec_from_file_location(
    "realtime_v2v_app", Path(__file__).parent / "app.py"
)
assert _spec is not None and _spec.loader is not None
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


# A structurally valid WebRTC offer with dummy ICE/DTLS credentials: enough
# for the server to parse it and produce a real answer, without any media.
FAKE_SDP = (
    "\r\n".join(
        [
            "v=0",
            "o=- 3700000000 3700000000 IN IP4 0.0.0.0",
            "s=-",
            "t=0 0",
            "a=group:BUNDLE 0",
            "a=msid-semantic:WMS *",
            "m=video 9 UDP/TLS/RTP/SAVPF 96",
            "c=IN IP4 0.0.0.0",
            "a=mid:0",
            "a=sendrecv",
            "a=rtcp-mux",
            "a=ice-ufrag:smoketest",
            "a=ice-pwd:smoketestsmoketestsmoke",
            "a=fingerprint:sha-256 " + ":".join(["AB"] * 32),
            "a=setup:actpass",
            "a=rtpmap:96 VP8/90000",
            "a=ssrc:1 cname:smoke",
            "a=msid:smoke smokevideo",
        ]
    )
    + "\r\n"
)


# --- signaling message contract (what fal's runtime does with client frames) --


def test_offer_parses_the_way_the_runtime_instantiates_it():
    message = app_module.RealtimeInput(**{"type": "offer", "sdp": FAKE_SDP})
    assert isinstance(message.root, app_module.OfferInput)
    assert message.root.sdp == FAKE_SDP


def test_icecandidate_parses_with_and_without_candidate():
    with_candidate = app_module.RealtimeInput(
        **{
            "type": "icecandidate",
            "candidate": {"candidate": "candidate:1 1 udp 1 1.2.3.4 5 typ host"},
        }
    )
    assert isinstance(with_candidate.root, app_module.IceCandidateInput)
    assert with_candidate.root.candidate.sdpMid is None

    end_of_candidates = app_module.RealtimeInput(
        **{"type": "icecandidate", "candidate": None}
    )
    assert end_of_candidates.root.candidate is None


def test_unknown_message_type_is_rejected():
    with pytest.raises(ValidationError):
        app_module.RealtimeInput(**{"type": "bogus"})


def test_output_messages_serialize_to_plain_dicts():
    answer = app_module.RealtimeOutput(
        root=app_module.AnswerOutput(type="answer", sdp="v=0")
    )
    assert answer.dict() == {"type": "answer", "sdp": "v=0"}


# --- ICE server resolution ----------------------------------------------------


def test_ice_falls_back_to_stun_without_turn_secrets():
    assert app_module.resolve_ice_servers(None, None) == app_module.STUN_SERVERS


def test_ice_requires_both_turn_secrets():
    assert app_module.resolve_ice_servers("secret", None) == app_module.STUN_SERVERS
    assert app_module.resolve_ice_servers(None, "label") == app_module.STUN_SERVERS


def test_stun_fallback_returns_a_fresh_copy():
    servers = app_module.resolve_ice_servers(None, None)
    servers[0]["urls"] = "tampered"
    assert app_module.STUN_SERVERS[0]["urls"] != "tampered"


# --- frame annotation -----------------------------------------------------------


def _fake_detections(np):
    box = SimpleNamespace(
        xyxy=[np.array([10.0, 10.0, 60.0, 60.0])],
        cls=[np.array(0)],
        conf=[np.array(0.9)],
    )
    return SimpleNamespace(boxes=[box], names={0: "person"})


def test_draw_boxes_marks_the_frame():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    annotated = app_module.draw_boxes(frame, _fake_detections(np))
    assert annotated.shape == (120, 160, 3)
    assert annotated.sum() > 0  # the green box actually landed


def test_draw_boxes_with_no_detections_is_a_noop():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    empty = SimpleNamespace(boxes=[], names={})
    assert app_module.draw_boxes(frame, empty).sum() == 0


def test_loopback_media_pipeline_annotates_real_rtp_frames():
    """Two in-process peers over localhost: frames must come back with boxes.
    Would catch e.g. addTrack racing createAnswer (the answer goes recvonly
    and no video ever returns)."""
    import asyncio

    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    aiortc = pytest.importorskip("aiortc")

    class StubYolo:
        def __call__(self, image, verbose=False, **kwargs):
            return [_fake_detections(np)]

    from aiortc.mediastreams import VideoStreamTrack

    class GrayTrack(VideoStreamTrack):
        async def recv(self):
            from av import VideoFrame

            pts, time_base = await self.next_timestamp()
            frame = VideoFrame.from_ndarray(
                np.full((240, 320, 3), 64, dtype=np.uint8), format="bgr24"
            )
            frame.pts = pts
            frame.time_base = time_base
            return frame

    async def loopback():
        client, server = aiortc.RTCPeerConnection(), aiortc.RTCPeerConnection()
        received: asyncio.Queue = asyncio.Queue()
        readers = []

        @server.on("track")
        def on_server_track(track):
            server.addTrack(app_module.create_yolo_track(track, StubYolo()))

        @client.on("track")
        def on_client_track(track):
            async def read():
                while True:
                    frame = await track.recv()
                    await received.put(frame.to_ndarray(format="bgr24"))

            readers.append(asyncio.ensure_future(read()))

        client.addTrack(GrayTrack())
        await client.setLocalDescription(await client.createOffer())
        await server.setRemoteDescription(client.localDescription)
        await server.setLocalDescription(await server.createAnswer())
        assert "a=sendrecv" in server.localDescription.sdp
        await client.setRemoteDescription(server.localDescription)
        try:
            # Skip the first frames: keyframes at session start may pre-date
            # the annotation settling through the encoder.
            for _ in range(5):
                frame = await asyncio.wait_for(received.get(), timeout=15)
            return frame
        finally:
            for reader in readers:
                reader.cancel()
            await client.close()
            await server.close()

    frame = asyncio.run(asyncio.wait_for(loopback(), timeout=45))
    assert frame.shape == (240, 320, 3)
    green = frame[:, :, 1].astype(int) - frame[:, :, 0].astype(int)
    assert green.max() > 60, "no green detection box survived the round-trip"


# Live tests below need an ephemeral deploy: app_client skips without HARNESS_APP_URL.


def _fal_headers():
    from fal.sdk import get_credentials

    return get_credentials().to_headers()


def test_detect_finds_the_canonical_objects(app_client):
    out = app_client.run("/", {"image_url": app_module.SAMPLE_IMAGE})
    labels = {d["label"] for d in out["detections"]}
    assert "person" in labels and "bus" in labels, f"got {labels}"
    for detection in out["detections"]:
        assert 0.25 <= detection["confidence"] <= 1.0
        x1, y1, x2, y2 = detection["box"]
        assert x1 < x2 and y1 < y2

    response = httpx.get(out["image"]["url"], timeout=60)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_detect_rejects_bad_image_before_inference(app_client):
    response = httpx.post(
        app_client.url + "/",
        json={"image_url": "https://storage.googleapis.com/falserverless/nope.png"},
        headers=_fal_headers(),
        timeout=120,
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "image_url"]


def test_realtime_signaling_full_session(app_client):
    """Signaling handshake + trickle-ICE leg over the realtime WebSocket.
    No media flows; every message the protocol defines gets exercised."""
    # Plain imports on purpose: both ship inside fal's pinned dependency
    # tree, and a broken harness env must fail loudly, not skip silently.
    import msgpack
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    url = app_client.url.replace("https://", "wss://", 1) + "/realtime"

    def run_session() -> dict:
        with connect(url, additional_headers=_fal_headers(), open_timeout=60) as ws:

            def send(payload: dict) -> None:
                ws.send(msgpack.packb(payload, use_bin_type=True))

            def next_message() -> dict:
                while True:
                    frame = ws.recv(timeout=60)
                    if isinstance(frame, bytes):  # str frames are x-fal metadata
                        return msgpack.unpackb(frame, raw=False)

            hello = next_message()
            assert hello["type"] == "iceservers", hello
            assert hello["iceservers"] and hello["iceservers"][0]["urls"]

            send({"type": "offer", "sdp": FAKE_SDP})
            answer = next_message()
            if answer["type"] != "answer":
                return answer

            # Trickle a host candidate, then both end-of-candidates forms
            # browsers use (empty candidate string, then null).
            send(
                {
                    "type": "icecandidate",
                    "candidate": {
                        "candidate": (
                            "candidate:1 1 udp 2130706431 192.0.2.10 51000 typ host"
                        ),
                        "sdpMid": "0",
                        "sdpMLineIndex": 0,
                    },
                }
            )
            send({"type": "icecandidate", "candidate": {"candidate": ""}})
            send({"type": "icecandidate", "candidate": None})
            # Silence is success: an ice_failed error (or a close) means the
            # server choked on a message every real browser sends.
            with pytest.raises(TimeoutError):
                next_frame = ws.recv(timeout=3)
                raise AssertionError(f"unexpected frame after ICE: {next_frame!r}")
            return answer

    # Fresh runners can accept the WS upgrade and close it (1000) before the
    # realtime session is wired up; real clients reconnect, so does this test.
    for attempt in range(5):
        try:
            answer = run_session()
            break
        except ConnectionClosed:
            if attempt == 4:
                raise
            time.sleep(3)

    assert answer["type"] == "answer", answer
    assert "m=video" in answer["sdp"]
    assert "a=candidate" in answer["sdp"]  # aiortc answers with ICE inline
    # sendrecv proves the YOLO return track made it into the answer - a
    # recvonly answer means addTrack lost a race with createAnswer.
    assert "a=sendrecv" in answer["sdp"]
