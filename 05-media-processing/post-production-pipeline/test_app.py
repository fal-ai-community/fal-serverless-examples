"""Tests for the post-production pipeline.

Three kinds, all run by CI:
  * unit tests over the pure helpers - free, no deploy, no ffmpeg binary
    needed (command builders return argv lists; nothing is executed);
  * live tests against a real ephemeral deploy (harness/check.py);
  * webhook-receiver verification tests (skipped unless `cryptography`
    is installed - it is a client-side dep, not an app dep).
"""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fal.exceptions import FieldException

_spec = importlib.util.spec_from_file_location(
    "postprod_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)

# Verified fal-hosted samples (research 2026-08-04): pika = 1920x1080 h264,
# 5.04s, no audio; seedance = 1280x720 h264 + AAC, 12.04s.
SAMPLE = app_module.SAMPLE_VIDEO
AUDIO_SAMPLE = (
    "https://storage.googleapis.com/falserverless/example_outputs/"
    "bytedance/seedance_2/output.mp4"
)
STILL_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"


# --- unit tests: pure helpers (no deploy, no network, no ffmpeg) -----------

def test_ladder_dedupes_sorts_and_skips_upscales():
    encode, skipped = app_module.normalize_ladder([480, 1080, 480, 2160], 1080)
    assert encode == [1080, 480]
    assert skipped == ["2160p (source is 1080p)"]


def test_ladder_all_upscale_is_a_typed_422():
    with pytest.raises(FieldException) as excinfo:
        app_module.normalize_ladder([2160, 1440], 720)
    assert excinfo.value.field == "ladder"
    assert excinfo.value.type == "nothing_to_encode"
    with pytest.raises(FieldException):
        app_module.normalize_ladder([], 1080)


def test_thumbnail_timestamps_dodge_the_edges():
    assert app_module.thumbnail_timestamps(10.0, 3) == [2.5, 5.0, 7.5]
    assert app_module.thumbnail_timestamps(10.0, 1) == [5.0]


def test_select_captions_accepts_at_most_one_source():
    assert app_module.select_captions(None, None) is None
    assert app_module.select_captions("1\n...", None) == "inline"
    assert app_module.select_captions(None, "https://x/y.srt") == "url"
    with pytest.raises(FieldException) as excinfo:
        app_module.select_captions("1\n...", "https://x/y.srt")
    assert excinfo.value.field == "captions_srt"


def _probe(streams, duration="5.0"):
    fmt = {"duration": duration} if duration is not None else {}
    return {"streams": streams, "format": fmt}


VIDEO_STREAM = {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080}


def test_parse_probe_reads_dims_duration_and_audio():
    info = app_module.parse_probe(
        _probe([VIDEO_STREAM, {"codec_type": "audio", "codec_name": "aac"}])
    )
    assert (info.width, info.height, info.has_audio) == (1920, 1080, True)
    assert app_module.parse_probe(_probe([VIDEO_STREAM])).has_audio is False


def test_parse_probe_rejects_stills_and_marathons():
    # A still image probes as a video stream with no duration.
    with pytest.raises(FieldException) as excinfo:
        app_module.parse_probe(_probe([VIDEO_STREAM], duration=None))
    assert excinfo.value.type == "invalid_video"
    with pytest.raises(FieldException) as excinfo:
        app_module.parse_probe(_probe([VIDEO_STREAM], duration="4000"))
    assert excinfo.value.type == "video_too_long"
    with pytest.raises(FieldException):
        app_module.parse_probe(_probe([{"codec_type": "audio"}]))


def test_rendition_cmd_toggles_audio_and_captions():
    full = app_module.build_rendition_cmd(
        Path("in.mp4"), Path("subs.srt"), 720, True, Path("out.mp4")
    )
    assert full[full.index("-vf") + 1] == "scale=-2:720"
    for flag in ("mov_text", "+faststart", "yuv420p", "aac"):
        assert any(flag in arg for arg in full), f"missing {flag}"

    bare = app_module.build_rendition_cmd(
        Path("in.mp4"), None, 480, False, Path("out.mp4")
    )
    joined = " ".join(bare)
    assert "aac" not in joined and "mov_text" not in joined and "subs" not in joined
    assert "+faststart" in joined  # streaming-readiness is unconditional


def test_run_ffmpeg_error_taxonomy_and_remaining_cmd_builders():
    # User-input failures become typed 422s; infra failures stay 500s.
    with pytest.raises(FieldException) as excinfo:
        app_module.run_ffmpeg(["false"], error_field="captions_srt")
    assert excinfo.value.field == "captions_srt"
    assert excinfo.value.type == "invalid_media"
    with pytest.raises(RuntimeError):
        app_module.run_ffmpeg(["false"])

    thumb = app_module.build_thumbnail_cmd(Path("in.mp4"), 2.5, 1280, Path("t.jpg"))
    assert thumb[thumb.index("-vf") + 1] == "scale=1280:-2"
    assert thumb[thumb.index("-ss") + 1] == "2.500"
    assert "-frames:v" in thumb

    vtt = app_module.build_vtt_cmd(Path("c.srt"), Path("c.vtt"))
    assert vtt[-2:] == ["c.srt", "c.vtt"]


# --- live tests: against a real ephemeral deploy (harness/check.py) --------

def assert_typed_422(exc_info, field: str, type_: str):
    response = exc_info.value.response
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", field]
    assert detail["type"] == type_
    return detail


def fetch(url: str) -> httpx.Response:
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response


def test_full_pipeline_ladder_captions_thumbnails(app_client):
    response = app_client.run(
        "/",
        {
            "video_url": SAMPLE,
            "ladder": [1080, 720, 480],
            "captions_srt": app_module.SAMPLE_SRT,
            "thumbnail_count": 2,
        },
    )
    renditions = response["renditions"]
    assert [r["label"] for r in renditions] == ["1080p", "720p", "480p"]
    assert [r["height"] for r in renditions] == [1080, 720, 480]
    for r in renditions:
        assert r["size_bytes"] > 0 and r["encode_ms"] > 0
        assert r["video"]["content_type"] == "video/mp4"

    # +faststart is the point: moov must come before mdat, or the CDN URL
    # can't stream. Verify on the smallest rendition's actual bytes.
    data = fetch(renditions[-1]["video"]["url"]).content
    moov, mdat = data.find(b"moov"), data.find(b"mdat")
    assert moov != -1, "no moov atom - not a valid MP4?"
    assert mdat == -1 or moov < mdat, "moov after mdat: faststart didn't apply"

    assert len(response["thumbnails"]) == 2
    for thumb in response["thumbnails"]:
        assert thumb["content_type"] == "image/jpeg"

    assert response["captions_vtt"] is not None
    assert "WEBVTT" in fetch(response["captions_vtt"]["url"]).text

    report = fetch(response["report"]["url"]).json()
    assert report["source"]["has_audio"] is False
    assert report["ladder"]["skipped_no_upscale"] == []
    assert report["renditions"][0]["captions_muxed"] is True
    assert report["toolchain"]["ffmpeg"].startswith("ffmpeg version")


def test_audio_source_transcodes_and_never_upscales(app_client):
    """720p source with AAC audio: 2160 must be skipped (never upscale),
    and the audio must survive into the rendition."""
    response = app_client.run(
        "/",
        {"video_url": AUDIO_SAMPLE, "ladder": [2160, 360], "thumbnail_count": 1},
    )
    assert [r["label"] for r in response["renditions"]] == ["360p"]
    assert len(response["thumbnails"]) == 1
    report = fetch(response["report"]["url"]).json()
    assert report["source"]["has_audio"] is True
    assert report["ladder"]["skipped_no_upscale"] == ["2160p (source is 720p)"]
    assert report["renditions"][0]["has_audio"] is True
    assert response["captions_vtt"] is None


def test_still_image_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"video_url": STILL_IMAGE})
    assert_typed_422(excinfo, "video_url", "invalid_video")


def test_both_caption_fields_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {
                "video_url": SAMPLE,
                "captions_srt": app_module.SAMPLE_SRT,
                "captions_url": "https://example.com/captions.srt",
            },
        )
    assert_typed_422(excinfo, "captions_srt", "value_error")


def test_garbage_captions_are_refused_before_encoding(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {"video_url": SAMPLE, "captions_srt": "this is not a subtitle file"},
        )
    assert_typed_422(excinfo, "captions_srt", "invalid_media")


# --- webhook receiver: signature verification (client-side dep) ------------
# The receiver is product-backend reference code, not part of the app image;
# its `cryptography` dep is not in requirements.txt, so skip when absent.

crypto = pytest.importorskip("cryptography")


def _receiver_module():
    spec = importlib.util.spec_from_file_location(
        "postprod_receiver", Path(__file__).parent / "webhook_receiver.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signed_delivery(timestamp: int | None = None):
    """Forge a delivery exactly the way fal's dispatcher signs one."""
    import hashlib as _hashlib
    import time as _time

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    body = json.dumps(
        {"request_id": "req-1", "gateway_request_id": "req-1",
         "status": "OK", "error": None, "payload": {"renditions": []}},
        sort_keys=True,
    ).encode()
    ts = str(timestamp if timestamp is not None else int(_time.time()))
    message = "\n".join(
        ["req-1", "user-1", ts, _hashlib.sha256(body).hexdigest()]
    ).encode()
    headers = {
        "x-fal-webhook-request-id": "req-1",
        "x-fal-webhook-user-id": "user-1",
        "x-fal-webhook-timestamp": ts,
        "x-fal-webhook-signature": private.sign(message).hex(),
    }
    return headers, body, [private.public_key()]


def test_receiver_accepts_a_correctly_signed_delivery():
    receiver = _receiver_module()
    headers, body, keys = _signed_delivery()
    ok, reason = receiver.verify(headers, body, keys)
    assert ok, reason


def test_receiver_rejects_tampered_bodies_and_stale_timestamps():
    receiver = _receiver_module()

    headers, body, keys = _signed_delivery()
    ok, reason = receiver.verify(headers, body + b" ", keys)
    assert not ok and "signature" in reason

    stale_headers, stale_body, stale_keys = _signed_delivery(timestamp=1_000_000)
    ok, reason = receiver.verify(stale_headers, stale_body, stale_keys)
    assert not ok and "timestamp" in reason

    missing = dict(headers)
    del missing["x-fal-webhook-signature"]
    ok, reason = receiver.verify(missing, body, keys)
    assert not ok and "header" in reason
