"""Tests for the input gauntlet.

Two kinds, both run by CI:
  * unit tests over the pure gate functions - free, no deploy, run on every
    PR by the static job;
  * live tests that fire hostile payloads at a real ephemeral deploy and
    assert the typed 422 that comes back (harness/check.py).
"""
import base64
import importlib.util
import io
import zipfile
from pathlib import Path

import httpx
import pytest
from fal.exceptions import FieldException

SAMPLE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"
# ~5.9 MB, comfortably over a 1 MB cap. The download gate runs before the
# format check, so a non-image is fine here.
BIG_FILE = "https://storage.googleapis.com/falserverless/model_weights/flux2/flux2-tiny-vae.safetensors"

_spec = importlib.util.spec_from_file_location(
    "bulletproof_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


def data_uri(payload: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(payload).decode()}"


def pixel_bomb_uri() -> str:
    """~400 KB on the wire, ~430 MB decoded. The gate must catch it by
    header dimensions alone."""
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = None  # let the fixture exist; the app is what we test
    buf = io.BytesIO()
    PILImage.new("RGB", (12000, 12000), (255, 255, 255)).save(
        buf, format="PNG", compress_level=9
    )
    return data_uri(buf.getvalue(), "image/png")


def zip_bomb_uri(members: int = 1, member_mb: int = 200) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for i in range(members):
            archive.writestr(f"bomb{i}.bin", b"\0" * (member_mb * 1024 * 1024))
    return data_uri(buf.getvalue(), "application/zip")


# --- unit tests: the gates themselves (no deploy, no network) --------------

def test_scheme_policy_rejects_file_urls():
    with pytest.raises(FieldException) as excinfo:
        app_module.check_url_policy("file:///etc/passwd", "image_url")
    assert excinfo.value.type == "value_error"
    assert excinfo.value.status_code == 422


def test_port_policy_rejects_odd_ports():
    with pytest.raises(FieldException) as excinfo:
        app_module.check_url_policy("http://example.com:6379/x.png", "image_url")
    assert "6379" in excinfo.value.message


def test_pixel_gate_rejects_bombs_and_passes_normal_images():
    with pytest.raises(FieldException) as excinfo:
        app_module.check_pixel_budget(12000, 12000, app_module.DEFAULT_MAX_PIXELS, "image_url")
    assert excinfo.value.type == "image_too_large"
    app_module.check_pixel_budget(1024, 768, app_module.DEFAULT_MAX_PIXELS, "image_url")


def test_archive_gates(tmp_path):
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("big.bin", b"\0" * (64 * 1024 * 1024))
    with pytest.raises(FieldException) as excinfo:
        app_module.inspect_zip(bomb, max_entries=100, max_uncompressed_mb=1, field="archive_url")
    assert excinfo.value.type == "file_too_large"

    stats = app_module.inspect_zip(bomb, max_entries=100, max_uncompressed_mb=128, field="archive_url")
    assert stats.entries == 1
    assert stats.compression_ratio > 100  # the bomb signature

    many = tmp_path / "many.zip"
    with zipfile.ZipFile(many, "w") as archive:
        for i in range(50):
            archive.writestr(f"f{i}.txt", b"")
    with pytest.raises(FieldException) as excinfo:
        app_module.inspect_zip(many, max_entries=10, max_uncompressed_mb=1, field="archive_url")
    assert excinfo.value.type == "archive_file_count_exceeds_maximum"

    with pytest.raises(FieldException) as excinfo:
        app_module.inspect_zip(tmp_path / "nope.zip", 10, 1, "archive_url")
    assert excinfo.value.type == "invalid_archive"


# --- live tests: hostile payloads against a real deploy --------------------

def test_valid_image_passes_every_gate(app_client):
    response = app_client.run("/", {"image_url": SAMPLE})
    assert response["accepted"] is True
    assert response["gates"] == ["url_policy", "download_cap", "format_allowlist", "pixel_budget", "decode"]
    assert (response["width"], response["height"]) == (768, 1024)


def assert_typed_422(exc_info, field: str, type_: str):
    response = exc_info.value.response
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", field]
    assert detail["type"] == type_
    return detail


def test_metadata_endpoint_is_refused(app_client):
    """The classic SSRF target. The SDK's client blocks it; we assert the
    caller gets a typed 422, not a 500."""
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": "http://169.254.169.254/latest/meta-data/"})
    assert_typed_422(excinfo, "image_url", "file_download_error")


def test_localhost_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": "http://127.0.0.1:8080/secrets"})
    # Port policy fires before the SDK's SSRF refusal, so the type is
    # value_error with the port in the message.
    detail = assert_typed_422(excinfo, "image_url", "value_error")
    assert "8080" in detail["msg"]


def test_file_scheme_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": "file:///etc/passwd"})
    assert_typed_422(excinfo, "image_url", "value_error")


def test_oversized_download_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": BIG_FILE, "max_download_mb": 1})
    assert_typed_422(excinfo, "image_url", "file_too_large")


def test_pixel_bomb_is_refused_before_decode(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": pixel_bomb_uri()})
    detail = assert_typed_422(excinfo, "image_url", "image_too_large")
    assert "12000x12000" in detail["msg"]


def test_zip_bomb_is_refused_before_extraction(app_client):
    # 200 KB on the wire, 200 MB extracted - refused against a 64 MB budget
    # without ever calling extractall().
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {
                "image_url": SAMPLE,
                "archive_url": zip_bomb_uri(),
                "max_uncompressed_mb": 64,
            },
        )
    detail = assert_typed_422(excinfo, "archive_url", "file_too_large")
    assert "200.0 MB" in detail["msg"]


def test_archive_within_budget_is_accepted(app_client):
    response = app_client.run(
        "/",
        {
            "image_url": SAMPLE,
            "archive_url": zip_bomb_uri(member_mb=8),
            "max_uncompressed_mb": 64,
        },
    )
    assert response["accepted"] is True
    assert "archive_gates" in response["gates"]
    assert response["archive"]["compression_ratio"] > 100


def test_native_pydantic_constraint_still_produces_a_documented_422(app_client):
    """Field constraints need no code and yield the same envelope shape."""
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_url": SAMPLE, "output_format": "tiff"})
    response = excinfo.value.response
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "output_format"]
