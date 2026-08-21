"""Smoke test - run by the harness against the live ephemeral app."""
import httpx
import pytest

SAMPLE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"


def test_processes_to_requested_format(app_client):
    response = app_client.run("/", {"image_url": SAMPLE, "max_size": 512})
    assert response["image"]["content_type"] == "image/webp"  # default output_format
    assert max(response["image"]["width"], response["image"]["height"]) <= 512
    assert max(response["thumbnail"]["width"], response["thumbnail"]["height"]) <= 256
    assert response["report"]["content_type"] == "application/json"


def test_output_format_sets_content_type_and_cache_hits(app_client):
    # first call primes the cache so the repeat hits it regardless of test ordering
    app_client.run("/", {"image_url": SAMPLE, "output_format": "jpeg"})
    response = app_client.run("/", {"image_url": SAMPLE, "output_format": "jpeg"})
    assert response["image"]["content_type"] == "image/jpeg"
    report = httpx.get(response["report"]["url"], timeout=30).json()
    assert report["source"]["cache_hit"] is True
    assert report["download_ms"] < 2000
    assert "upload_ms" in report


def test_sync_mode_returns_data_uris(app_client):
    response = app_client.run("/", {"image_url": SAMPLE, "sync_mode": True})
    assert response["image"]["url"].startswith("data:image/")
    assert response["report"]["url"].startswith("data:application/json")


def test_bad_url_is_a_typed_422(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {"image_url": "https://storage.googleapis.com/falserverless/does-not-exist.png"},
        )
    assert excinfo.value.response.status_code == 422
    detail = excinfo.value.response.json()["detail"]
    assert detail[0]["loc"] == ["body", "image_url"]
    assert detail[0]["type"] == "file_download_error"
