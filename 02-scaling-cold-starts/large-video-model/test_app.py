"""Smoke tests - run by the harness against the live ephemeral app.

'/' is the harness-smoked JSON endpoint; '/stream' is exercised with a raw
SSE client because the harness only speaks request/response.
"""
import base64
import json
import os

import httpx


def test_invoke_contract(app_client):
    response = app_client.run(
        "/", {"prompt": "a red fox trotting through fresh snow at golden hour"}
    )
    assert response["video"]["url"].startswith("https://")
    # The README documents this exact shape; File.from_path would otherwise
    # hand back application/octet-stream and the temp file's random name.
    assert response["video"]["content_type"] == "video/mp4"
    assert response["video"]["file_name"] == "output.mp4"
    assert isinstance(response["seed"], int)
    assert response["timings"]["generate_s"] > 0


def test_stream_previews(app_client):
    from fal.sdk import get_credentials

    timeout = float(os.environ.get("HARNESS_REQUEST_TIMEOUT", "120"))
    events = []
    with httpx.stream(
        "POST",
        app_client.url + "/stream",
        json={"prompt": "a red fox trotting through fresh snow at golden hour"},
        headers=get_credentials().to_headers(),
        timeout=timeout,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))

    previews = [e for e in events if e["type"] == "preview"]
    assert len(previews) >= 2, f"expected >=2 previews, got {len(previews)}"
    steps = [p["step"] for p in previews]
    assert steps == sorted(set(steps)), "preview steps must be strictly increasing"
    # Previews carry the same File-shaped `video` object as the terminal result
    # event (url/content_type/file_size), so clients parse one shape for both -
    # the url is a data URI instead of a CDN link.
    preview_video = previews[0]["video"]
    assert preview_video["url"].startswith("data:video/mp4;base64,")
    assert preview_video["content_type"] == "video/mp4"
    payload = base64.b64decode(preview_video["url"].split(",", 1)[1])
    assert b"ftyp" in payload[:64], "preview payload is not an mp4"
    assert preview_video["file_size"] == len(payload)
    assert events[-1]["type"] == "result"
    assert events[-1]["video"]["url"].startswith("https://")
    # Same contract as "/": from_path_async must carry the explicit
    # content_type/file_name too, or clients get application/octet-stream.
    assert events[-1]["video"]["content_type"] == "video/mp4"
    assert events[-1]["video"]["file_name"] == "output.mp4"
