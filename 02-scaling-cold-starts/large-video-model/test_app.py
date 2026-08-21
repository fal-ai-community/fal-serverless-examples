"""Harness smoke tests; /stream needs a raw SSE client because the harness only speaks request/response."""
import base64
import json
import os

import httpx


def test_invoke_contract(app_client):
    response = app_client.run(
        "/", {"prompt": "a red fox trotting through fresh snow at golden hour"}
    )
    assert response["video"]["url"].startswith("https://")
    # README documents this exact shape (explicit content_type + file_name)
    assert response["video"]["content_type"] == "video/mp4"
    assert response["video"]["file_name"] == "output.mp4"
    assert isinstance(response["seed"], int)
    # 57 already sits on the 8k+1 grid, so it echoes unchanged
    assert response["num_frames"] == 57
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
    # previews carry the same File-shaped `video` object as the result event
    preview_video = previews[0]["video"]
    assert preview_video["url"].startswith("data:video/mp4;base64,")
    assert preview_video["content_type"] == "video/mp4"
    payload = base64.b64decode(preview_video["url"].split(",", 1)[1])
    assert b"ftyp" in payload[:64], "preview payload is not an mp4"
    assert preview_video["file_size"] == len(payload)
    assert events[-1]["type"] == "result"
    assert events[-1]["video"]["url"].startswith("https://")
    # same contract as "/": from_path_async must carry explicit content_type/file_name
    assert events[-1]["video"]["content_type"] == "video/mp4"
    assert events[-1]["video"]["file_name"] == "output.mp4"
    assert events[-1]["num_frames"] == 57
