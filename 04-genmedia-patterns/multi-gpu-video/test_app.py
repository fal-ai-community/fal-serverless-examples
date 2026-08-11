"""Free unit tests (static job) + one live harness test for 04.5.

The live test runs only under harness/check.py (the app_client fixture
skips otherwise), so plain `pytest` costs nothing. One live generation
total - each one spins 4 GPUs.
"""
import importlib.util
from pathlib import Path

import httpx
import pytest

# The repo pins pytest to --import-mode=importlib (same test_app.py basename
# in every example), so load the sibling app.py by path.
_spec = importlib.util.spec_from_file_location(
    "multi_gpu_video_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)

Input = app_module.Input
RESOLUTIONS = app_module.RESOLUTIONS


# ---- free unit tests (no GPU, no deploy) ----

def test_num_frames_rule_rejects_off_grid():
    with pytest.raises(ValueError):
        Input(prompt="x", num_frames=34)


def test_num_frames_rule_accepts_grid_values():
    for n in (9, 17, 33, 49, 81):
        assert Input(prompt="x", num_frames=n).num_frames == n


def test_resolution_map():
    assert RESOLUTIONS["480p"] == (480, 832)
    assert RESOLUTIONS["720p"] == (720, 1280)


def test_defaults_are_the_ci_smoke_config():
    i = Input(prompt="x")
    assert i.resolution == "480p"
    assert i.num_frames == 33
    assert i.num_inference_steps == 6


# ---- live test (harness only) ----

def test_generate_smoke(app_client):
    response = app_client.run(
        "/",
        {
            "prompt": "a red panda napping on a mossy branch, soft light",
            "resolution": "480p",
            "num_frames": 33,
            "num_inference_steps": 6,
            "seed": 7,
        },
    )
    assert response["seed"] == 7
    assert response["generation_seconds"] > 0
    url = response["video"]["url"]
    assert url.startswith("https://")
    head = httpx.get(url, timeout=60).content[:64]
    assert b"ftyp" in head[:16], f"not an mp4: {head!r}"
