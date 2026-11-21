"""Free unit tests plus one live harness smoke test for multi-gpu-video-custom-image."""

import importlib.util
import re
from pathlib import Path

import httpx

HERE = Path(__file__).parent

# --import-mode=importlib (duplicate test_app.py basenames): load app.py by path.
_spec = importlib.util.spec_from_file_location(
    "multi_gpu_video_custom_image_app", HERE / "app.py"
)
assert _spec is not None and _spec.loader is not None
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)

WAN_ATTENTION_HEADS = 40


def test_ulysses_degree_divides_heads():
    assert WAN_ATTENTION_HEADS % app_module.NUM_GPUS == 0


def test_app_requests_eight_gpus_from_the_dockerfile():
    app_cls = app_module.MultiGpuVideoCustomImage
    assert app_cls.num_gpus == 8
    assert app_cls.image is not None
    assert not app_cls.requirements


def test_requirements_txt_mirrors_dockerfile_pins():
    dockerfile = (HERE / "Dockerfile").read_text()
    docker_pins = set(re.findall(r"([A-Za-z0-9_.-]+==[0-9][^\s\\]*)", dockerfile))
    docker_pins.discard("pip==25.2")
    txt_pins = {
        line.strip()
        for line in (HERE / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert docker_pins == txt_pins


def test_base_image_is_digest_pinned():
    from_line = next(
        line
        for line in (HERE / "Dockerfile").read_text().splitlines()
        if line.startswith("FROM ")
    )
    assert re.search(r"@sha256:[0-9a-f]{64}$", from_line), from_line


def test_dockerfile_installs_python_311_and_fal_last():
    pip_lines = [
        line
        for line in (HERE / "Dockerfile").read_text().splitlines()
        if line.startswith("RUN python3.11 -m pip install")
    ]
    assert pip_lines, "no python3.11 pip installs"
    assert "fal==" in pip_lines[-1]


# Live: only under harness/check.py (app_client skips otherwise); each run spins 8 GPUs.
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
    assert response["num_gpus"] == 8
    assert response["timings"]["generate"] > 0
    url = response["video"]["url"]
    assert url.startswith("https://")
    # Range request: check the mp4 ftyp box without downloading the clip.
    head = httpx.get(url, timeout=60, headers={"Range": "bytes=0-63"}).content[:64]
    assert b"ftyp" in head[:16], f"not an mp4: {head!r}"
