"""Tests for 04.4 train-serve-lora.

Two layers, one file (repo convention):

1. Unit tests for shared.py — run free on every plain `pytest`, no GPU,
   no live app. They exercise the exact validation code both apps ship.
2. Live tests — run by the harness against each deployed variant
   (HARNESS_VARIANT is "app" for the server, "app_trainer" for the
   trainer); the app_client fixture skips outside the harness.
"""
import importlib.util
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tsl_shared", Path(__file__).parent / "shared.py"
)
shared = importlib.util.module_from_spec(_spec)
sys.modules["tsl_shared"] = shared
_spec.loader.exec_module(shared)

VARIANT = os.environ.get("HARNESS_VARIANT", "app")

# The mirror convention: App.requirements in both apps must match this file.
REQUIREMENTS = (Path(__file__).parent / "requirements.txt").read_text().split()


# ---------------------------------------------------------------- unit: core

def test_slug_pattern_accepts_and_rejects():
    ok = ["paper-diorama", "u123", "a", "x" * 40, "with_underscore"]
    bad = ["", "-leading", "UPPER", "has space", "x" * 41, "dots.bad", "ünïcode"]
    for s in ok:
        assert re.match(shared.SLUG_PATTERN, s), s
    for s in bad:
        assert not re.match(shared.SLUG_PATTERN, s), s


def test_lora_paths_compose():
    assert shared.lora_weights_path("abc") == Path("/data/loras/abc/pytorch_lora_weights.safetensors")
    assert shared.lora_config_path("abc") == Path("/data/loras/abc/config.json")


def test_atomic_replace_moves_file(tmp_path):
    dest = tmp_path / "out" / "final.bin"
    tmp = tmp_path / "out" / ".tmp-x"
    tmp.parent.mkdir(parents=True)
    tmp.write_bytes(b"payload")
    shared.atomic_replace(tmp, dest)
    assert dest.read_bytes() == b"payload"
    assert not tmp.exists()


def test_tmp_sibling_same_directory(tmp_path):
    dest = tmp_path / "d" / "w.safetensors"
    tmp = shared.tmp_sibling(dest)
    assert tmp.parent == dest.parent  # same fs => os.replace is atomic
    assert tmp.exists()


# ---------------------------------------------------------- unit: lora gates

def _write_lora_file(path, keys_shapes, corrupt=False, zeros=False):
    torch = pytest.importorskip("torch", reason="LoRA-gate tests need torch (installed in example/runner envs, not static CI)")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    tensors = {}
    for key, shape in keys_shapes.items():
        t = torch.zeros(shape) if zeros else torch.randn(shape)
        tensors[key] = t
    safetensors_torch.save_file(tensors, str(path))
    if corrupt:
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])


GOOD_KEYS = {
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": (16, 3072),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": (3072, 16),
}


def test_validate_headers_accepts_good_file(tmp_path):
    p = tmp_path / "w.safetensors"
    _write_lora_file(p, GOOD_KEYS)
    shapes = shared.validate_safetensors_headers(p)
    assert set(shapes) == set(GOOD_KEYS)
    assert shapes["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"][0] == (16, 3072)


def test_validate_headers_rejects_corrupt_file(tmp_path):
    p = tmp_path / "w.safetensors"
    _write_lora_file(p, GOOD_KEYS, corrupt=True)
    with pytest.raises(shared.ArtifactError):
        shared.validate_safetensors_headers(p)


def test_validate_headers_rejects_all_zero_tensors(tmp_path):
    p = tmp_path / "w.safetensors"
    _write_lora_file(p, GOOD_KEYS, zeros=True)
    with pytest.raises(shared.ArtifactError, match="zero"):
        shared.validate_safetensors_headers(p)


def test_validate_headers_rejects_oversize(tmp_path):
    p = tmp_path / "w.safetensors"
    _write_lora_file(p, GOOD_KEYS)
    with pytest.raises(shared.ArtifactError, match="MB"):
        shared.validate_safetensors_headers(p, max_mb=0)


def test_assert_diffusers_keys_rejects_foreign_prefix(tmp_path):
    with pytest.raises(shared.ArtifactError):
        shared.assert_diffusers_lora_keys({"unet.down.0.lora_A.weight": ((16, 8), "F32")})
    # and accepts our own trainer's output shape
    shared.assert_diffusers_lora_keys({k: (v, "BF16") for k, v in GOOD_KEYS.items()})


# ------------------------------------------------------- unit: dataset prep

def _make_zip(tmp_path, entries):
    """entries: {archive_name: bytes}"""
    zp = tmp_path / "ds.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return zp


def _png_bytes(w=1024, h=1024, color=(200, 30, 30)):
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_prepare_dataset_happy_path(tmp_path):
    zp = _make_zip(tmp_path, {
        "a.png": _png_bytes(),
        "a.txt": b"a red square on a table",
        "b.png": _png_bytes(color=(30, 30, 200)),
    })
    ds = shared.prepare_dataset(zp, tmp_path / "w", default_caption="a photo")
    assert ds.num_images == 2
    assert ds.num_sidecar_captions == 1
    captions = dict((p.name, c) for p, c in ds.items)
    assert captions["000.png"] == "a red square on a table"
    assert captions["001.png"] == "a photo"
    assert all(p.exists() for p, _ in ds.items)


def test_prepare_dataset_prepends_trigger_phrase(tmp_path):
    zp = _make_zip(tmp_path, {"a.png": _png_bytes(), "a.txt": b"a red square"})
    ds = shared.prepare_dataset(zp, tmp_path / "w", trigger_phrase="pprdrma style,")
    assert ds.items[0][1] == "pprdrma style, a red square"


def test_prepare_dataset_missing_captions_names_files(tmp_path):
    zp = _make_zip(tmp_path, {"a.png": _png_bytes(), "b.png": _png_bytes()})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w")
    assert excinfo.value.code == "missing_captions"
    assert "a.png" in excinfo.value.message


def test_prepare_dataset_rejects_traversal(tmp_path):
    zp = _make_zip(tmp_path, {"../evil.png": _png_bytes()})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w", default_caption="x")
    assert excinfo.value.code == "invalid_archive"


def test_prepare_dataset_rejects_no_images(tmp_path):
    zp = _make_zip(tmp_path, {"notes.txt": b"hello"})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w", default_caption="x")
    assert excinfo.value.code == "no_images_found"


def test_prepare_dataset_rejects_non_zip(tmp_path):
    fake = tmp_path / "page.zip"
    fake.write_bytes(b"<!doctype html><html>expired presigned url</html>")
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(fake, tmp_path / "w", default_caption="x")
    assert excinfo.value.code == "invalid_archive"


def test_prepare_dataset_rejects_corrupt_image(tmp_path):
    zp = _make_zip(tmp_path, {"a.png": _png_bytes()[:100], "a.txt": b"cap"})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w")
    assert excinfo.value.code == "invalid_image"
    assert "a.png" in excinfo.value.message


def test_prepare_dataset_rejects_tiny_image_and_warns_small(tmp_path):
    zp = _make_zip(tmp_path, {"tiny.png": _png_bytes(64, 64), "tiny.txt": b"c"})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w")
    assert excinfo.value.code == "image_too_small"

    zp2 = _make_zip(tmp_path, {"small.png": _png_bytes(512, 512), "small.txt": b"c"})
    ds = shared.prepare_dataset(zp2, tmp_path / "w2")
    assert any("1024" in w for w in ds.warnings)


def test_prepare_dataset_skips_macosx_junk(tmp_path):
    zp = _make_zip(tmp_path, {
        "__MACOSX/._a.png": b"junk",
        "a.png": _png_bytes(),
        "a.txt": b"cap",
    })
    ds = shared.prepare_dataset(zp, tmp_path / "w")
    assert ds.num_images == 1


def test_prepare_dataset_blank_sidecar_caption_is_missing(tmp_path):
    zp = _make_zip(tmp_path, {"a.png": _png_bytes(), "a.txt": b"   \n"})
    with pytest.raises(shared.DatasetError) as excinfo:
        shared.prepare_dataset(zp, tmp_path / "w")
    assert excinfo.value.code == "missing_captions"
    ds = shared.prepare_dataset(zp, tmp_path / "w2", default_caption="fallback")
    assert ds.items[0][1] == "fallback"
    assert ds.num_sidecar_captions == 0


# ----------------------------------------------------- unit: trainer module

def _load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"tsl_{name}", Path(__file__).parent / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Allow the loaded modules to import shared without sys.path manipulation
sys.modules.setdefault("shared", shared)


def test_trainer_module_loads_with_expected_settings():
    trainer = _load_module("app_trainer")
    cls = trainer.KleinLoraTrainer
    assert cls.machine_type == "GPU-H100"
    assert cls.request_timeout == 7200
    assert cls.skip_retry_conditions == ["timeout"]
    assert cls.max_concurrency == 1
    assert cls.host_kwargs["keep_alive"] == 0
    assert cls.requirements == REQUIREMENTS
    assert cls.local_python_modules == ["shared"]


def test_trainer_preflight_input_has_smoke_example():
    trainer = _load_module("app_trainer")
    field = trainer.PreflightInput.model_fields["image_data_url"]
    assert field.examples == [shared.DATASET_ZIP_URL]


# ------------------------------------------------------ unit: server module

def test_server_module_loads_with_expected_settings():
    server = _load_module("app")
    cls = server.KleinLoraServer
    assert cls.machine_type == ["GPU-L40", "GPU-H100"]
    assert cls.requirements == REQUIREMENTS
    assert server.Input.model_fields["lora_name"].default == shared.DEMO_LORA
    scale = server.Input.model_fields["lora_scale"]
    assert scale.default == 1.0
    assert cls.host_kwargs["keep_alive"] == 300
    assert cls.startup_timeout == 1200
    assert cls.app_auth == "private"
    assert cls.local_python_modules == ["shared"]
    assert cls.max_multiplexing == 1


# ------------------------------------------------------------- live: helpers

import httpx  # noqa: E402  (used only by live tests; harness env has it)

server_only = pytest.mark.skipif(VARIANT != "app", reason="server variant only")
trainer_only = pytest.mark.skipif(VARIANT != "app_trainer", reason="trainer variant only")


def assert_typed_422(excinfo, field, type_):
    response = excinfo.value.response
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", field]
    assert detail["type"] == type_
    return detail


# -------------------------------------------------------------- live: server

@server_only
def test_generate_with_seeded_demo_adapter(app_client):
    response = app_client.run(
        "/",
        {"prompt": "pprdrma style, a lighthouse on a rocky island", "seed": 7},
    )
    assert response["seed"] == 7
    assert response["image"]["url"].startswith("https://")
    assert response["serving"]["lora_name"] == "paper-diorama"
    assert response["serving"]["adapter_source"] in {"cached", "disk", "seeded"}
    assert set(response["timings"]) == {"lora", "inference"}


@server_only
def test_lora_scale_zero_still_generates(app_client):
    response = app_client.run(
        "/", {"prompt": "a lighthouse on a rocky island", "lora_scale": 0.0, "seed": 7}
    )
    assert response["image"]["url"].startswith("https://")


@server_only
def test_unknown_lora_name_is_typed_422(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"prompt": "x", "lora_name": "does-not-exist"})
    assert_typed_422(excinfo, "lora_name", "lora_not_found")


@server_only
def test_bad_slug_is_schema_422(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"prompt": "x", "lora_name": "NOT A SLUG"})
    assert excinfo.value.response.status_code == 422  # pydantic pattern, free


# ------------------------------------------------------------- live: trainer

@trainer_only
def test_preflight_accepts_published_dataset(app_client):
    response = app_client.run("/", {"image_data_url": shared.DATASET_ZIP_URL})
    assert response["ready"] is True
    assert response["num_images"] == 16
    assert response["num_sidecar_captions"] == 16
    assert response["example_caption"]  # non-empty


@trainer_only
def test_preflight_trigger_phrase_lands_in_captions(app_client):
    response = app_client.run(
        "/",
        {"image_data_url": shared.DATASET_ZIP_URL, "trigger_phrase": "pprdrma style,"},
    )
    assert response["example_caption"].startswith("pprdrma style,")


@trainer_only
def test_preflight_rejects_html_page_as_archive(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"image_data_url": "https://example.com"})
    detail = assert_typed_422(excinfo, "image_data_url", "invalid_archive")
    assert "ZIP" in detail["msg"]


@trainer_only
def test_preflight_rejects_unroutable_host(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/", {"image_data_url": "http://169.254.169.254/latest/meta-data/"}
        )
    assert_typed_422(excinfo, "image_data_url", "file_download_error")
