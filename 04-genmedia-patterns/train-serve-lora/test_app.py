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
import zipfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tsl_shared", Path(__file__).parent / "shared.py"
)
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)

VARIANT = os.environ.get("HARNESS_VARIANT", "app")


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
