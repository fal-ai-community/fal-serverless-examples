"""Shared, GPU-free helpers for the train-serve-lora example (04.4).

Imported by BOTH apps and shipped to each runner via `app_files` — the
"Import Code" mechanism this example teaches (docs: import-code). Everything
here runs on CPU with stdlib + Pillow + safetensors, so the unit tests in
test_app.py exercise the exact code the runners execute, for free.
"""
from __future__ import annotations

import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

# --- the published demo artifacts (see README "How the demo was made") ----
HF_REPO = "fal/flux-2-klein-4b-paper-diorama-lora"
DATASET_ZIP_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/dataset/paper-diorama-16.zip"
REFERENCE_ADAPTER_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/pytorch_lora_weights.safetensors"

# --- the model pair: gradients on base, 4-step inference on distilled -----
TRAIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
SERVE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# --- /data handoff conventions ---------------------------------------------
LORA_ROOT = Path("/data/loras")
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
LORA_CONFIG_NAME = "config.json"
DEMO_LORA = "paper-diorama"

# lora_name doubles as a /data directory name; the strict pattern keeps it
# path-safe by construction (same discipline as 04.3's user_id).
SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"

# --- input bounds -----------------------------------------------------------
MAX_ZIP_MB = 200            # streamed cap on the dataset download
MAX_ZIP_ENTRIES = 200
MAX_UNCOMPRESSED_MB = 500   # zip-bomb gate: declared size, before extraction
MIN_IMAGE_EDGE = 256        # hard floor; below this, training is hopeless
RECOMMENDED_EDGE = 1024     # BFL's style-LoRA guidance; smaller -> warning
MAX_IMAGES = 50             # style LoRAs want 10-40; caps the precompute bill
MAX_ADAPTER_MB = 300        # klein-4B rank-16 adapters are ~100 MB
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def lora_dir(name: str) -> Path:
    return LORA_ROOT / name


def lora_weights_path(name: str) -> Path:
    return lora_dir(name) / LORA_WEIGHT_NAME


def lora_config_path(name: str) -> Path:
    return lora_dir(name) / LORA_CONFIG_NAME


class DatasetError(ValueError):
    """A user-fixable dataset problem. Apps map this onto a typed 422."""

    def __init__(self, field: str, message: str, code: str):
        super().__init__(message)
        self.field = field
        self.message = message
        self.code = code


class ArtifactError(ValueError):
    """A LoRA file that must not be published or served."""


def tmp_sibling(dest: Path) -> Path:
    """A temp file in dest's own directory — same filesystem, so the final
    os.replace is atomic (/data documents renames as atomic; a reader sees
    the old complete file or the new one, never a partial write)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=dest.suffix)
    os.close(fd)
    return Path(name)


def atomic_replace(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)


def validate_safetensors_headers(
    path: Path, max_mb: int = MAX_ADAPTER_MB
) -> dict[str, tuple]:
    """Header-only sanity gate: parses, non-empty, size-capped, finite, not
    all-zeros. Returns {key: (shape, dtype)} without loading tensors to GPU.
    Trainer runs it before publishing; server runs it before load_lora_weights.
    """
    import safetensors
    import torch
    from safetensors import safe_open

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ArtifactError(
            f"adapter is {size_mb:.0f} MB; maximum is {max_mb} MB"
        )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                raise ArtifactError("empty safetensors file")
            shapes = {
                k: (tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
                for k in keys
            }
            # probe the few smallest tensors (CPU, cheap): finite + signal
            probe = sorted(keys, key=lambda k: math.prod(shapes[k][0]))[:4]
            any_nonzero = False
            for k in probe:
                t = f.get_tensor(k)
                if not torch.isfinite(t).all():
                    raise ArtifactError(f"non-finite values in {k}")
                any_nonzero = any_nonzero or bool(t.count_nonzero())
            if not any_nonzero:
                raise ArtifactError("probed tensors are all zeros - corrupt or untrained")
    except safetensors.SafetensorError as exc:
        raise ArtifactError(f"not a valid safetensors file: {exc}") from None
    return shapes


def assert_diffusers_lora_keys(shapes: dict[str, tuple]) -> None:
    """The trainer's publish gate: our own artifacts must be diffusers-native
    (`transformer.<module>.lora_A/B.weight`) so Flux2KleinPipeline (and 04.3's
    /register, and fal's hosted klein LoRA endpoints) load them directly."""
    bad = [
        k
        for k in shapes
        if not (k.startswith("transformer.") and (".lora_A." in k or ".lora_B." in k))
    ]
    if bad:
        raise ArtifactError(
            f"unexpected LoRA key layout (first offender: {bad[0]!r}); "
            "expected diffusers-native transformer.*.lora_A/B.weight keys"
        )
