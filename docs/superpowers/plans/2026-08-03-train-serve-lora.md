# 04.4 Train-Serve-LoRA Implementation Plan

> **Historical planning artifact.** Execution deviated where reality disagreed
> (import mechanism, LoRA targets, lr schedule, demo recipe). The example
> folder and the spec are the source of truth for what shipped.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build catalog entry 04.4 (`04-genmedia-patterns/train-serve-lora/`, SERV-1312): a trainer app that fine-tunes a FLUX.2-klein-base-4B LoRA in-app and hands it to a single-adapter serving app via `/data`.

**Architecture:** Two `fal.App` files in one folder — `app_trainer.py` (H100, `keep_alive=0`, preflight on `/`, real training on `/train`, ~120-line PEFT flow-matching loop copied from the official diffusers v0.39.0 klein script) and `app.py` (distilled klein-4B server, one resident adapter swapped by `lora_name`, boot-seeded from a published reference adapter). `shared.py` (GPU-free: dataset validation, `/data` conventions, artifact gates) ships to both apps. Handoff = atomic temp+rename into `/data/loras/<lora_name>/`.

**Tech Stack:** fal 1.78.2 · diffusers 0.39.0 (`Flux2KleinPipeline`) · peft 0.19.1 · torch 2.12.1 · transformers 5.3.0 · safetensors 0.8.0 · Pillow 11.3.0.

**Spec:** `docs/superpowers/specs/2026-08-03-train-serve-lora-design.md` (approved 2026-08-03). Read it first.

## Global Constraints

- Python 3.11 (CI; cloudpickle requires client/runner agreement).
- Pins, identical in BOTH apps' `requirements` and mirrored exactly by `requirements.txt`: `fal==1.78.2`, `hf-transfer==0.1.9`, `diffusers==0.39.0`, `transformers==5.3.0`, `torch==2.12.1`, `accelerate==1.14.0`, `peft==0.19.1`, `safetensors==0.8.0`, `pillow==11.3.0`.
- Exactly one `fal.App` subclass per `app*.py` file (harness hard-fails otherwise). Every **required** Input field carries `examples=[...]` (harness builds the smoke payload from them). The harness smokes `/` on each app.
- Models: train on `black-forest-labs/FLUX.2-klein-base-4B`, serve on `black-forest-labs/FLUX.2-klein-4B` (both Apache-2.0, ungated; NEVER the 9B variants — gated + non-commercial).
- Training math is copied from `train_dreambooth_lora_flux2_klein.py` at the diffusers **v0.39.0 tag** — no scheme substitutions. One deliberate arg-value choice: text embeddings are computed with the **pipeline inference default** `text_encoder_out_layers=(9, 18, 27)` (by not passing the arg), NOT the script's CLI default `[10, 20, 30]`, so the LoRA trains against the embeddings the server will produce.
- Published artifacts live in HF repo **`fal/flux-2-klein-4b-paper-diorama-lora`** (needs fal-org perms; fallback: publish under Rehan's personal HF account and change only `HF_REPO` in `shared.py`).
- No wall-clock/latency/cost-measurement claims in README (2026-07-21 measurement descope). "A few minutes" phrasing only.
- Public-safe: no references to fal-internal repos, infra, or account names in any committed file.
- Steps marked 💸 spend real GPU/API money and need the project owner's go-ahead at execution time.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Scaffold, `shared.py` core (constants, paths, atomic writes, artifact gates), unit tests

**Files:**
- Create: `04-genmedia-patterns/train-serve-lora/shared.py`
- Create: `04-genmedia-patterns/train-serve-lora/test_app.py`
- Create: `04-genmedia-patterns/train-serve-lora/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces (used by every later task):
  - Constants: `HF_REPO: str`, `TRAIN_MODEL_ID: str`, `SERVE_MODEL_ID: str`, `LORA_ROOT: Path`, `LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"`, `LORA_CONFIG_NAME = "config.json"`, `DEMO_LORA = "paper-diorama"`, `DATASET_ZIP_URL: str`, `REFERENCE_ADAPTER_URL: str`, `SLUG_PATTERN: str`, `MAX_ZIP_MB = 200`, `MAX_ZIP_ENTRIES = 200`, `MAX_UNCOMPRESSED_MB = 500`, `MIN_IMAGE_EDGE = 256`, `RECOMMENDED_EDGE = 1024`, `MAX_IMAGES = 50`, `MAX_ADAPTER_MB = 300`, `IMAGE_SUFFIXES: set[str]`
  - `class DatasetError(ValueError)` with `.field: str`, `.message: str`, `.code: str`
  - `class ArtifactError(ValueError)`
  - `lora_dir(name: str) -> Path`, `lora_weights_path(name: str) -> Path`, `lora_config_path(name: str) -> Path`
  - `tmp_sibling(dest: Path) -> Path`, `atomic_replace(tmp: Path, dest: Path) -> None`
  - `validate_safetensors_headers(path: Path, max_mb: int = MAX_ADAPTER_MB) -> dict[str, tuple]` (key → (shape tuple, dtype str); raises `ArtifactError`)
  - `assert_diffusers_lora_keys(shapes: dict[str, tuple]) -> None` (raises `ArtifactError`)

- [ ] **Step 1: Create the folder and `requirements.txt`**

```bash
mkdir -p 04-genmedia-patterns/train-serve-lora
```

Write `04-genmedia-patterns/train-serve-lora/requirements.txt`:

```
fal==1.78.2
hf-transfer==0.1.9
diffusers==0.39.0
transformers==5.3.0
torch==2.12.1
accelerate==1.14.0
peft==0.19.1
safetensors==0.8.0
pillow==11.3.0
```

- [ ] **Step 2: Write the failing unit tests (shared core)**

Create `04-genmedia-patterns/train-serve-lora/test_app.py`:

```python
"""Tests for 04.4 train-serve-lora.

Two layers, one file (repo convention):

1. Unit tests for shared.py — run free on every plain `pytest`, no GPU,
   no live app. They exercise the exact validation code both apps ship.
2. Live tests — run by the harness against each deployed variant
   (HARNESS_VARIANT is "app" for the server, "app_trainer" for the
   trainer); the app_client fixture skips outside the harness.
"""
import importlib.util
import json
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
    import torch
    from safetensors.torch import save_file

    tensors = {}
    for key, shape in keys_shapes.items():
        t = torch.zeros(shape) if zeros else torch.randn(shape)
        tensors[key] = t
    save_file(tensors, str(path))
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: FAIL at import — `FileNotFoundError` (no `shared.py`).

- [ ] **Step 4: Write `shared.py` (core half)**

Create `04-genmedia-patterns/train-serve-lora/shared.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: all PASS (live tests aren't written yet; unit tests green). If `safetensors`/`torch`/`pillow` are missing locally: `pip install safetensors==0.8.0 pillow==11.3.0` (torch is already required by the harness env).

- [ ] **Step 6: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/
git commit -m "feat(04-genmedia-patterns): scaffold train-serve-lora shared core (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `shared.py` dataset pipeline (safe unzip, image checks, caption pairing)

**Files:**
- Modify: `04-genmedia-patterns/train-serve-lora/shared.py` (append)
- Modify: `04-genmedia-patterns/train-serve-lora/test_app.py` (append)

**Interfaces:**
- Produces:
  - `@dataclass PreparedDataset` with `items: list[tuple[Path, str]]` (processed RGB PNG path, final caption), `num_images: int`, `num_sidecar_captions: int`, `warnings: list[str]`
  - `prepare_dataset(zip_path: Path, work_dir: Path, default_caption: str | None = None, trigger_phrase: str | None = None) -> PreparedDataset` — raises `DatasetError` (field is always `"image_data_url"` except caption errors which use `"default_caption"`)

- [ ] **Step 1: Write the failing unit tests**

Append to `test_app.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: FAIL with `AttributeError: module 'tsl_shared' has no attribute 'prepare_dataset'`.

- [ ] **Step 3: Implement the dataset pipeline**

Append to `shared.py`:

```python
@dataclass
class PreparedDataset:
    """What prepare_dataset returns: training-ready items + a preflight report."""

    items: list[tuple[Path, str]]   # (processed RGB png path, final caption)
    num_images: int
    num_sidecar_captions: int
    warnings: list[str]


def _safe_infolist(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Zip safety: entry cap, zip-slip (traversal) rejection, declared-size
    bomb gate, macOS junk filtering. Runs BEFORE any extraction."""
    infos = [i for i in zf.infolist() if not i.filename.endswith("/")]
    if len(infos) > MAX_ZIP_ENTRIES:
        raise DatasetError(
            "image_data_url",
            f"archive has {len(infos)} entries; maximum is {MAX_ZIP_ENTRIES}",
            "archive_too_many_files",
        )
    keep, total = [], 0
    for info in infos:
        parts = Path(info.filename).parts
        if Path(info.filename).is_absolute() or ".." in parts:
            raise DatasetError(
                "image_data_url",
                f"archive entry {info.filename!r} escapes the extraction directory",
                "invalid_archive",
            )
        if any(p.lower() == "__macosx" for p in parts) or Path(info.filename).name.startswith("._"):
            continue
        total += info.file_size
        if total > MAX_UNCOMPRESSED_MB * 1024 * 1024:
            raise DatasetError(
                "image_data_url",
                f"archive declares more than {MAX_UNCOMPRESSED_MB} MB uncompressed",
                "file_too_large",
            )
        keep.append(info)
    return keep


def prepare_dataset(
    zip_path: Path,
    work_dir: Path,
    default_caption: str | None = None,
    trigger_phrase: str | None = None,
) -> PreparedDataset:
    """The trainer's entire input path, shared verbatim by the preflight
    endpoint: unzip safely, decode-check every image, normalize EXIF
    orientation to RGB PNG, pair captions. Deterministic item order.
    """
    from PIL import Image as PILImage
    from PIL import ImageOps

    if not zipfile.is_zipfile(zip_path):
        raise DatasetError(
            "image_data_url",
            "not a ZIP archive (an expired or HTML error page URL is the usual cause)",
            "invalid_archive",
        )

    raw_dir = work_dir / "raw"
    out_dir = work_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = _safe_infolist(zf)
        for info in members:
            zf.extract(info, raw_dir)

    extracted = [raw_dir / i.filename for i in members]
    images = sorted(p for p in extracted if p.suffix.lower() in IMAGE_SUFFIXES)
    captions = {
        p.with_suffix("").name: p.read_text(encoding="utf-8", errors="replace").strip()
        for p in extracted
        if p.suffix.lower() == ".txt"
    }

    if not images:
        raise DatasetError(
            "image_data_url",
            f"no images found in the archive (accepted: {', '.join(sorted(IMAGE_SUFFIXES))})",
            "no_images_found",
        )
    if len(images) > MAX_IMAGES:
        raise DatasetError(
            "image_data_url",
            f"{len(images)} images; maximum is {MAX_IMAGES} (style LoRAs want 10-40)",
            "dataset_too_large",
        )

    missing = [p.name for p in images if p.with_suffix("").name not in captions]
    if missing and default_caption is None:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        raise DatasetError(
            "default_caption",
            f"{len(missing)} image(s) have no .txt caption ({shown}); "
            "add sidecar captions or set default_caption",
            "missing_captions",
        )

    items: list[tuple[Path, str]] = []
    warnings: list[str] = []
    small: list[str] = []
    for idx, path in enumerate(images):
        try:
            with PILImage.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.load()
        except Exception as exc:
            raise DatasetError(
                "image_data_url",
                f"could not decode {path.name!r}: {exc}",
                "invalid_image",
            ) from None
        if min(img.size) < MIN_IMAGE_EDGE:
            raise DatasetError(
                "image_data_url",
                f"{path.name!r} is {img.size[0]}x{img.size[1]}; "
                f"minimum edge is {MIN_IMAGE_EDGE}px",
                "image_too_small",
            )
        if min(img.size) < RECOMMENDED_EDGE:
            small.append(path.name)
        processed = out_dir / f"{idx:03d}.png"
        img.save(processed, format="PNG")
        caption = captions.get(path.with_suffix("").name) or default_caption
        if trigger_phrase:
            caption = f"{trigger_phrase} {caption}"
        items.append((processed, caption))

    if small:
        warnings.append(
            f"{len(small)} image(s) below {RECOMMENDED_EDGE}px shortest edge "
            f"({', '.join(small[:5])}); BFL recommends >={RECOMMENDED_EDGE}px for style LoRAs"
        )
    return PreparedDataset(
        items=items,
        num_images=len(items),
        num_sidecar_captions=len(images) - len(missing),
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/
git commit -m "feat(04-genmedia-patterns): train-serve-lora dataset pipeline with zip-safety gates (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `make_dataset.py` + generate & publish the demo dataset 💸

**Files:**
- Create: `04-genmedia-patterns/train-serve-lora/make_dataset.py`

**Interfaces:**
- Consumes: `shared.DEMO_LORA` naming only (standalone script; runs on the maintainer's/reader's machine, not on a runner).
- Produces: `paper-diorama-16.zip` — 16 images + 16 caption `.txt`s; published at `shared.DATASET_ZIP_URL`. The trainer's `Field(examples=[DATASET_ZIP_URL])` and the harness smoke depend on this URL resolving.

- [ ] **Step 1: Verify the hosted klein endpoint id**

```bash
curl -s "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/flux-2-klein" | head -c 400
```

Expected: OpenAPI JSON (not an error object). If it errors, probe `fal-ai/flux-2-klein/4b` and use whichever resolves as `DEFAULT_ENDPOINT` below.

- [ ] **Step 2: Write `make_dataset.py`**

```python
"""Generate the paper-diorama style training set with klein itself.

16 subjects x 1 fixed style suffix x fixed seeds -> 16 images at 1024px,
each with a sidecar .txt caption describing CONTENT ONLY (the style lives
in the trigger phrase at training time, so it transfers to new subjects).

Runs anywhere with a FAL_KEY (hosted endpoint) - or point --endpoint at
your own 00.2 deployment (e.g. <username>/first-image-api).

    pip install fal-client
    python make_dataset.py --out paper-diorama-16.zip
"""
import argparse
import io
import zipfile
from pathlib import Path

import fal_client
import httpx

DEFAULT_ENDPOINT = "fal-ai/flux-2-klein"
STYLE_SUFFIX = (
    ", handcrafted paper-cutout diorama, layered colored cardstock, "
    "visible paper texture, soft studio lighting, shallow depth"
)
SUBJECTS = [
    "a lighthouse on a rocky island at dusk",
    "a red fox in a snowy forest",
    "a sailboat on a calm turquoise sea",
    "a hot air balloon over rolling hills",
    "a cottage with a smoking chimney in a meadow",
    "a whale breaching next to a fishing boat",
    "a desert canyon with a winding river",
    "a night market street with lanterns",
    "a mountain climber on a snowy ridge",
    "an owl perched on a birch branch",
    "a tram crossing a city bridge at sunset",
    "a diver exploring a coral reef",
    "a windmill among tulip fields",
    "a campsite with a tent under the aurora",
    "a lion resting under an acacia tree",
    "a rocket launching from a jungle clearing",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", default="paper-diorama-16.zip")
    args = parser.parse_args()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, subject in enumerate(SUBJECTS):
            result = fal_client.subscribe(
                args.endpoint,
                arguments={
                    "prompt": subject + STYLE_SUFFIX,
                    "image_size": {"width": 1024, "height": 1024},
                    "num_inference_steps": 4,
                    "seed": 1000 + i,
                },
            )
            url = result["images"][0]["url"] if "images" in result else result["image"]["url"]
            png = httpx.get(url, timeout=120).content
            zf.writestr(f"{i:03d}.png", png)
            zf.writestr(f"{i:03d}.txt", subject)
            print(f"[{i + 1}/{len(SUBJECTS)}] {subject}")
    Path(args.out).write_bytes(buf.getvalue())
    print(f"wrote {args.out} ({len(buf.getvalue()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 💸 Run it and eyeball the style**

```bash
cd 04-genmedia-patterns/train-serve-lora && python make_dataset.py --out /tmp/paper-diorama-16.zip
```

Expected: 16 progress lines, a ~20-40 MB zip. Open a few images — they must share an unmistakable paper-diorama look. If the style is weak or inconsistent, tune `STYLE_SUFFIX` wording (not the subjects) and re-run. The output schema key (`images` vs `image`) depends on the endpoint — the script handles both.

- [ ] **Step 4: Publish the ZIP (human gate: HF org perms)**

```bash
pip install -q huggingface_hub && hf auth whoami
hf upload fal/flux-2-klein-4b-paper-diorama-lora /tmp/paper-diorama-16.zip dataset/paper-diorama-16.zip --repo-type=model --commit-message "paper-diorama training set (16 images, klein-generated)"
curl -sI "https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/dataset/paper-diorama-16.zip" | head -1
```

Expected: final curl prints `HTTP/2 302` (resolve redirect = published). If org perms aren't available yet, upload to `<personal>/flux-2-klein-4b-paper-diorama-lora` and change only `HF_REPO` in `shared.py` (leave a `TODO(SERV-1312): move to fal org` in the PR description, NOT in code).

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/make_dataset.py
git commit -m "feat(04-genmedia-patterns): train-serve-lora demo dataset generator (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `app_trainer.py` — app shell + preflight endpoint

**Files:**
- Create: `04-genmedia-patterns/train-serve-lora/app_trainer.py`
- Modify: `04-genmedia-patterns/train-serve-lora/test_app.py` (append)

**Interfaces:**
- Consumes: `shared.prepare_dataset`, `shared.DatasetError`, `shared.DATASET_ZIP_URL`, `shared.MAX_ZIP_MB`, `shared.SLUG_PATTERN`.
- Produces: `KleinLoraTrainer(fal.App)` with `@fal.endpoint("/")` `preflight(request: PreflightInput) -> PreflightReport`; models `PreflightInput{image_data_url, default_caption, trigger_phrase}`, `PreflightReport{ready, num_images, num_sidecar_captions, example_caption, warnings}`. Task 5 appends the `/train` endpoint to this same class.

- [ ] **Step 1: Write the failing unit tests**

Append to `test_app.py`:

```python
# ----------------------------------------------------- unit: trainer module

def _load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"tsl_{name}", Path(__file__).parent / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trainer_module_loads_with_expected_settings():
    trainer = _load_module("app_trainer")
    cls = trainer.KleinLoraTrainer
    assert cls.machine_type == "GPU-H100"
    assert cls.request_timeout == 7200
    assert cls.skip_retry_conditions == ["timeout"]
    assert cls.max_concurrency == 1
    assert cls.host_kwargs["keep_alive"] == 0
    assert cls.requirements == shared_requirements()


def shared_requirements():
    return [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "safetensors==0.8.0",
        "pillow==11.3.0",
    ]


def test_trainer_preflight_input_has_smoke_example():
    trainer = _load_module("app_trainer")
    field = trainer.PreflightInput.model_fields["image_data_url"]
    assert field.examples == [shared.DATASET_ZIP_URL]
```

Note: if `host_kwargs["keep_alive"]` proves not to be where fal 1.78.2 stores it, inspect `fal.App` subclass attrs interactively (`python -c "import fal; ..."`) and assert wherever the value actually lands — the point of the assertion is that `keep_alive=0` is recorded, not the attribute path.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q -k trainer
```

Expected: FAIL — `FileNotFoundError` for `app_trainer.py`.

- [ ] **Step 3: Write `app_trainer.py` (shell + preflight; `/train` arrives in Task 5)**

```python
"""Train a FLUX.2-klein LoRA on fal serverless - the trainer half of 04.4.

The interesting part is the settings block: this is the long-job pattern
the docs imply but never assemble in one place.

  keep_alive=0                       a training job pays for exactly what it
                                     uses; no idle GPU, no adapter-state leak
  request_timeout=7200               training outlives the 3600s default;
                                     7200 is the account-level maximum
  skip_retry_conditions=["timeout"]  a timed-out training must NOT be silently
                                     re-run (and re-billed) by the queue
  max_concurrency=1                  one training per GPU - concurrent jobs on
                                     one runner would fight for VRAM

`POST /` is a dataset preflight: the exact validation head of `/train`,
no GPU time. (Hosted fal trainers validate inline instead - this split
exists so CI can exercise the input path without paying for a train.)
Training math in `/train` is the official diffusers klein DreamBooth
script's core, inlined - see the pinned source link at TRAINING_SOURCE.
"""
import tempfile
from pathlib import Path

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

import shared

TRAINING_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/v0.39.0/"
    "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
)

# The product-matching hidden defaults (fixed on purpose - the public surface
# stays small; change them here, not per-request):
RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
RESOLUTION = 1024        # BFL style-LoRA guidance; drop to 512 for cheap iteration
LR_WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
TRAIN_SEED = 42
LOG_EVERY = 25


class PreflightInput(BaseModel):
    image_data_url: str = Field(
        description=(
            "URL of a ZIP of 10-40 style images (png/jpg/webp), each optionally "
            "paired with a same-name .txt caption describing its CONTENT "
            "(the style itself belongs in trigger_phrase)."
        ),
        examples=[shared.DATASET_ZIP_URL],
    )
    default_caption: str | None = Field(
        default=None,
        description="Caption used for any image without a sidecar .txt.",
    )
    trigger_phrase: str | None = Field(
        default=None,
        description=(
            "Prepended to every caption at training time (plain text prepend, "
            "no token surgery). Use a non-word like 'pprdrma style,' and repeat "
            "it in serving prompts to invoke the style."
        ),
    )


class PreflightReport(BaseModel):
    ready: bool = Field(description="True - errors surface as typed 422s instead.")
    num_images: int = Field(description="Images that passed decoding and size gates.")
    num_sidecar_captions: int = Field(description="Images with their own .txt caption.")
    example_caption: str = Field(description="The final caption of the first image.")
    warnings: list[str] = Field(description="Non-fatal dataset advice.")


class TrainInput(PreflightInput):
    lora_name: str = Field(
        pattern=shared.SLUG_PATTERN,
        description=(
            "Adapter slug: becomes /data/loras/<lora_name>/ on your account, "
            "which the serving app (app.py) loads by name."
        ),
        examples=["my-style"],
    )
    steps: int = Field(
        default=1000,
        ge=100,
        le=10_000,
        multiple_of=100,
        description="Training steps. Style LoRAs peak around 750-1500.",
    )
    learning_rate: float = Field(
        default=5e-5,
        ge=1e-6,
        le=1e-3,
        description="AdamW learning rate (cosine schedule, 100 warmup steps).",
    )


class TrainOutput(BaseModel):
    lora_file: File = Field(
        description=(
            "The trained adapter (diffusers-native keys) on the CDN - portable: "
            "drop-in for 04.3's /register or fal's hosted klein LoRA endpoints."
        )
    )
    data_path: str = Field(description="Where the adapter landed on /data.")
    steps: int = Field(description="Steps actually trained.")
    final_loss: float = Field(description="Mean loss over the last 25 steps.")
    config: dict = Field(description="Echo of the training inputs (config.json sidecar).")


def _fetch_dataset(url: str, work_dir: Path) -> Path:
    """Bounded, SSRF-guarded download (both built into download_file)."""
    try:
        return download_file(url, work_dir, filesize_limit=shared.MAX_ZIP_MB)
    except DownloadError as exc:
        detail = str(exc)
        code = "file_too_large" if ("exceed" in detail or "limit" in detail) else "file_download_error"
        raise FieldException(
            field="image_data_url",
            message=f"could not download (limit {shared.MAX_ZIP_MB} MB): {detail}",
            type=code,
        ) from None


def _prepare_or_422(zip_path: Path, work_dir: Path, request: PreflightInput):
    try:
        return shared.prepare_dataset(
            zip_path,
            work_dir,
            default_caption=request.default_caption,
            trigger_phrase=request.trigger_phrase,
        )
    except shared.DatasetError as exc:
        raise FieldException(field=exc.field, message=exc.message, type=exc.code) from None


class KleinLoraTrainer(fal.App, keep_alive=0):
    machine_type = "GPU-H100"  # training wants speed; L40 works (see README)
    app_auth = "private"
    startup_timeout = 1200     # first-ever boot downloads ~16 GB of base weights
    request_timeout = 7200     # account-level max; a 10k-step run needs it
    skip_retry_conditions = ["timeout"]
    max_concurrency = 1
    app_files = ["shared.py"]  # Import Code: ship the shared module with the app
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "safetensors==0.8.0",
        "pillow==11.3.0",
    ]

    def setup(self) -> None:
        """Load the BASE model (gradients need the undistilled weights).
        Hard-fail: a trainer that can't load its model must not look healthy."""
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            shared.TRAIN_MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

    @fal.endpoint("/")
    def preflight(self, request: PreflightInput) -> PreflightReport:
        """Validate a dataset without spending a training run on it.
        This is byte-for-byte the head of /train."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            zip_path = _fetch_dataset(request.image_data_url, work / "dl")
            ds = _prepare_or_422(zip_path, work, request)
        return PreflightReport(
            ready=True,
            num_images=ds.num_images,
            num_sidecar_captions=ds.num_sidecar_captions,
            example_caption=ds.items[0][1],
            warnings=ds.warnings,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: all PASS. If the `host_kwargs["keep_alive"]` assertion fails, find where fal 1.78.2 records it (`python -c "import app_trainer ..."` from the folder) and fix the assertion, not the app.

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/
git commit -m "feat(04-genmedia-patterns): train-serve-lora trainer shell + dataset preflight (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `app_trainer.py` — the `/train` endpoint (training core)

**Files:**
- Modify: `04-genmedia-patterns/train-serve-lora/app_trainer.py` (append method to `KleinLoraTrainer`, after `preflight`)

**Interfaces:**
- Consumes: Task 4's models/helpers; `shared.validate_safetensors_headers`, `shared.assert_diffusers_lora_keys`, `shared.atomic_replace`, `shared.tmp_sibling`, `shared.lora_weights_path`, `shared.lora_config_path`, `shared.LORA_ROOT`.
- Produces: `POST /train` → `TrainOutput`. The artifact contract Tasks 6/9 rely on: `/data/loras/<lora_name>/pytorch_lora_weights.safetensors` (diffusers-native keys) + `config.json` sidecar `{model, steps, learning_rate, trigger_phrase, default_caption, rank, lora_alpha, resolution}`.

- [ ] **Step 1: Append the training endpoint**

Append inside `KleinLoraTrainer` (after `preflight`). Every math line is the official script's, inlined; comments cite the deviation points only:

```python
    @fal.endpoint("/train")
    def train(self, request: TrainInput) -> TrainOutput:
        """Fine-tune a LoRA and publish it to /data (atomic) + the CDN.

        The loop is the core of the official diffusers klein DreamBooth
        script (TRAINING_SOURCE), inlined so you can read it: precompute
        embeddings/latents once, offload encoders, train transformer-only.
        """
        import copy
        import json
        import random
        import time

        import numpy as np
        import torch
        from diffusers import Flux2KleinPipeline
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import (
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
            free_memory,
        )
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from PIL import Image as PILImage

        pipe = self.pipe
        device = "cuda"
        t_start = time.monotonic()

        # ------------------------------------------------ 1. dataset (same head as /)
        staging = Path(tempfile.mkdtemp(prefix="train-"))
        zip_path = _fetch_dataset(request.image_data_url, staging / "dl")
        ds = _prepare_or_422(zip_path, staging, request)
        print(f"[train] dataset ok: {ds.num_images} images, "
              f"{ds.num_sidecar_captions} sidecar captions, {len(ds.warnings)} warnings")

        # ------------------------------------------------ 2. precompute (once)
        # Text: encoded with the PIPELINE's inference default layers (9, 18, 27)
        # - NOT the script's CLI default [10, 20, 30] - because the serving app
        # re-encodes prompts with pipeline defaults and the LoRA must train
        # against the embeddings it will be served with.
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        embeds, text_ids_list = [], []
        with torch.no_grad():
            for _, caption in ds.items:
                pe, ti = pipe.encode_prompt(prompt=caption, device=device)
                embeds.append(pe)
                text_ids_list.append(ti)

        # Images: min-edge resize + center crop to RESOLUTION (deterministic
        # dataset prep; the tensor math below is the script's), then VAE
        # encode -> patchify -> BatchNorm-normalize (klein has no
        # scaling/shift factor; it normalizes with the VAE's BN stats).
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(device)
        latents = []
        with torch.no_grad():
            for path, _ in ds.items:
                img = PILImage.open(path)  # already RGB, EXIF-normalized
                scale = RESOLUTION / min(img.size)
                img = img.resize(
                    (round(img.width * scale), round(img.height * scale)),
                    PILImage.BILINEAR,
                )
                left = (img.width - RESOLUTION) // 2
                top = (img.height - RESOLUTION) // 2
                img = img.crop((left, top, left + RESOLUTION, top + RESOLUTION))
                px = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
                px = (px.permute(2, 0, 1) * 2 - 1).unsqueeze(0)
                px = px.to(device, dtype=pipe.vae.dtype)
                lat = pipe.vae.encode(px).latent_dist.mode()  # script uses mode()
                lat = Flux2KleinPipeline._patchify_latents(lat)
                lat = (lat - bn_mean) / bn_std
                latents.append(lat)
        # The single biggest VRAM/speed lever: encoders leave the GPU for the
        # whole loop; only the transformer stays.
        pipe.text_encoder.to("cpu")
        pipe.vae.to("cpu")
        free_memory()
        print(f"[train] precompute done in {time.monotonic() - t_start:.0f}s "
              f"({len(latents)} latents cached)")

        # ------------------------------------------------ 3. LoRA on the transformer
        transformer = pipe.transformer
        transformer.requires_grad_(False)
        # Verbatim from the script: attention modules in double blocks, fused
        # qkv+mlp projection + attention out in single blocks. range(24) is
        # the 9B count; peft ignores names that don't exist in the 4B (20).
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"] + [
            "to_qkv_mlp_proj",
            *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)],
        ]
        transformer.add_adapter(
            LoraConfig(
                r=RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        )
        transformer.enable_gradient_checkpointing()
        transformer.train()
        lora_params = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=request.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )
        lr_sched = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=LR_WARMUP_STEPS,
            num_training_steps=request.steps,
        )
        sched = copy.deepcopy(pipe.scheduler)  # never mutate the pipeline's copy

        def get_sigmas(timesteps, n_dim, dtype):
            # Verbatim helper from the script.
            sigmas = sched.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = sched.timesteps.to(device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        # ------------------------------------------------ 4. the loop
        gen = torch.Generator(device=device).manual_seed(TRAIN_SEED)
        rng = random.Random(TRAIN_SEED)
        order: list[int] = []
        window: list[float] = []
        try:
            for step in range(1, request.steps + 1):
                if not order:  # epoch boundary: reshuffle, no replacement
                    order = list(range(len(latents)))
                    rng.shuffle(order)
                i = order.pop()
                model_input = latents[i]
                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device)
                noise = torch.randn(
                    model_input.shape, device=device, dtype=model_input.dtype, generator=gen
                )

                # Timestep sampling - script defaults (weighting_scheme="none"
                # => uniform density, unit loss weighting).
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=1,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * sched.config.num_train_timesteps).long()
                timesteps = sched.timesteps[indices].to(device)
                sigmas = get_sigmas(timesteps, model_input.ndim, model_input.dtype)

                # zt = (1 - sigma) * x + sigma * noise  (flow matching)
                noisy = (1.0 - sigmas) * model_input + sigmas * noise
                packed = Flux2KleinPipeline._pack_latents(noisy)

                model_pred = transformer(
                    hidden_states=packed,
                    timestep=timesteps / 1000,
                    guidance=None,  # klein: guidance_embeds=False
                    encoder_hidden_states=embeds[i],
                    txt_ids=text_ids_list[i],
                    img_ids=model_input_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed.size(1)]
                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
                    model_pred, model_input_ids
                )

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas
                )
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                ).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

                window.append(loss.item())
                if len(window) > LOG_EVERY:
                    window.pop(0)
                if step == 1 or step % LOG_EVERY == 0:
                    print(
                        f"[train] step {step}/{request.steps} "
                        f"loss {sum(window) / len(window):.4f} "
                        f"lr {lr_sched.get_last_lr()[0]:.2e} "
                        f"elapsed {time.monotonic() - t_start:.0f}s"
                    )
            # (CUDA OOM propagates as a 5xx on purpose: retryable platform
            # fault, never a permanent 422 blaming the user's dataset.)

            # -------------------------------------------- 5. gate -> publish
            transformer.eval()
            lora_layers = get_peft_model_state_dict(transformer)
            shared.LORA_ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=shared.LORA_ROOT) as td:
                # temp dir on /data itself => same filesystem => atomic rename
                Flux2KleinPipeline.save_lora_weights(
                    save_directory=td, transformer_lora_layers=lora_layers
                )
                tmp_weights = Path(td) / shared.LORA_WEIGHT_NAME
                shapes = shared.validate_safetensors_headers(tmp_weights)
                shared.assert_diffusers_lora_keys(shapes)

                config = {
                    "model": shared.TRAIN_MODEL_ID,
                    "steps": request.steps,
                    "learning_rate": request.learning_rate,
                    "trigger_phrase": request.trigger_phrase,
                    "default_caption": request.default_caption,
                    "rank": RANK,
                    "lora_alpha": LORA_ALPHA,
                    "resolution": RESOLUTION,
                }
                # CDN copy first (portable URL), then config, then weights
                # LAST - the weights file appearing is the readiness signal.
                lora_file = File.from_path(str(tmp_weights), repository="fal_v3")
                cfg_tmp = shared.tmp_sibling(shared.lora_config_path(request.lora_name))
                cfg_tmp.write_text(json.dumps(config, indent=2))
                shared.atomic_replace(cfg_tmp, shared.lora_config_path(request.lora_name))
                shared.atomic_replace(
                    tmp_weights, shared.lora_weights_path(request.lora_name)
                )
        finally:
            # Fresh adapter per request, belt and braces with keep_alive=0.
            try:
                transformer.delete_adapters("default")
            except Exception:
                pass

        print(f"[train] published /data/loras/{request.lora_name} "
              f"in {time.monotonic() - t_start:.0f}s total")
        return TrainOutput(
            lora_file=lora_file,
            data_path=str(shared.lora_weights_path(request.lora_name)),
            steps=request.steps,
            final_loss=round(sum(window) / len(window), 4),
            config=config,
        )
```

- [ ] **Step 2: Verify the module still loads and unit tests pass**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: all PASS (heavy imports live inside methods, so the module loads without torch on the GPU path).

- [ ] **Step 3: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/app_trainer.py
git commit -m "feat(04-genmedia-patterns): train-serve-lora training core - official klein script inlined (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `app.py` — the serving app

**Files:**
- Create: `04-genmedia-patterns/train-serve-lora/app.py`
- Modify: `04-genmedia-patterns/train-serve-lora/test_app.py` (append)

**Interfaces:**
- Consumes: `shared.SERVE_MODEL_ID`, `shared.DEMO_LORA`, `shared.REFERENCE_ADAPTER_URL`, `shared.MAX_ADAPTER_MB`, `shared.lora_weights_path`, `shared.tmp_sibling`, `shared.atomic_replace`, `shared.validate_safetensors_headers`, `shared.SLUG_PATTERN`.
- Produces: `KleinLoraServer(fal.App)` with `@fal.endpoint("/")` `generate(request: GenerateInput) -> GenerateOutput`; `GenerateOutput.serving = ServingInfo{lora_name, adapter_source: "cached"|"disk"|"seeded"}`; `GenerateOutput.timings = {"lora": s, "inference": s}`.

- [ ] **Step 1: Write the failing unit tests**

Append to `test_app.py`:

```python
# ------------------------------------------------------ unit: server module

def test_server_module_loads_with_expected_settings():
    server = _load_module("app")
    cls = server.KleinLoraServer
    assert cls.machine_type == ["GPU-L40", "GPU-H100"]
    assert cls.requirements == shared_requirements()
    assert server.GenerateInput.model_fields["lora_name"].default == shared.DEMO_LORA
    scale = server.GenerateInput.model_fields["lora_scale"]
    assert scale.default == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q -k server
```

Expected: FAIL — `FileNotFoundError` for `app.py`.

- [ ] **Step 3: Write `app.py`**

```python
"""Serve a freshly-trained klein LoRA from /data - the serving half of 04.4.

The trainer (app_trainer.py) writes /data/loras/<lora_name>/; this app loads
adapters from there by name onto the DISTILLED klein-4B (4-step inference).
Train where gradients behave, serve where inference is fast.

One resident adapter, swapped on demand - deliberately the 20-line version
of 04.3's LRU fleet (see README "Next steps").
"""
import random
import time
from pathlib import Path

import fal
from fal.exceptions import FieldException
from fal.toolkit import Image, ImageSizeInput, download_file, get_image_size
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

import shared


class GenerateInput(BaseModel):
    prompt: str = Field(
        description=(
            "What to generate. Repeat your training trigger phrase to invoke "
            "the style (the demo adapter's is 'pprdrma style,')."
        ),
        examples=["pprdrma style, a tram crossing a city bridge at sunset"],
    )
    lora_name: str = Field(
        default=shared.DEMO_LORA,
        pattern=shared.SLUG_PATTERN,
        description="Adapter to serve, by trainer slug (/data/loras/<lora_name>/).",
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="Adapter strength. 0 disables the style; ~0.7-1.2 is typical.",
    )
    image_size: ImageSizeInput = Field(
        default="square_hd",
        description="A preset name or a custom {width, height} (multiples of 16).",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, le=8,
        description="Denoising steps. klein is distilled for 4; more buys little.",
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted; the response echoes the seed used.",
    )
    sync_mode: bool = Field(
        default=False,
        description="Return a data URI instead of a hosted CDN URL.",
    )


class ServingInfo(BaseModel):
    lora_name: str = Field(description="The adapter that produced this image.")
    adapter_source: str = Field(
        description=(
            "'cached' = already on the GPU; 'disk' = loaded from /data for this "
            "request; 'seeded' = loaded from the /data copy that setup() "
            "bootstrapped from the published reference adapter."
        )
    )


class GenerateOutput(BaseModel):
    image: Image = Field(description="The generated image.")
    seed: int = Field(description="The seed that produced this image.")
    timings: dict = Field(description="Seconds spent in adapter handling vs inference.")
    serving: ServingInfo = Field(description="How this request was served.")


class KleinLoraServer(fal.App, keep_alive=300):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    startup_timeout = 1200  # first boot: ~16 GB of weights + the demo adapter
    app_files = ["shared.py"]  # Import Code: same module the trainer ships
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "safetensors==0.8.0",
        "pillow==11.3.0",
    ]

    def setup(self) -> None:
        """Load -> seed -> adopt -> warm up, all through the adapter path,
        so a healthy worker is proven able to serve LoRA traffic."""
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            shared.SERVE_MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        # Idempotent /data bootstrap (04.3's featured-user pattern): if the
        # demo adapter isn't on this account's /data yet, fetch the published
        # reference copy - CI stays green on a fresh account, and the example
        # works before you've run your first training.
        self._resident: str | None = None
        self._adapter_id: str | None = None
        self._counter = 0
        self._seeded: set[str] = set()
        demo_weights = shared.lora_weights_path(shared.DEMO_LORA)
        if not demo_weights.exists():
            tmp = shared.tmp_sibling(demo_weights)
            try:
                fetched = download_file(
                    shared.REFERENCE_ADAPTER_URL,
                    tmp.parent,
                    filesize_limit=shared.MAX_ADAPTER_MB,
                )
            except DownloadError as exc:
                raise RuntimeError(
                    f"could not seed the demo adapter from {shared.REFERENCE_ADAPTER_URL}: {exc}"
                ) from exc
            shared.validate_safetensors_headers(Path(fetched))
            shared.atomic_replace(Path(fetched), demo_weights)
            tmp.unlink(missing_ok=True)
            self._seeded.add(shared.DEMO_LORA)

        self._load_adapter(shared.DEMO_LORA)
        # Warmup THROUGH the adapter: proves the whole serving path at boot.
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )

    def _load_adapter(self, name: str) -> str:
        """Swap the resident adapter; returns the adapter_source tag."""
        if self._resident == name:
            return "cached"
        path = shared.lora_weights_path(name)
        if not path.exists():
            raise FieldException(
                field="lora_name",
                message=(
                    f"no adapter at /data/loras/{name}/ - train one first "
                    "(app_trainer.py /train) or check the slug"
                ),
                type="lora_not_found",
            )
        try:
            shared.validate_safetensors_headers(path)
        except shared.ArtifactError as exc:
            raise FieldException(
                field="lora_name", message=str(exc), type="lora_invalid_format"
            ) from None

        if self._adapter_id is not None:
            self.pipe.delete_adapters(self._adapter_id)  # frees the old slot's VRAM
            self._resident = None
            self._adapter_id = None
        adapter_id = f"{name}-{self._counter}"  # loaded names can't be reused
        self._counter += 1
        try:
            self.pipe.load_lora_weights(
                str(path), adapter_name=adapter_id, use_safetensors=True
            )
        except RuntimeError as exc:
            if "size mismatch for" in str(exc):
                # The diagnostic that answers most bad-LoRA support tickets.
                raise FieldException(
                    field="lora_name",
                    message=(
                        "adapter tensors don't fit this model - the file was "
                        "likely trained for a different architecture or size "
                        "(e.g. a 9B or FLUX.2-dev LoRA on klein-4B)"
                    ),
                    type="lora_incompatible",
                ) from None
            raise  # CUDA/OOM faults stay 5xx: retryable, not the caller's fault
        if adapter_id not in getattr(self.pipe.transformer, "peft_config", {}):
            # diffusers "loads" foreign-prefix files as a silent no-op; make it loud.
            raise FieldException(
                field="lora_name",
                message="file contains no FLUX.2 transformer LoRA keys",
                type="lora_invalid_format",
            )
        self._log_key_audit(path, adapter_id)
        self._resident = name
        self._adapter_id = adapter_id
        return "seeded" if name in self._seeded else "disk"

    def _log_key_audit(self, path: Path, adapter_id: str) -> None:
        """'N/M modules loaded' - the other half of LoRA debuggability."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        sd = type(self.pipe).lora_state_dict(str(path))
        file_modules = {
            k.removeprefix("transformer.").removesuffix(".lora_A.weight")
            for k in sd
            if k.endswith(".lora_A.weight")
        }
        injected = {
            n
            for n, m in self.pipe.transformer.named_modules()
            if isinstance(m, BaseTunerLayer) and adapter_id in m.lora_A
        }
        loaded = len(file_modules & injected)
        print(f"[lora] {loaded}/{len(file_modules)} adapter modules loaded from {path}")
        if loaded < len(file_modules):
            print(f"[lora] unused modules (first 5): {sorted(file_modules - injected)[:5]}")

    @fal.endpoint("/")
    def generate(self, request: GenerateInput) -> GenerateOutput:
        import torch

        t0 = time.monotonic()
        source = self._load_adapter(request.lora_name)
        self.pipe.set_adapters(self._adapter_id, adapter_weights=request.lora_scale)
        t_lora = time.monotonic() - t0

        size = get_image_size(request.image_size)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        t1 = time.monotonic()
        result = self.pipe(
            prompt=request.prompt,
            height=size.height,
            width=size.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0,  # klein is guidance-distilled; the arg is ignored
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        t_inference = time.monotonic() - t1

        repository = "in_memory" if request.sync_mode else "fal_v3"
        image = Image.from_pil(result.images[0], repository=repository)
        return GenerateOutput(
            image=image,
            seed=seed,
            timings={"lora": round(t_lora, 3), "inference": round(t_inference, 3)},
            serving=ServingInfo(lora_name=request.lora_name, adapter_source=source),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -x -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/
git commit -m "feat(04-genmedia-patterns): train-serve-lora serving app - one-slot adapter swap + /data seeding (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `test_app.py` — live harness suites (variant-branched)

**Files:**
- Modify: `04-genmedia-patterns/train-serve-lora/test_app.py` (append)

**Interfaces:**
- Consumes: the `app_client` fixture (`harness/pytest_plugin.py` — one method: `run(path, payload) -> dict`; raises `httpx.HTTPStatusError` on 4xx/5xx); `HARNESS_VARIANT` env (`"app"` = server, `"app_trainer"` = trainer).
- Produces: the CI contract for both deploys.

- [ ] **Step 1: Append the live tests**

```python
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
        app_client.run("/", {"image_data_url": "https://fal.ai"})
    detail = assert_typed_422(excinfo, "image_data_url", "invalid_archive")
    assert "ZIP" in detail["msg"]


@trainer_only
def test_preflight_rejects_unroutable_host(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/", {"image_data_url": "http://169.254.169.254/latest/meta-data/"}
        )
    assert_typed_422(excinfo, "image_data_url", "file_download_error")
```

- [ ] **Step 2: Verify unit tests still pass and live tests skip locally**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -q
```

Expected: unit tests PASS, all live tests SKIP (`no live app under test`).

- [ ] **Step 3: Commit**

```bash
git add 04-genmedia-patterns/train-serve-lora/test_app.py
git commit -m "test(04-genmedia-patterns): train-serve-lora live suites for both variants (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: 💸 Trainer live smoke via `fal run` (preflight + import-code verification)

**Files:** none (verification only; fixes commit under this task if needed).

- [ ] **Step 1: 💸 Boot the trainer ephemerally**

```bash
cd 04-genmedia-patterns/train-serve-lora && fal run app_trainer.py
```

Expected: env build, ~16 GB first-boot download (cached on /data thereafter), then a printed ephemeral URL. **This is also the `app_files=["shared.py"]` verification** — if boot fails with `ModuleNotFoundError: shared`, switch both apps to `local_python_modules = ["shared"]` (the other documented Import Code mechanism) and note the swap in the README's How-it-works.

- [ ] **Step 2: Smoke the preflight against the live URL (second terminal)**

```bash
curl -s -X POST "<EPHEMERAL_URL>/" -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" -d '{"image_data_url": "https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/dataset/paper-diorama-16.zip"}' | python3 -m json.tool
```

Expected: `{"ready": true, "num_images": 16, "num_sidecar_captions": 16, ...}`. Ctrl-C the `fal run` afterwards.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A 04-genmedia-patterns/train-serve-lora/
git commit -m "fix(04-genmedia-patterns): train-serve-lora fixes from live trainer smoke (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(Skip the commit if nothing changed.)

---

### Task 9: 💸 The one-time validated `/train` run + publish the reference adapter (human gate)

**Files:** none in-repo (artifacts go to HF; grid images feed Task 10).

This closes the seam no automated test exercises: a real train, served for real. **All steps here spend real money and need the owner's go.**

- [ ] **Step 1: 💸 Deploy both apps registered (not ephemeral)**

```bash
cd 04-genmedia-patterns/train-serve-lora && fal deploy app_trainer.py && fal deploy app.py
```

Expected: two app URLs printed, e.g. `<username>/klein-lora-trainer` and `<username>/klein-lora-server` (actual aliases come from the class names; note them).

- [ ] **Step 2: 💸 Queue-submit the real training (~1000 steps)**

```python
# validated_train.py - run from anywhere with FAL_KEY; poll, don't stream.
import fal_client

handle = fal_client.submit(
    "<username>/klein-lora-trainer",  # from Step 1
    path="/train",
    arguments={
        "image_data_url": "https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/dataset/paper-diorama-16.zip",
        "trigger_phrase": "pprdrma style,",
        "lora_name": "paper-diorama",
        "steps": 1000,
        "learning_rate": 5e-5,
    },
)
print("request id:", handle.request_id)
for event in handle.iter_events(with_logs=True):
    print(event)
result = handle.get()
print(result["lora_file"]["url"], result["data_path"], result["final_loss"])
```

Expected: `[train] step N/1000 ...` logs (first-step silence up to a couple of minutes = CUDA warm-up, not a hang), then a CDN URL + `/data/loras/paper-diorama/...`. Save the CDN URL.

- [ ] **Step 3: 💸 Serve the fresh adapter and produce the README grid**

```python
# grid_shots.py - the server was deployed BEFORE the reference adapter was
# published, so its setup() found /data/loras/paper-diorama/ ALREADY there
# from Step 2 if trainer ran first; otherwise it seeded from HF. Either way:
import fal_client, httpx, pathlib

PROMPT = "pprdrma style, a tram crossing a city bridge at sunset"
for name, args in {
    "base": {"prompt": PROMPT.removeprefix("pprdrma style, "), "lora_scale": 0.0, "seed": 7},
    "scale-07": {"prompt": PROMPT, "lora_scale": 0.7, "seed": 7},
    "scale-10": {"prompt": PROMPT, "lora_scale": 1.0, "seed": 7},
}.items():
    r = fal_client.subscribe("<username>/klein-lora-server", arguments=args)
    pathlib.Path(f"grid-{name}.png").write_bytes(httpx.get(r["image"]["url"]).content)
    print(name, r["serving"], r["timings"])
```

Expected: three images; `scale-10` visibly paper-diorama, `base` visibly not. **This validates base-trained-adapter-on-distilled end to end.** If the style is weak at 1000 steps, re-run Step 2 with `steps: 1500` (BFL's style band is 750–1500) before publishing.

- [ ] **Step 4: Publish the reference adapter + grid (human gate: HF org perms)**

```bash
curl -sL -o /tmp/pytorch_lora_weights.safetensors "<CDN URL from Step 2>"
hf upload fal/flux-2-klein-4b-paper-diorama-lora /tmp/pytorch_lora_weights.safetensors pytorch_lora_weights.safetensors --repo-type=model
hf upload fal/flux-2-klein-4b-paper-diorama-lora grid-base.png images/grid-base.png --repo-type=model
hf upload fal/flux-2-klein-4b-paper-diorama-lora grid-scale-07.png images/grid-scale-07.png --repo-type=model
hf upload fal/flux-2-klein-4b-paper-diorama-lora grid-scale-10.png images/grid-scale-10.png --repo-type=model
curl -sI "https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/pytorch_lora_weights.safetensors" | head -1
```

Expected: `HTTP/2 302`. Also set the HF repo license to Apache-2.0 and paste a short model card (what it is, the dataset provenance = klein-generated, the trigger phrase, a link back to the example folder).

- [ ] **Step 5: Tear down the validation deploys (they bill while warm)**

```bash
fal apps list
fal apps delete <username>/klein-lora-trainer && fal apps delete <username>/klein-lora-server
```

Expected: both gone from `fal apps list`. (If `apps delete` isn't the 1.78.2 verb, use the dashboard — the point is no warm runners left behind.)

---

### Task 10: README, tier row, llms.txt

**Files:**
- Create: `04-genmedia-patterns/train-serve-lora/README.md`
- Modify: `04-genmedia-patterns/README.md:12` (the 04.4 row)
- Modify: `llms.txt` (append under `## Live examples`)

**Interfaces:**
- Consumes: grid image URLs from Task 9 (`https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-*.png`).

- [ ] **Step 1: Write `README.md`** (template §4-A heading order is mandatory)

```markdown
# Train a LoRA → serve it

You shipped an image feature; now users want *their* style. This example
closes the loop on fal serverless: POST a ZIP of ~16 style images to your own
trainer app, get back a LoRA fine-tuned on FLUX.2-klein-**base**-4B, and serve
it from `/data` on the **distilled** klein-4B at 4 steps — train where
gradients behave, serve where inference is fast.

## What you get

| base model (`lora_scale: 0`) | `lora_scale: 0.7` | `lora_scale: 1.0` |
|---|---|---|
| ![base](https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-base.png) | ![0.7](https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-scale-07.png) | ![1.0](https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-scale-10.png) |

Same prompt, same seed — `pprdrma style, a tram crossing a city bridge at
sunset`. The adapter was trained by this folder's own trainer on 16
klein-generated images ([dataset](https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora/tree/main/dataset), [how it was made](#how-the-demo-was-made)).

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py            # the server, seeded with the demo adapter
```

Then train your own (deploy first — training runs go through the queue):

```bash
fal deploy app_trainer.py
```

```python
import fal_client

# 1. preflight - free sanity check on your dataset
print(fal_client.run("<username>/klein-lora-trainer", arguments={
    "image_data_url": "https://.../your-style.zip",
    "trigger_phrase": "mystyle style,",
}))

# 2. train - submit to the queue and poll; a full run takes a few minutes
handle = fal_client.submit("<username>/klein-lora-trainer", path="/train",
    arguments={
        "image_data_url": "https://.../your-style.zip",
        "trigger_phrase": "mystyle style,",
        "lora_name": "my-style",
        "steps": 1000,
    })
for event in handle.iter_events(with_logs=True):
    print(event)
print(handle.get()["data_path"])   # -> /data/loras/my-style/...

# 3. serve it - the server loads your adapter from /data by name
print(fal_client.run("<username>/klein-lora-server", arguments={
    "prompt": "mystyle style, a lighthouse at dusk",
    "lora_name": "my-style",
}))
```

## How it works

```
ZIP of images ──▶ app_trainer.py /train ──▶ /data/loras/<name>/ ──▶ app.py /
                  (klein-base-4B,            (atomic temp+rename)    (distilled
                   LoRA gradients)                                    klein-4B,
                                                                      4 steps)
```

- **Two apps, one folder, one shared module.** `shared.py` (dataset gates,
  `/data` conventions, artifact checks) ships to both runners via
  `app_files` — [Import Code](https://docs.fal.ai/serverless/development/import-code).
  Unit tests exercise the exact code the runners execute.
- **The trainer's settings block is the long-job pattern**: `keep_alive=0`,
  `request_timeout=7200`, `skip_retry_conditions=["timeout"]` (a timed-out
  training must not be silently re-run and re-billed), `max_concurrency=1`.
- **`/data` is the handoff** — mounted account-wide across all your apps
  ([persistent storage](https://docs.fal.ai/serverless/development/use-persistent-storage)).
  The trainer writes to a temp file *on /data* and `os.rename`s it into
  place: readers see the old file or the new one, never a partial write.
  The training response returns only after the rename — job completion IS
  the readiness signal.
- **The loop is the official recipe, inlined.** `/train` runs the core of
  HF's [klein DreamBooth script](https://github.com/huggingface/diffusers/blob/v0.39.0/examples/dreambooth/train_dreambooth_lora_flux2_klein.py)
  (pinned at the same diffusers 0.39.0 the server pins): precompute text
  embeddings + VAE latents once, offload both encoders to CPU, train
  transformer-only with flow matching. ~120 lines you can actually read.
- **Train on base, serve on distilled.** Gradients want the undistilled
  `klein-base-4B`; the saved adapter (diffusers-native keys) loads straight
  onto distilled `klein-4B` for 4-step inference — BFL's recommended split.
- **Validation before GPU time.** `POST /` is a dataset preflight (the exact
  head of `/train`): zip-slip-safe unpack, decode-check on every image,
  caption pairing — every user-fixable problem is a typed 422 naming the
  offending file, before a single training step is paid for. (Hosted fal
  trainers validate inline instead; the split here is for cheap CI.)

## Deploy

```bash
fal deploy app_trainer.py && fal deploy app.py
```

Both apps are `app_auth="private"`; the playground and the snippets above
authenticate with your key. CI re-verifies this example on every push: both
apps deploy ephemerally, the trainer's preflight and the server's generate
are smoked, and this folder's test suite runs against each live app (the
training run itself was validated once against a real deployment — the
demo adapter *is* that run's output).

## Cost & tuning

- **Training cost is `steps × seconds-per-step` on your `machine_type`** — see
  [pricing](https://fal.ai/pricing). Two cheap levers while you iterate:
  `RESOLUTION = 512` (a constant in `app_trainer.py`) roughly quarters
  per-step work, and `machine_type = "GPU-L40"` trades speed for a lower
  rate (klein-4B LoRA training fits comfortably in 48 GB).
- **Steps**: BFL's guidance for style LoRAs is 750–1500; the demo adapter
  used 1000. Below ~500 expect drift, not style.
- **`keep_alive=0` on the trainer** means every job pays boot time but idle
  costs nothing — right for occasional training. If you're queueing many
  jobs back-to-back, raise it so the 16 GB weights load once.
- **The server** follows 00.2's economics (`keep_alive=300`, L40-first) —
  adapter swaps add one `/data` read + `load_lora_weights` call, visible in
  `timings.lora` and `serving.adapter_source`.

## How the demo was made

`make_dataset.py` generated the 16 training images with klein itself (fixed
prompts + seeds, one consistent paper-diorama style suffix, captions describe
content only) — no external dataset, no licensing questions. The published
[reference adapter](https://huggingface.co/fal/flux-2-klein-4b-paper-diorama-lora)
is the output of this folder's own `/train` on that ZIP (1000 steps,
lr 5e-5) — the server seeds it into `/data/loras/paper-diorama/` at boot if
missing, so the example works before your first training.

## Next steps

- **Serve thousands of these** — one adapter per user, LRU-cached on a warm
  fleet with routing hints: example 04.3. Your `lora_file` URL is drop-in
  for its `/register`.
- **Managed alternative**: [fal's hosted klein trainer](https://fal.ai/models/fal-ai/flux-2-klein-4b-base-trainer)
  is this pattern productized (same field vocabulary, no app to run).
- **Cold starts**: the 16 GB boot download is example 02.1's problem —
  FlashPack, `/data` caches, and compiled artifacts.
- **Progress streaming**: training logs stream through the queue; for
  first-class SSE progress see example 03.1.
```

- [ ] **Step 2: Update the tier row** in `04-genmedia-patterns/README.md`

Old:

```markdown
| `04.4` | Train a LoRA → serve it (closes the WAN training loop) | `needs-deploy` | [SERV-1312](https://linear.app/features-and-labels/issue/SERV-1312) |
```

New:

```markdown
| `04.4` | Train a LoRA → serve it (klein-base gradients, distilled serving, /data handoff) — [`train-serve-lora/`](train-serve-lora/) | `needs-deploy` | [SERV-1312](https://linear.app/features-and-labels/issue/SERV-1312) |
```

- [ ] **Step 3: Append the llms.txt entry** under `## Live examples`

```markdown
- [04.4 train-serve-lora](04-genmedia-patterns/train-serve-lora/README.md): train a FLUX.2-klein LoRA in-app and serve it - long-request trainer pattern (keep_alive=0, request_timeout, skip_retry), dataset preflight with typed 422s, atomic /data handoff, one-slot adapter serving on distilled klein
```

- [ ] **Step 4: Verify + commit**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -q && git add 04-genmedia-patterns/ llms.txt && git commit -m "docs(04-genmedia-patterns): train-serve-lora README, tier row, llms.txt (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: 💸 Full harness deploy-check (both variants)

**Files:** none (verification; fixes commit under this task).

- [ ] **Step 1: 💸 Run the deploy-check exactly as CI does**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python harness/check.py --example 04-genmedia-patterns/train-serve-lora
```

Expected: `app.py` deploys (boot downloads distilled weights + seeds the demo adapter from HF on a fresh account), `/health` OK, smoke on `/` generates with the demo adapter, server live tests pass, teardown confirmed; then the same cycle for `app_trainer.py` with preflight smoke + trainer live tests. Exit code 0. First boot of each app can take several minutes (~16 GB each, once per account) — the harness startup timeout (600 s) may need `HARNESS_STARTUP_TIMEOUT=1200` on a cold account.

- [ ] **Step 2: Commit any fixes; re-run until green**

```bash
git add -A 04-genmedia-patterns/train-serve-lora/
git commit -m "fix(04-genmedia-patterns): train-serve-lora deploy-check fixes (SERV-1312)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Land it

**Files:**
- Verify only (branch `claude/lora-training-serving-946bd2` → PR to `main`).

- [ ] **Step 1: Full local test sweep**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/lora-training-serving-946bd2 && python -m pytest 04-genmedia-patterns/train-serve-lora/test_app.py -q && git log --oneline main..HEAD
```

Expected: unit tests pass (live skip), commit list reads as the task sequence above.

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat(04-genmedia-patterns): train a LoRA and serve it (SERV-1312)" --body "$(cat <<'EOF'
Catalog entry 04.4: trainer app (klein-base-4B, in-app PEFT flow-matching loop
from the official diffusers v0.39.0 klein script) + single-adapter serving app
(distilled klein-4B), handoff via atomic /data writes. Spec:
docs/superpowers/specs/2026-08-03-train-serve-lora-design.md.

- CI: both variants deploy-checked; trainer smoke = dataset preflight (no
  training in CI); server seeds the published reference adapter.
- Validated deployment: real 1000-step /train run served end-to-end; the
  published demo adapter is that run's output.
- Retarget note: SERV-1312 predates 04.3's model choice ("WAN training
  loop") - retargeted to FLUX.2-klein per owner decision 2026-08-03.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Comment on SERV-1312** (via Linear MCP at execution time)

Post: the WAN→klein retarget rationale (serve side reuses 04.3's model; adapter feeds its `/register`; old WAN docs page superseded in the fal-docs phase), a link to the spec file and PR, and the note that the DoD's "measured numbers" line reads as "passes the CI deploy-check" per the 2026-07-21 descope. Flag to Rehan that the ticket is assigned to Jim (reassign or hand off the fal-docs phase).

- [ ] **Step 4: Merge per repo flow** (owner approves; squash-merge like #2–#7), then delete the worktree branch.

---

## Self-review (done at plan-writing time)

- **Spec coverage**: story/shape (T4–T6), API surface incl. product-matching bounds (T4/T5/T6), training internals 1–5 (T5), artifact gate + atomic publish (T1/T5), server seeding + warmup-through-adapter (T6), error taxonomy 422-vs-5xx (T4/T5/T6), CI split with no `/train` in CI (T7/T11), unit tests for shared (T1/T2), manual validation protocol (T9), demo assets via klein (T3), README §4-A + tier row + llms.txt (T10), landing checklist (T12). Fal-docs page: explicitly out of scope (spec: separate later phase).
- **Known deviations from spec, both deliberate**: (1) `text_encoder_out_layers` uses the pipeline default rather than the script CLI default — argued inline in T5, serves the spec's train/serve-consistency goal; (2) trainer smoke-ZIP is the published demo dataset, which also makes the preflight smoke a real end-to-end test of the documented dataset format.
- **Sequencing hazards**: T3 must complete (ZIP published) before T8/T11 trainer smokes; T9 must complete (adapter published) before T11's server check passes on a fresh account. Both flagged in-task.
- **Type consistency checked**: `shared.*` names used in T4–T7 match T1/T2 definitions; `PreflightInput`/`TrainInput` inheritance consistent; `adapter_source` enum values consistent across app.py, ServingInfo docs, and live tests.
