# 02.1 · Cut cold starts by Nx (FlashPack + /data + compiled caches, WAN 2.1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build catalog slot 02.1 — a baseline vs optimized WAN 2.1 text-to-video app pair that demonstrates fal's three cold-start levers (/data persistence, FlashPack loading, shared inductor caches) with a measured before/after, validated on real ephemeral deploys.

**Architecture:** Two self-contained fal apps sharing one endpoint schema — `app_baseline.py` (no levers: fp32 Hub download + cast + full `torch.compile` every boot) and `app.py` (pack-once to `/data` at serving dtypes, `from_pretrained_flashpack`, `synchronized_inductor_cache`) — plus `wan_flashpack.py`, a factory module holding the four flagship classes from FlashPack's own WAN test, shipped to the runner via `local_python_modules` with a runtime `sys.modules` self-registration that makes the transformers mixin and diffusers component resolution work on fal.

**Tech Stack:** fal 1.78.2, flashpack 0.4.0, diffusers 0.39.0 (WanPipeline), transformers 5.3.0 (UMT5), torch 2.12.1, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, GPU-H100.

**Spec:** `docs/superpowers/specs/2026-08-10-flashpack-wan-cold-starts-design.md` · **Ticket:** [SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286) · **Gate to comment on:** [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271)

## Global Constraints

- Example folder: `02-scaling-cold-starts/cut-cold-starts/` — never create files outside it except: tier README row, `.github/workflows/check.yml` (Task 6), and the docs under `docs/superpowers/`.
- Pins (identical in both apps' `App.requirements` AND `requirements.txt`, byte-for-byte, in this order):
  `fal==1.78.2`, `flashpack==0.4.0`, `diffusers==0.39.0`, `transformers==5.3.0`, `torch==2.12.1`, `accelerate==1.14.0`, `imageio==2.37.4`, `imageio-ffmpeg==0.6.0`
- `machine_type = "GPU-H100"` (pinned — inductor caches are per GPU type), `app_auth = "private"`, `keep_alive = 60`, `startup_timeout = 2700` on BOTH apps.
- Top-level imports in `app.py`, `app_baseline.py`, `wan_flashpack.py`: **only** stdlib + `fal` + `pydantic` (+ `wan_flashpack` itself in app.py). torch/diffusers/transformers/flashpack are imported **inside functions** — CI's static job loads every `app*.py` with only `fal, httpx, pytest` installed.
- Python 3.11. Every commit message references SERV-1286.
- Never publish a performance number that did not come from a dated `bench.py` run (CONTEXT.md fact #2).
- GPU-spending steps (Tasks 4, 7) require fal credentials: run `python -c "from fal.sdk import get_credentials; print(get_credentials().to_headers().keys())"` first; if it raises, stop and ask the user to `fal auth login` or set `FAL_KEYFILE`/`FAL_KEY`.
- The local check venv: `python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt` (repo root; `.venv/` is gitignored). All "Run:" gates below use `.venv/bin/python`.

## Verified platform facts the code below relies on (do not re-derive)

1. **fal `local_python_modules` is cloudpickle by-value registration** (`fal/_serialization.py::include_module`). A by-value module reconstructs runner-side as a working module **object** but is NOT inserted into `sys.modules` (verified empirically with fal 1.78.2 + cloudpickle 3.1.2). Consequences: transformers 5.x crashes with `KeyError` on `sys.modules[cls.__module__]` when instantiating a `PreTrainedModel` subclass, and diffusers' pipeline loader cannot `importlib.import_module("wan_flashpack")` to resolve component classes from `model_index.json`. **Fix (verified locally end-to-end): `_publish_classes()` in Task 1 installs a real `sys.modules` entry at runtime; `importlib` then resolves through it.**
2. **Never chunk-write packs directly onto `/data`** (~0.1 MB/s measured July 2026). Pack to local disk, one bulk `copytree` to a `/data` temp dir, atomic `rename`. Reads from `/data` are fast (~544 MB/s).
3. `flashpack==0.4.0` (public wheel) == repo `main` except one CI-only commit. `FlashPackDiffusionPipeline.from_pretrained_flashpack(dir, device_map="cuda")` is valid (`SUPPORTED_DEVICE_MAP = ["balanced", get_device()]` is flashpack's own). `save_pretrained_flashpack(dir)` writes every mixin component as `<component>/model.flashpack` + config, tokenizer/scheduler via their standard save methods, and `model_index.json` — the pack dir is fully self-contained.
4. diffusers 0.39.0 `WanPipeline.__init__(tokenizer, text_encoder, vae, scheduler, transformer=None, transformer_2=None, boundary_ratio=None, expand_timesteps=False)`; `__call__` defaults `height=480, width=832, num_frames=81, num_inference_steps=50, guidance_scale=5.0, output_type="np"`. `export_to_video` uses imageio + imageio-ffmpeg. Dict `torch_dtype` (per-component + `"default"`) is supported by `from_pretrained`.
5. Model repo `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` ships fp32: text_encoder 22.7 GB (5 shards), transformer 5.7 GB, VAE 0.5 GB (~29 GB download). At serving dtypes (bf16/bf16/fp32) the FlashPack pack is ~15 GB. `model_index.json` scheduler is `UniPCMultistepScheduler`.
6. Harness (`harness/check.py`): discovers `app*.py` variants per example, deploys each independently, health-checks, smokes `POST /` with a payload built from **every Input field that carries `examples=[...]`** (required fields must have them; optional fields with examples are also included), then runs `test_app.py` with `HARNESS_APP_URL`/`HARNESS_VARIANT` set. Timeouts (env-overridable): `HARNESS_URL_TIMEOUT=900`, `HARNESS_STARTUP_TIMEOUT=600`, `HARNESS_REQUEST_TIMEOUT=120`, `HARNESS_TEST_TIMEOUT=600`. WAN-scale boots need `HARNESS_STARTUP_TIMEOUT=2700` (Task 6 wires this into CI; local gates pass it inline).
7. WAN frame counts must be `4k+1` (VAE temporal compression 4). 81 frames @ 16 fps = 5 s. Latent T for 17 frames = 5, for 81 = 21 — warming both extremes under `torch.compile(dynamic=True)` marks the dim dynamic.

---

### Task 1: Scaffold + `wan_flashpack.py` + free unit tests

**Files:**
- Create: `02-scaling-cold-starts/cut-cold-starts/wan_flashpack.py`
- Create: `02-scaling-cold-starts/cut-cold-starts/test_app.py`
- Create: `02-scaling-cold-starts/cut-cold-starts/requirements.txt`
- Create: `02-scaling-cold-starts/cut-cold-starts/README.md` (interim copy of `templates/example/README.md`, finalized in Task 7)

**Interfaces:**
- Produces: `wan_flashpack.get_classes() -> dict[str, type]` with keys `FlashPackWanTransformer3DModel`, `FlashPackAutoencoderKLWan`, `FlashPackUMT5EncoderModel`, `FlashPackWanPipeline` (Tasks 3–4 consume). `wan_flashpack._publish_classes(module_name: str, classes: dict[str, type]) -> types.ModuleType` (unit-tested here).
- Produces: `requirements.txt` with the Global Constraints pins (Tasks 2–3 mirror it in `App.requirements`).

- [ ] **Step 1: Write the failing unit tests**

Create `02-scaling-cold-starts/cut-cold-starts/test_app.py`:

```python
"""One suite for both variants plus free (no-GPU, no-deploy) unit tests.

Live tests use the harness `app_client` fixture (skips outside harness runs).
The harness deploys every app*.py in this folder and runs this file against
each; HARNESS_VARIANT says which one is live. Free tests exercise the shared
module logic that must hold before any GPU is spent.
"""
import os
import sys
import types
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wan_flashpack  # top level is stdlib-only by design

EXPECTED_VARIANT = {"app": "optimized", "app_baseline": "baseline"}
# The same small/seeded request the harness smoke uses (Field examples).
SMALL_REQUEST = {
    "prompt": "A calico cat surfing a small wave at golden hour, cinematic",
    "num_frames": 17,
    "num_inference_steps": 8,
    "seed": 42,
}


def variant() -> str:
    return os.environ.get("HARNESS_VARIANT", "app")


# --------------------------- free unit tests -------------------------------

def test_publish_classes_creates_missing_module_entry():
    """On a fal runner the module arrives as a cloudpickle by-value
    reconstruction with NO sys.modules entry; _publish_classes must create
    one so transformers introspection and diffusers importlib resolution work."""
    name = "wan_flashpack_test_ghost"
    sys.modules.pop(name, None)
    try:
        mod = wan_flashpack._publish_classes(name, {"Marker": dict})
        assert sys.modules[name] is mod
        assert mod.Marker is dict
        import importlib
        assert importlib.import_module(name) is mod
    finally:
        sys.modules.pop(name, None)


def test_publish_classes_is_idempotent_on_real_modules():
    """When the module is normally importable (local runs), the existing
    entry must be reused, not replaced."""
    before = sys.modules["wan_flashpack"]
    mod = wan_flashpack._publish_classes("wan_flashpack", {})
    assert mod is before
    assert sys.modules["wan_flashpack"] is before


def test_module_top_level_is_import_light():
    """Static CI loads app files with only fal+httpx+pytest installed; the
    heavy imports must all live inside get_classes()."""
    import ast
    tree = ast.parse((HERE / "wan_flashpack.py").read_text())
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.add((node.module or "").split(".")[0])
    assert top_level_imports <= {"sys", "types", "__future__"}, top_level_imports
```

- [ ] **Step 2: Run tests to verify they fail**

Run (repo root): `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'wan_flashpack'`

- [ ] **Step 3: Write `wan_flashpack.py`**

```python
"""The four FlashPack + WAN 2.1 classes, exactly as FlashPack's flagship WAN
example defines them (github.com/fal-ai/flashpack, tests/test_wan_pipeline.py),
wrapped in a factory so this module's top level stays stdlib-only — the
deploying client and CI's static checks import this file without torch.

Shipped to the runner via `local_python_modules` (cloudpickle by-value). A
by-value module reconstructs WITHOUT a sys.modules entry, which breaks two
things this example needs:

  - transformers 5.x reads sys.modules[cls.__module__] when instantiating any
    PreTrainedModel subclass -> KeyError;
  - diffusers' pipeline loader resolves the component classes recorded in the
    pack's model_index.json via importlib.import_module("wan_flashpack")
    -> ModuleNotFoundError.

_publish_classes() repairs both at runtime by installing a real module entry.
"""
from __future__ import annotations

import sys
import types

_CLASSES: dict[str, type] = {}


def _publish_classes(module_name: str, classes: dict[str, type]) -> types.ModuleType:
    """Make `module_name` importable with `classes` as attributes.

    Reuses the existing sys.modules entry when the module was imported
    normally (local runs, tests); creates one when running from a cloudpickle
    by-value reconstruction (fal runner).
    """
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    for name, cls in classes.items():
        setattr(module, name, cls)
    return module


def get_classes() -> dict[str, type]:
    """Build (once per process) and return the four FlashPack WAN classes."""
    if _CLASSES:
        return _CLASSES

    # Publish the module entry BEFORE defining the classes: transformers may
    # introspect the defining module during class creation, not just __init__.
    _publish_classes(__name__, {})

    from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.pipelines import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from flashpack.integrations.diffusers import (
        FlashPackDiffusersModelMixin,
        FlashPackDiffusionPipeline,
    )
    from flashpack.integrations.transformers import FlashPackTransformersModelMixin
    from transformers import AutoTokenizer, UMT5EncoderModel

    class FlashPackWanTransformer3DModel(
        WanTransformer3DModel, FlashPackDiffusersModelMixin
    ):
        # rope buffers are recomputed at init; packing them would only bloat
        # the file (upstream flagship does the same).
        flashpack_ignore_prefixes = ["rope"]

    class FlashPackAutoencoderKLWan(AutoencoderKLWan, FlashPackDiffusersModelMixin):
        pass

    class FlashPackUMT5EncoderModel(UMT5EncoderModel, FlashPackTransformersModelMixin):
        # tied to shared.weight; transformers re-ties after load.
        flashpack_ignore_names = ["encoder.embed_tokens.weight"]

    class FlashPackWanPipeline(WanPipeline, FlashPackDiffusionPipeline):
        def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: FlashPackUMT5EncoderModel,
            vae: FlashPackAutoencoderKLWan,
            scheduler: UniPCMultistepScheduler,
            transformer: FlashPackWanTransformer3DModel | None = None,
            transformer_2: FlashPackWanTransformer3DModel | None = None,
            boundary_ratio: float | None = None,
            expand_timesteps: bool = False,
        ):
            super().__init__(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                transformer_2=transformer_2,
                scheduler=scheduler,
                boundary_ratio=boundary_ratio,
                expand_timesteps=expand_timesteps,
            )

    _CLASSES.update(
        FlashPackWanTransformer3DModel=FlashPackWanTransformer3DModel,
        FlashPackAutoencoderKLWan=FlashPackAutoencoderKLWan,
        FlashPackUMT5EncoderModel=FlashPackUMT5EncoderModel,
        FlashPackWanPipeline=FlashPackWanPipeline,
    )
    _publish_classes(__name__, _CLASSES)
    return _CLASSES
```

- [ ] **Step 4: Create `requirements.txt` and interim README**

`02-scaling-cold-starts/cut-cold-starts/requirements.txt` (exactly these lines):

```
fal==1.78.2
flashpack==0.4.0
diffusers==0.39.0
transformers==5.3.0
torch==2.12.1
accelerate==1.14.0
imageio==2.37.4
imageio-ffmpeg==0.6.0
```

Copy the README template: `cp templates/example/README.md 02-scaling-cold-starts/cut-cold-starts/README.md` and set the first line to `# Cut cold starts by Nx: FlashPack + /data + compiled caches` (full content lands in Task 7).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: `3 passed` (live tests don't exist yet; free tests green)

- [ ] **Step 6: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/
git commit -m "feat(02-scaling): wan_flashpack factory module + free unit tests (SERV-1286)"
```

---

### Task 2: `app_baseline.py` — the no-levers app

**Files:**
- Create: `02-scaling-cold-starts/cut-cold-starts/app_baseline.py`

**Interfaces:**
- Consumes: `requirements.txt` pins from Task 1 (mirrored verbatim into `App.requirements`).
- Produces: endpoints `POST /` (`TextToVideoInput -> TextToVideoOutput`) and `POST /setup-report` (`SetupReportInput -> SetupReport`) with `SetupReport.variant == "baseline"`. Task 5's live tests and Task 7's bench consume these schemas exactly as written here. Note `TextToVideoInput`/`TextToVideoOutput`/`SetupReport` are deliberately DUPLICATED in app.py (Task 3) — each app file is a self-contained teaching artifact (01.1 precedent); `test_app.py` running identically against both is what keeps them honest.

- [ ] **Step 1: Write `app_baseline.py`**

```python
"""Cut cold starts — the BASELINE app: no levers pulled.

Catalog slot 02.1 (SERV-1286). WAN 2.1 1.3B text-to-video the way everyone
writes it first: every worker boot downloads ~29 GB of fp32 shards from the
Hub, deserializes them through CPU, casts to serving dtypes, moves to GPU,
and compiles kernels from scratch. app.py is the same model and endpoints
with all three cold-start levers pulled; bench.py measures both.
"""
import random
import tempfile
import time
from pathlib import Path

import fal
from fal.toolkit import Video
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WIDTH, HEIGHT = 832, 480      # WAN 2.1 1.3B's native resolution, fixed so
WARMUP_FRAME_COUNTS = (17, 81)  # compile warmup covers every serving shape
FPS = 16


class TextToVideoInput(BaseModel):
    prompt: str = Field(
        description="What to generate. WAN 2.1 rewards detailed, cinematic prompts.",
        examples=["A calico cat surfing a small wave at golden hour, cinematic"],
    )
    negative_prompt: str = Field(
        default="",
        description="What to steer away from (artifacts, styles, objects).",
    )
    num_frames: int = Field(
        default=81,
        ge=17,
        le=81,
        description="Frames at 16 fps; must be 4k+1 (17, 21, ... 81). 81 = 5 s.",
        examples=[17],
    )
    num_inference_steps: int = Field(
        default=28,
        ge=4,
        le=50,
        description="Denoising steps - the quality/latency dial.",
        examples=[8],
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted; echoed back so results reproduce.",
        examples=[42],
    )

    @field_validator("num_frames")
    @classmethod
    def _frames_are_4k_plus_1(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("num_frames must be 4k+1 (17, 21, ..., 81)")
        return v


class TextToVideoOutput(BaseModel):
    video: Video = Field(
        description="The generated MP4 (832x480 @ 16 fps) on the fal CDN."
    )
    seed: int = Field(description="The seed that produced this video.")
    generation_ms: int = Field(
        description="Server-side model time (denoise + decode), excluding upload."
    )


class StageTiming(BaseModel):
    name: str = Field(description="Setup stage name.")
    seconds: float = Field(description="Wall-clock duration of the stage.")


class SetupReportInput(BaseModel):
    """No parameters - POST an empty JSON object."""


class SetupReport(BaseModel):
    variant: str = Field(description="'baseline' or 'optimized'.")
    gpu: str = Field(description="GPU this worker booted on.")
    built_caches_this_boot: bool = Field(
        description="Always False for the baseline: it has no caches to build."
    )
    stages: list[StageTiming] = Field(description="Per-stage setup timings, in order.")
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartBaseline(fal.App, keep_alive=60):
    machine_type = "GPU-H100"
    app_auth = "private"
    startup_timeout = 2700  # ~29 GB Hub download + CPU deserialize + full compile
    # Identical pins to app.py so both share one cached environment and the
    # benchmark isolates the levers, not pip-install noise. requirements.txt
    # mirrors this list exactly (static check enforces it).
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        """Everything from scratch, every boot: download, deserialize, compile."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append(
                {"name": name, "seconds": round(time.perf_counter() - start, 2)}
            )

        t0 = time.perf_counter()
        import torch
        from diffusers import WanPipeline
        stage("imports", t0)

        # The Hub repo ships fp32 (~29 GB); serving dtypes are bf16 for the
        # transformer and text encoder, fp32 for the VAE - the same dtypes
        # the optimized app packs, so the two apps serve identical pipelines.
        t0 = time.perf_counter()
        self.pipe = WanPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype={
                "transformer": torch.bfloat16,
                "text_encoder": torch.bfloat16,
                "vae": torch.float32,
                "default": torch.bfloat16,
            },
        )
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        stage("download + load weights (Hub, safetensors -> CPU -> GPU)", t0)

        # Same compiled end state as app.py - but no shared cache, so every
        # boot pays the full kernel compilation. Warmup at both frame-count
        # extremes marks the latent time dim dynamic; output_type='latent'
        # skips VAE decode (compile only cares about transformer shapes).
        t0 = time.perf_counter()
        self.pipe.transformer = torch.compile(self.pipe.transformer, dynamic=True)
        for frames in WARMUP_FRAME_COUNTS:
            self.pipe(
                prompt="warmup",
                height=HEIGHT,
                width=WIDTH,
                num_frames=frames,
                num_inference_steps=1,
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(0),
            )
        stage("compile + warmup (no cache)", t0)

        self.setup_stages = stages
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: TextToVideoInput) -> TextToVideoOutput:
        import torch
        from diffusers.utils import export_to_video

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        t0 = time.perf_counter()
        frames = self.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=HEIGHT,
            width=WIDTH,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).frames[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            export_to_video(frames, str(path), fps=FPS)
            generation_ms = int((time.perf_counter() - t0) * 1000)
            video = Video.from_path(path)
        return TextToVideoOutput(video=video, seed=seed, generation_ms=generation_ms)

    @fal.endpoint("/setup-report")
    def setup_report(self, request: SetupReportInput) -> SetupReport:
        """What this worker's cold start actually spent its time on."""
        return SetupReport(
            variant="baseline",
            gpu=self.gpu_name,
            built_caches_this_boot=False,
            stages=[StageTiming(**s) for s in self.setup_stages],
            total_seconds=self.setup_total,
        )
```

- [ ] **Step 2: Run the static gate**

Run (repo root): `.venv/bin/python harness/static_check.py`
Expected: exit 0 with NO line for `cut-cold-starts` yet — example discovery keys on `app.py` existing (`harness/check.py::discover_examples`), so the folder is skipped entirely until Task 3 lands. The real gate here is a manual variant load:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from harness.check import load_app_class, pick_endpoint, sample_input
cls = load_app_class(Path('02-scaling-cold-starts/cut-cold-starts/app_baseline.py'))
path, fn = pick_endpoint(cls)
print('app_auth:', cls.app_auth, '| endpoint:', path, '| smoke payload:', sample_input(fn))
"
```

Expected: `app_auth: private | endpoint: / | smoke payload: {'prompt': 'A calico cat surfing a small wave at golden hour, cinematic', 'num_frames': 17, 'num_inference_steps': 8, 'seed': 42}`

Also run the free tests: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/app_baseline.py
git commit -m "feat(02-scaling): baseline WAN 2.1 app - no cold-start levers (SERV-1286)"
```

---

### Task 3: `app.py` — the optimized app (all three levers)

**Files:**
- Create: `02-scaling-cold-starts/cut-cold-starts/app.py`

**Interfaces:**
- Consumes: `wan_flashpack.get_classes()` (Task 1), the schema classes duplicated from Task 2 (identical field-for-field).
- Produces: same two endpoints as Task 2 with `SetupReport.variant == "optimized"` and `built_caches_this_boot` real. `/data` layout: pack at `/data/examples/cut-cold-starts/wan21-t2v-13b-bf16-v1/`, readiness marker inside it, inductor key `cut-cold-starts-wan21-13b/v1`.

- [ ] **Step 1: Write `app.py`**

The schema classes (`TextToVideoInput`, `TextToVideoOutput`, `StageTiming`, `SetupReportInput`, `SetupReport`) and the constants `MODEL_ID`, `WIDTH`, `HEIGHT`, `WARMUP_FRAME_COUNTS`, `FPS` are IDENTICAL to `app_baseline.py` — copy them verbatim (same docstrings, one exception: `SetupReport.built_caches_this_boot`'s description becomes the optimized-app text shown below). Only the module docstring, extra constants, App class, and `setup()` differ:

```python
"""Cut cold starts — the OPTIMIZED app: /data + FlashPack + compiled caches.

Catalog slot 02.1 (SERV-1286). The same WAN 2.1 1.3B endpoint as
app_baseline.py, with every cold-start lever pulled:

  1. /data persistent storage - the ~29 GB fp32 repo is downloaded and
     repacked ONCE per account; every later boot skips the network entirely.
  2. FlashPack loading - the ~15 GB pack streams /data -> GPU directly
     (github.com/fal-ai/flashpack), no safetensors CPU deserialization.
  3. synchronized_inductor_cache - torch.compile kernels are built once,
     synced to /data, and every later boot unpacks them instead of compiling.

wan_flashpack.py holds the integration classes - the exact pattern from
FlashPack's own WAN 2.1 flagship example - and ships to the runner via
local_python_modules. bench.py measures this app against the baseline.
"""
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import fal
from fal.toolkit import Video
from pydantic import BaseModel, Field, field_validator

import wan_flashpack

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WIDTH, HEIGHT = 832, 480
WARMUP_FRAME_COUNTS = (17, 81)
FPS = 16

# Everything this app persists lives under one versioned /data prefix. Bump
# the version when MODEL_ID, serving dtypes, or the torch pin change - a new
# pin must never load an old pack or old kernels.
PACK_VERSION = "v1"
PACK_DIR = Path(f"/data/examples/cut-cold-starts/wan21-t2v-13b-bf16-{PACK_VERSION}")
INDUCTOR_CACHE_KEY = f"cut-cold-starts-wan21-13b/{PACK_VERSION}"
# Written (atomically) only after BOTH caches exist: the FlashPack pack on
# /data AND the synced kernel cache. A worker that crashes mid-setup leaves
# no marker, so the next boot honestly reports itself as cache-building.
READY_MARKER = PACK_DIR / f".ready-{INDUCTOR_CACHE_KEY.replace('/', '-')}"

# ... schema classes: copy from app_baseline.py verbatim, except this field
# description on SetupReport:
#   built_caches_this_boot: "True when this boot found no readiness marker
#   and built cache state (/data pack and/or compiled kernels). Steady-state
#   boots find the marker, reuse both, and report False."


class ColdStartOptimized(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # pin ONE type: inductor caches are per GPU type
    app_auth = "private"
    startup_timeout = 2700  # covers the account's first-ever boot (download + pack + compile)
    local_python_modules = ["wan_flashpack"]
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        """Pack once to /data, stream via FlashPack, compile against the shared cache."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append(
                {"name": name, "seconds": round(time.perf_counter() - start, 2)}
            )

        t0 = time.perf_counter()
        import torch
        from fal.toolkit import synchronized_inductor_cache

        # Builds the FlashPack WAN classes and installs the sys.modules entry
        # that transformers introspection and the pack's model_index.json
        # class resolution both need (see wan_flashpack.py's docstring).
        classes = wan_flashpack.get_classes()
        FlashPackWanPipeline = classes["FlashPackWanPipeline"]
        stage("imports", t0)

        self.built_caches_this_boot = not READY_MARKER.is_file()

        # LEVER 1 - first boot per ACCOUNT (not per worker): download the fp32
        # repo, rebuild at serving dtypes, pack with FlashPack. The pack is
        # written to LOCAL disk and published to /data as one bulk copy +
        # atomic rename: chunked writes straight to /data run at ~0.1 MB/s
        # (measured), and the rename means concurrently booting workers can
        # never observe a half-written pack - the loser of the race cleans up.
        if not PACK_DIR.is_dir():
            t0 = time.perf_counter()
            from diffusers.schedulers import UniPCMultistepScheduler
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            repo_dir = Path(snapshot_download(MODEL_ID))
            transformer = classes["FlashPackWanTransformer3DModel"].from_pretrained(
                repo_dir / "transformer", torch_dtype=torch.bfloat16
            )
            vae = classes["FlashPackAutoencoderKLWan"].from_pretrained(
                repo_dir / "vae", torch_dtype=torch.float32
            )
            text_encoder = classes["FlashPackUMT5EncoderModel"].from_pretrained(
                repo_dir / "text_encoder", torch_dtype=torch.bfloat16
            )
            build_pipe = FlashPackWanPipeline(
                tokenizer=AutoTokenizer.from_pretrained(repo_dir / "tokenizer"),
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=UniPCMultistepScheduler.from_pretrained(repo_dir / "scheduler"),
            )
            local_dir = Path(tempfile.mkdtemp(prefix="wan-flashpack-build-")) / "pack"
            local_dir.mkdir(parents=True)
            build_pipe.save_pretrained_flashpack(str(local_dir))
            del build_pipe, transformer, vae, text_encoder
            stage("download + build FlashPack pack (first boot only)", t0)

            t0 = time.perf_counter()
            PACK_DIR.parent.mkdir(parents=True, exist_ok=True)
            for stale in PACK_DIR.parent.glob(".build-*"):  # crashed builds
                if time.time() - stale.stat().st_mtime > 3600:
                    shutil.rmtree(stale, ignore_errors=True)
            publish_dir = PACK_DIR.parent / f".build-{uuid.uuid4().hex}"
            shutil.copytree(local_dir, publish_dir)
            try:
                publish_dir.rename(PACK_DIR)
            except OSError:
                shutil.rmtree(publish_dir, ignore_errors=True)
                if not PACK_DIR.is_dir():
                    # Nobody else published either: a real storage failure,
                    # not a lost race - fail the boot.
                    raise
            shutil.rmtree(local_dir.parent, ignore_errors=True)
            stage("publish pack to /data (first boot only)", t0)

        # LEVER 2 - every boot: the pack streams /data -> GPU. The pack dir is
        # self-contained (model_index.json + per-component packs + tokenizer +
        # scheduler), so this one call rebuilds the whole pipeline.
        t0 = time.perf_counter()
        pipe = FlashPackWanPipeline.from_pretrained_flashpack(
            str(PACK_DIR), device_map="cuda"
        )
        for component_name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipe, component_name)
            leftover = [
                name
                for name, tensor in (
                    list(module.named_parameters()) + list(module.named_buffers())
                )
                if tensor.is_meta
            ]
            # Must run BEFORE any .to(): meta tensors ride through .to()
            # unchanged, so a later check could be masked.
            if leftover:
                raise RuntimeError(
                    f"{component_name}: tensors not materialized from pack: {leftover[:5]}"
                )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        stage("load weights (FlashPack, /data -> GPU)", t0)

        # LEVER 3 - compile against the shared inductor cache. First boot
        # compiles for real and syncs kernels to /data on context exit; every
        # later boot unpacks them and warmup is cache-hit execution. Warmup
        # MUST run inside the context or nothing is captured.
        t0 = time.perf_counter()
        with synchronized_inductor_cache(INDUCTOR_CACHE_KEY):
            self.pipe.transformer = torch.compile(self.pipe.transformer, dynamic=True)
            for frames in WARMUP_FRAME_COUNTS:
                self.pipe(
                    prompt="warmup",
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=frames,
                    num_inference_steps=1,
                    output_type="latent",
                    generator=torch.Generator("cuda").manual_seed(0),
                )
        stage("compile + warmup (shared inductor cache)", t0)

        # Both caches now exist; publish the readiness marker atomically. A
        # crash anywhere above leaves no marker -> next boot rebuilds honestly.
        if self.built_caches_this_boot:
            marker_tmp = READY_MARKER.with_name(
                f"{READY_MARKER.name}.tmp-{uuid.uuid4().hex}"
            )
            marker_tmp.write_text("ok")
            marker_tmp.rename(READY_MARKER)

        self.setup_stages = stages
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    # generate(): IDENTICAL to app_baseline.py's - copy verbatim.
    # setup_report(): copy from app_baseline.py, with
    #   variant="optimized", built_caches_this_boot=self.built_caches_this_boot
```

(Write the full file — the `# copy verbatim` notes above mean literal duplication from Task 2's code, not stubs.)

- [ ] **Step 2: Run the static gate**

Run: `.venv/bin/python harness/static_check.py`
Expected: `ok   02-scaling-cold-starts/cut-cold-starts`, exit 0 — now covering BOTH variants (app.py + app_baseline.py): loads, `app_auth` set, smoke payload derivable, requirements mirror exact.

Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/app.py
git commit -m "feat(02-scaling): optimized WAN 2.1 app - /data + FlashPack + inductor cache (SERV-1286)"
```

---

### Task 4: GPU spike — prove the mechanism for cents before spending real money

**Files:**
- Create: `<scratchpad>/spike/spike_app.py` (NEVER committed — lives outside the repo)
- Create: `<scratchpad>/spike/run_spike.py`
- Copy: `02-scaling-cold-starts/cut-cold-starts/wan_flashpack.py` → `<scratchpad>/spike/wan_flashpack.py` (the spike app imports it from its own dir)

**Interfaces:**
- Consumes: `wan_flashpack.get_classes()` (Task 1), harness helpers (`spawn_owned`, `wait_for_url`, `wait_healthy`, `teardown`, `RestRunnerPoller`, `BootWatcher`, `auth_headers`, `load_app_class` from `harness.check`).
- Produces: a go/no-go on the flagship-pattern-on-fal mechanism. **GO** → proceed unchanged. **NO-GO** → apply the documented fallback (below) to `app.py` before Task 5.

- [ ] **Step 1: Verify fal credentials**

Run: `.venv/bin/python -c "from fal.sdk import get_credentials; print(type(get_credentials()).__name__)"`
Expected: a credentials class name (e.g. `FalKeyCredentials`/`AuthenticatedUser`). If it raises, STOP — ask the user to authenticate.

- [ ] **Step 2: Write the spike app**

`<scratchpad>/spike/spike_app.py`:

```python
"""Spike (SERV-1286): does the flagship FlashPack WAN pattern work on fal?

Validates, on a real runner, with NO model download:
  1. wan_flashpack ships via local_python_modules and get_classes() runs;
  2. instantiating the FlashPackTransformersModelMixin subclass works (the
     PR #5 crash point: transformers 5.x sys.modules introspection);
  3. sys.modules['wan_flashpack'] + importlib resolution hold runner-side;
  4. a tiny pack round-trips through /data via pack-local-then-copy.
"""
import shutil
import tempfile
import uuid
from pathlib import Path

import fal
from pydantic import BaseModel, Field

import wan_flashpack

SPIKE_DIR = Path("/data/examples/cut-cold-starts/.spike")


class SpikeInput(BaseModel):
    """No parameters - POST an empty JSON object."""


class SpikeReport(BaseModel):
    checks: dict[str, str] = Field(description="check name -> 'ok' or error text")


class MixinSpike(fal.App, keep_alive=60):
    machine_type = "GPU-H100"
    app_auth = "private"
    startup_timeout = 900
    local_python_modules = ["wan_flashpack"]
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        import importlib
        import sys

        import torch
        from flashpack import assign_from_file, pack_to_file
        from transformers import UMT5Config

        checks: dict[str, str] = {}

        def check(name: str, fn) -> None:
            try:
                fn()
                checks[name] = "ok"
            except Exception as exc:  # report, don't crash: we want ALL results
                checks[name] = f"{type(exc).__name__}: {exc}"

        classes = wan_flashpack.get_classes()
        check("get_classes", lambda: classes["FlashPackUMT5EncoderModel"])
        check(
            "sys_modules_entry",
            lambda: sys.modules["wan_flashpack"].FlashPackUMT5EncoderModel,
        )
        check(
            "importlib_resolution",
            lambda: getattr(
                importlib.import_module("wan_flashpack"), "FlashPackWanPipeline"
            ),
        )

        # The PR #5 crash point: PreTrainedModel subclass instantiation.
        tiny_cfg = UMT5Config(
            d_model=64, d_ff=128, num_layers=2, num_heads=2, d_kv=32,
            vocab_size=256, tie_word_embeddings=False,
        )
        state = {}

        def build_tiny():
            state["model"] = classes["FlashPackUMT5EncoderModel"](tiny_cfg)

        check("transformers_mixin_init", build_tiny)

        def pack_roundtrip():
            model = state["model"]
            local = Path(tempfile.mkdtemp()) / "tiny.flashpack"
            pack_to_file(model, str(local), None)
            SPIKE_DIR.mkdir(parents=True, exist_ok=True)
            dest_tmp = SPIKE_DIR / f".tmp-{uuid.uuid4().hex}"
            shutil.copy2(local, dest_tmp)
            dest = SPIKE_DIR / "tiny.flashpack"
            dest_tmp.rename(dest)
            fresh = classes["FlashPackUMT5EncoderModel"](tiny_cfg)
            assign_from_file(fresh, str(dest), device="cuda")
            assert not any(
                t.is_meta for _, t in fresh.named_parameters()
            ), "meta tensors left after assign_from_file"

        check("pack_roundtrip_via_data", pack_roundtrip)
        self.checks = checks

    @fal.endpoint("/")
    def report(self, request: SpikeInput) -> SpikeReport:
        return SpikeReport(checks=self.checks)
```

- [ ] **Step 3: Write the driver**

`<scratchpad>/spike/run_spike.py`:

```python
"""Ephemeral-deploy the spike app, print its checks, tear it down.

Usage: python run_spike.py <repo-root>
"""
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO_ROOT))

from harness.check import (
    BootWatcher, RestRunnerPoller, auth_headers, load_app_class,
    spawn_owned, teardown, wait_for_url, wait_healthy,
)


def main() -> int:
    from fal.app import wrap_app

    app_cls = load_app_class(Path(__file__).parent / "spike_app.py")
    headers = auth_headers()
    fn = wrap_app(app_cls)
    poller = RestRunnerPoller(headers)
    poller.start()
    spawn_info, failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()
    try:
        url = wait_for_url(spawn_info, failure)
        if watcher.runner_id:
            poller.runner_id = watcher.runner_id
        wait_healthy(url, headers, poller)
        resp = httpx.post(
            url.rstrip("/") + "/", json={}, headers=headers, timeout=60
        )
        resp.raise_for_status()
        checks = resp.json()["checks"]
        print(json.dumps(checks, indent=2))
        return 0 if all(v == "ok" for v in checks.values()) else 1
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        watcher.stop()
        if leaked:
            print(f"ERROR: leaked runner {leaked} - terminate it manually")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the spike**

```bash
cp 02-scaling-cold-starts/cut-cold-starts/wan_flashpack.py <scratchpad>/spike/wan_flashpack.py
cd <scratchpad>/spike && <repo>/.venv/bin/python run_spike.py <repo>
```

Expected: all five checks print `"ok"`, exit 0, `runner ... stopped` in the log.

- [ ] **Step 5: Decide**

- **All ok** → record the result (it is SERV-1271 evidence; save the JSON output to `<scratchpad>/spike/spike-result.json`) and continue to Task 5. No commit (nothing in the repo changed).
- **`transformers_mixin_init` failed** → apply the fallback to `app.py` and `wan_flashpack.py`: the text encoder abandons the mixin; build it with `UMT5EncoderModel(config)` under `accelerate.init_empty_weights(include_buffers=False)` + `flashpack.assign_from_file(model, str(PACK_DIR / "text_encoder" / "model.flashpack"), device="cuda")`, pack it at build time with `flashpack.pack_to_file(text_encoder, str(local_dir / "text_encoder" / "model.flashpack"), None)` + `config.save_pretrained(...)`, and pass the loaded model into `from_pretrained_flashpack(..., text_encoder=text_encoder)` (the pipeline loader accepts pre-instantiated components). Re-run the spike with the fallback exercised, then continue.
- **`importlib_resolution` or `sys_modules_entry` failed** → the self-registration assumption broke on the real runner; STOP and report findings to the user before continuing (this invalidates the pipeline-level load path).

---

### Task 5: Live tests in `test_app.py`

**Files:**
- Modify: `02-scaling-cold-starts/cut-cold-starts/test_app.py` (append live tests)

**Interfaces:**
- Consumes: `app_client` fixture (root conftest → `harness/pytest_plugin.py`), `HARNESS_VARIANT` env, endpoint schemas from Tasks 2–3, `EXPECTED_VARIANT` / `SMALL_REQUEST` constants from Task 1.
- Produces: the suite `harness/check.py` runs against BOTH live variants. Deliberate deviation from 01.1's cross-variant transcript ledger: video bytes are not deterministic across hosts/compile, so parity is enforced by the same suite passing on both variants, not by output equality.

- [ ] **Step 1: Append the live tests**

Append to `test_app.py`:

```python
# --------------------------- live tests (harness) ---------------------------
# The harness deploys each app*.py and runs this whole file against it with
# HARNESS_APP_URL + HARNESS_VARIANT set; app_client skips otherwise.


def test_small_seeded_generation(app_client):
    response = app_client.run("/", SMALL_REQUEST)
    assert response["seed"] == 42, "seed must echo back for reproducibility"
    assert response["video"]["url"].startswith("https://"), response["video"]
    assert response["generation_ms"] > 0


def test_setup_report_matches_variant(app_client):
    report = app_client.run("/setup-report", {})
    assert report["variant"] == EXPECTED_VARIANT[variant()]
    assert report["total_seconds"] > 0
    stage_names = [s["name"] for s in report["stages"]]
    assert stage_names, "setup() must record its stages"
    if report["variant"] == "optimized":
        assert any("FlashPack" in name for name in stage_names), stage_names
        assert any("inductor" in name for name in stage_names), stage_names
    else:
        assert report["built_caches_this_boot"] is False


def test_rejects_non_4k_plus_1_frames(app_client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        app_client.run("/", {**SMALL_REQUEST, "num_frames": 18})
    assert exc_info.value.response.status_code == 422
```

- [ ] **Step 2: Verify they skip cleanly outside the harness and free tests still pass**

Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: `3 passed, 3 skipped` (live tests skip without `HARNESS_APP_URL`)

- [ ] **Step 3: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/test_app.py
git commit -m "test(02-scaling): live suite for both cut-cold-starts variants (SERV-1286)"
```

---

### Task 6: CI workflow headroom for video-scale examples

**Files:**
- Modify: `.github/workflows/check.yml:67` (deploy job `timeout-minutes`) and the `Deploy-check examples` step env (after line 85)

**Interfaces:**
- Consumes: nothing. Produces: CI that survives WAN-scale boots (baseline boot and the account's one-time pack+compile boot both exceed the 600 s default).

- [ ] **Step 1: Edit the workflow**

In `.github/workflows/check.yml`, change the deploy job's `timeout-minutes: 60` to `timeout-minutes: 120`, and add `HARNESS_STARTUP_TIMEOUT: "2700"` to the `Deploy-check examples` step's `env:` block, with a comment:

```yaml
  deploy-check:
    # ...existing comments unchanged...
    timeout-minutes: 120  # video-scale examples (02.1+) boot for many minutes
    # ...
      - name: Deploy-check examples
        env:
          FAL_KEY: ${{ secrets.FAL_KEY }}
          EXAMPLE: ${{ inputs.example }} # via env, never interpolated into the script
          # WAN-scale setup() (download+pack or full compile) exceeds the 600s
          # default; healthy apps return early, so this only delays failures.
          HARNESS_STARTUP_TIMEOUT: "2700"
        run: python harness/check.py --example "$EXAMPLE"
```

- [ ] **Step 2: Validate the YAML**

Run: `.venv/bin/python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/check.yml').read_text()); print('yaml ok')"` (if `yaml` is missing: `.venv/bin/pip install pyyaml`)
Expected: `yaml ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/check.yml
git commit -m "ci: raise deploy-check startup headroom for video-scale examples (SERV-1286)"
```

---

### Task 7: `bench.py` + full live validation + measured numbers

**Files:**
- Create: `02-scaling-cold-starts/cut-cold-starts/bench.py`
- Modify: `02-scaling-cold-starts/cut-cold-starts/test_app.py` (append one free unit test for the table formatter)

**Interfaces:**
- Consumes: harness helpers from `harness.check` (same set as Task 4's driver), `/setup-report` schema from Tasks 2–3.
- Produces: `bench.py --variant {app,app_baseline} --runs N` printing per-run JSON lines and a markdown stage table; `format_table(runs: list[dict]) -> str` (unit-tested). The measured numbers Task 8's README embeds.

- [ ] **Step 1: Write the failing formatter test**

Append to `test_app.py`:

```python
def _load_bench():
    """Load bench.py under a unique module name: plain `import bench` could
    collide with another example's bench.py in a whole-repo pytest run."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cut_cold_starts_bench", HERE / "bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_table_formats_stages_in_order():
    bench = _load_bench()

    runs = [
        {
            "spawn_to_healthy_s": 100.0,
            "variant": "optimized",
            "gpu": "H100",
            "built_caches_this_boot": False,
            "stages": [
                {"name": "imports", "seconds": 5.0},
                {"name": "load weights (FlashPack, /data -> GPU)", "seconds": 9.5},
            ],
            "total_seconds": 90.0,
        },
        {
            "spawn_to_healthy_s": 700.0,
            "variant": "optimized",
            "gpu": "H100",
            "built_caches_this_boot": True,
            "stages": [
                {"name": "imports", "seconds": 5.2},
                {"name": "download + build FlashPack pack (first boot only)", "seconds": 400.0},
                {"name": "load weights (FlashPack, /data -> GPU)", "seconds": 12.0},
            ],
            "total_seconds": 650.0,
        },
    ]
    table = bench.format_table(runs)
    lines = table.splitlines()
    assert lines[0].startswith("| stage |")
    assert "| imports | 5.0 | 5.2 |" in table
    # stages absent from a run render as - (dashes), never invented zeros
    assert "| download + build FlashPack pack (first boot only) | - | 400.0 |" in table
    assert "| setup() total | 90.0 | 650.0 |" in table
    assert "| spawn -> healthy | 100.0 | 700.0 |" in table
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py::test_bench_table_formats_stages_in_order -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bench'`

- [ ] **Step 3: Write `bench.py`**

```python
"""Measured before/after for the README table - real ephemeral deploys.

Usage (from the repo root, with fal credentials configured):

    HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app_baseline --runs 2
    HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app --runs 3

Each run: ephemeral deploy -> wait healthy (wall clock = "spawn -> healthy")
-> POST /setup-report -> teardown with runner-stop confirmation. Prints one
JSON line per run and a markdown table over all runs. The first optimized run
on a fresh account builds the /data pack + kernel cache (one-time); later
runs are the steady state every cold start gets.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.check import (  # noqa: E402
    BootWatcher,
    RestRunnerPoller,
    auth_headers,
    load_app_class,
    spawn_owned,
    teardown,
    wait_for_url,
    wait_healthy,
)


def bench_once(app_file: Path) -> dict:
    from fal.app import wrap_app

    app_cls = load_app_class(app_file)
    headers = auth_headers()
    fn = wrap_app(app_cls)
    poller = RestRunnerPoller(headers)
    poller.start()
    started = time.monotonic()
    spawn_info, failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()
    try:
        url = wait_for_url(spawn_info, failure)
        if watcher.runner_id:
            poller.runner_id = watcher.runner_id
        wait_healthy(url, headers, poller)
        spawn_to_healthy_s = round(time.monotonic() - started, 1)
        response = httpx.post(
            url.rstrip("/") + "/setup-report", json={}, headers=headers, timeout=60
        )
        response.raise_for_status()
        return {"spawn_to_healthy_s": spawn_to_healthy_s, **response.json()}
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        watcher.stop()
        if leaked:
            print(f"ERROR: leaked runner {leaked} - terminate it manually")


def format_table(runs: list[dict]) -> str:
    """Markdown: one column per run, stage rows in first-seen order.

    Stages a run did not execute render as '-' (e.g. steady-state boots skip
    the first-boot pack build) - never invented zeros.
    """
    stage_names: list[str] = []
    for run in runs:
        for stage in run["stages"]:
            if stage["name"] not in stage_names:
                stage_names.append(stage["name"])

    def label(i: int, run: dict) -> str:
        kind = "first boot" if run.get("built_caches_this_boot") else "steady"
        return f"run {i + 1} ({run['variant']}, {kind})"

    header = "| stage | " + " | ".join(label(i, r) for i, r in enumerate(runs)) + " |"
    divider = "|---" * (len(runs) + 1) + "|"
    rows = []
    for name in stage_names:
        cells = []
        for run in runs:
            match = [s["seconds"] for s in run["stages"] if s["name"] == name]
            cells.append(str(match[0]) if match else "-")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    rows.append(
        "| setup() total | "
        + " | ".join(str(r["total_seconds"]) for r in runs) + " |"
    )
    rows.append(
        "| spawn -> healthy | "
        + " | ".join(str(r["spawn_to_healthy_s"]) for r in runs) + " |"
    )
    return "\n".join([header, divider, *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["app", "app_baseline"], required=True)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    app_file = EXAMPLE_DIR / f"{args.variant}.py"
    runs = []
    for index in range(args.runs):
        print(f"--- {args.variant} run {index + 1}/{args.runs}")
        report = bench_once(app_file)
        print(json.dumps(report))
        runs.append(report)
    print()
    print(format_table(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the formatter test**

Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q`
Expected: `4 passed, 3 skipped`

- [ ] **Step 5: Full harness validation (spends real money)**

Run (repo root):

```bash
HARNESS_STARTUP_TIMEOUT=2700 .venv/bin/python harness/check.py --example 02-scaling-cold-starts/cut-cold-starts
```

Expected: for EACH variant — deploy, `url:`, health, `smoke response:` (a JSON body with a video URL), `running test_app.py against the live app`, pytest green, `runner ... stopped`. Exit 0. The optimized variant's first-ever run on this account builds the pack (slow, one-time); if the run fails on a timeout during that first build, re-run once — the pack persists.

- [ ] **Step 6: Bench runs for the README numbers**

```bash
HARNESS_STARTUP_TIMEOUT=2700 .venv/bin/python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app_baseline --runs 2 | tee /tmp/bench-baseline.txt
HARNESS_STARTUP_TIMEOUT=2700 .venv/bin/python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app --runs 2 | tee /tmp/bench-optimized.txt
```

Expected: JSON lines + tables. (The one-time pack build already happened in Step 5, so both optimized runs here are steady-state; Step 5's timing for the first boot is in its log — capture `spawn -> healthy` for the first-boot column from a third bench run ONLY if Step 5's log did not record it: in that case delete `READY_MARKER` and `PACK_DIR` is NOT needed — instead run one extra optimized bench run and label it honestly as "kernel-cache rebuild" if it rebuilt, or reuse Step 5's numbers.) Record: date, GPU name, torch/diffusers/flashpack versions alongside the tables.

- [ ] **Step 7: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/bench.py 02-scaling-cold-starts/cut-cold-starts/test_app.py
git commit -m "feat(02-scaling): bench.py - measured before/after via ephemeral deploys (SERV-1286)"
```

---

### Task 8: README + tier README row

**Files:**
- Modify: `02-scaling-cold-starts/cut-cold-starts/README.md` (replace the interim template copy)
- Modify: `02-scaling-cold-starts/README.md:9` (the 02.1 row)

**Interfaces:**
- Consumes: measured tables + metadata from Task 7 Steps 5–6, sample video URL from the smoke/test output.

- [ ] **Step 1: Write the README**

Follow `templates/example/README.md` section order exactly (§4-A). Content requirements — all of these, written out with the real measured values (no placeholders may survive this task):

1. **Title + problem paragraph**: cold starts on GPU serverless = Hub download + CPU deserialization + kernel compilation, paid on every scale-from-zero; who hits it (bursty traffic, scale-to-zero apps).
2. **What you get**: the measured before/after summary sentence ("N.n× faster cold starts, measured YYYY-MM-DD on GPU-H100"), a sample output video (link the CDN URL from a real run; embed a still if the CDN URL is transient), and the response JSON shape.
3. **Run it**: the template's pinned pip install + `fal auth login` + `fal run app.py` block, plus a note that `fal run app_baseline.py` serves the identical endpoint the slow way.
4. **How it works**: one subsection per lever, each mapped to its named `setup()` stage and its row in the measured table; a subsection on `wan_flashpack.py` explaining (a) it is the verbatim flagship pattern from the FlashPack repo's WAN test, linked, and (b) why `_publish_classes` exists (the two `sys.modules` consumers — this is the FlashPack-on-fal knowledge the docs page will reuse); the pack-locally-then-publish rule with the ~0.1 MB/s number and its date; the readiness-marker honesty mechanism. Link `github.com/fal-ai/flashpack` — no fal concept doc exists yet (SERV-1286 notes the docs-side follow-up).
5. **Deploy**: `fal deploy app.py`, then a `curl` example against the deployed URL with the SMALL_REQUEST payload and a note on the first-boot economics (one boot per account pays download+pack+compile; every later cold start gets the steady state).
6. **Measured cold starts** table: the bench tables (baseline runs, optimized first boot, optimized steady state) with the run date, GPU, and torch/diffusers/flashpack versions, and the sentence "reproduce with `bench.py`" showing the exact commands. State the headline multiplier computed from `spawn -> healthy` medians. Note the runner-state caveat: ephemeral runners expose only PENDING/RUNNING over REST, so stage granularity comes from in-app timers.
7. **Cost & tuning**: keep_alive vs cold-start frequency; why GPU-H100 is pinned (per-GPU-type kernel caches); `PACK_VERSION` bump rule; when NOT to bother (small models, rarely-scaling apps); smoke/CI knobs (`num_frames=17`, `num_inference_steps=8` for fast checks).
8. **Next steps**: 02.2 autoscaling (`../..`-relative link), 02.3 serve-a-20GB-model (slot exists, may not have a folder yet — link the tier README), the FlashPack repo, and `optimize-cold-starts` on fal.ai/docs.

- [ ] **Step 2: Update the tier README row**

In `02-scaling-cold-starts/README.md`, change the 02.1 row to link the folder and reflect reality:

```markdown
| `02.1` | [Cut cold starts by Nx (FlashPack + /data + compiled caches)](cut-cold-starts/) | `needs-deploy` · [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271) findings posted, decision still open | [SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286) |
```

- [ ] **Step 3: Gates**

Run: `.venv/bin/python harness/static_check.py` — Expected: all `ok`, exit 0.
Run: `.venv/bin/python -m pytest 02-scaling-cold-starts/cut-cold-starts/test_app.py -q` — Expected: `4 passed, 3 skipped`.
Grep: `grep -rn "TODO\|TBD\|<Example name>\|<!--" 02-scaling-cold-starts/cut-cold-starts/README.md` — Expected: no output (template comments all replaced).

- [ ] **Step 4: Commit**

```bash
git add 02-scaling-cold-starts/cut-cold-starts/README.md 02-scaling-cold-starts/README.md
git commit -m "docs(02-scaling): cut-cold-starts README with measured before/after (SERV-1286)"
```

---

### Task 9: PR + Linear follow-ups

**Files:** none (process task)

- [ ] **Step 1: Push and open the PR**

Use the `superpowers:finishing-a-development-branch` skill. PR: base `main`, title `feat(02-scaling): cut cold starts by Nx — FlashPack + /data + compiled caches on WAN 2.1 (SERV-1286)`. Body: what/why, the three levers, the measured table, the `wan_flashpack.py` mechanism paragraph (spike-verified), gates run (static, free tests, harness deploy-check both variants, bench), the CI workflow change and its rationale, and open risks. End with the repo's PR footer convention.

- [ ] **Step 2: Comment findings on SERV-1271**

Via Linear MCP (`save_comment` on SERV-1271), summarizing: public wheel 0.4.0 == repo main (except one CI-only commit, compared 2026-08-10); the flagship WAN pattern works on fal via `local_python_modules` + runtime `sys.modules` self-registration (spike + live deploys); the `local_python_modules` limitation discovered (by-value modules get no `sys.modules` entry); pack-write-to-/data remains pathological (~0.1 MB/s, re-observed or carried from July); the decision itself stays Rehan's.

- [ ] **Step 3: Update SERV-1286**

Attach the PR link (Linear auto-links via the magic words; verify with `get_issue`), leave status In Progress (DoD convention: merged ≠ Done; validated deployment done, docs page still a follow-up).

---

## Self-review notes (per writing-plans checklist)

- **Spec coverage**: files/folder ✓ (T1–T3, T7), flagship classes verbatim ✓ (T1), three levers ✓ (T2–T3), spike-before-build ✓ (T4 — moved after app code exists since the spike ships the same module, but before any full-model deploy), meta-tensor check ✓ (T3), atomic publish + sweep ✓ (T3), versioned pack dir ✓ (T3), pins ✓ (global constraints), endpoints + setup-report ✓ (T2–T3), CI both-variants ✓ (harness does this natively; T5 suite), bench fresh+minimal ✓ (T7, reusing harness helpers rather than resurrecting PR #5 code), README §4-A + dated numbers ✓ (T8), tier README ✓ (T8), SERV-1271 comment ✓ (T9), workflow timeouts (discovered during fact-finding, spec's "startup_timeout ≥ 1800 s" line) ✓ (T6).
- **Deviation from spec, deliberate**: spec's spike was "tiny app, no model weights, before any real build" — kept (T4 needs no model download) but sequenced after T2–T3 so it ships the real `wan_flashpack.py`; the fallback path is fully specified in T4 Step 5. Spec's "17–81 4k+1" input contract enforced via pydantic validator + live 422 test. Cross-variant output-equality ledger (01.1 style) deliberately not adopted — GPU/compile nondeterminism; noted in T5.
- **Placeholder scan**: the only intentionally deferred content is Task 8's measured numbers, which Task 7 produces at execution time — the task cannot be executed out of order because the README content requirement names the exact tables/metadata to embed. `<scratchpad>`/`<repo>` in Task 4 are execution-environment paths the executor substitutes.
- **Type consistency**: `get_classes()` keys match between T1/T3/T4; `SetupReport` fields match between T2/T3/T5/T7 (`variant`, `gpu`, `built_caches_this_boot`, `stages[{name, seconds}]`, `total_seconds`); bench run dict = `{spawn_to_healthy_s} ∪ SetupReport` in T7's test and code; `SMALL_REQUEST`/`EXPECTED_VARIANT` defined once (T1) and consumed in T5.
