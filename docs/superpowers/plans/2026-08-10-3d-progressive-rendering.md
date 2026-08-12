# 04.7 · 3D Progressive Rendering (SAM-3D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `04-genmedia-patterns/3d-progressive-rendering/` — a fal serverless example that streams SAM-3D Objects reconstruction progress over SSE (voxels appearing live during both diffusion stages, final GLB/splat by URL), rebuilt on Meta's official release with no model fork.

**Architecture:** A container-based `fal.App` with two endpoints: `/` (plain JSON generate, smoked by the CI harness) and `/stream` (SSE). Streaming drives the official `InferencePipeline`'s generators via their public `generate_iter()` per-step iterator instead of forked callbacks. A bounded queue with drop-oldest semantics bridges the GPU worker thread to the SSE generator; progress frames are disposable, terminal events guaranteed. Spec: `docs/superpowers/specs/2026-08-10-3d-progressive-rendering-design.md`.

**Tech Stack:** fal==1.78.2, Python 3.11, CUDA 12.1 / torch 2.5.1+cu121 (Meta's tested baseline), `facebookresearch/sam-3d-objects` @ `f91db411c50efee93d8db7aeb323885650f6f722`, gated HF weights `facebook/sam-3d-objects` @ `2e73555018d2741ccd486e56c24fac41155a1dc6`, Three.js 0.180.0 (viewer).

## Global Constraints

- Target Python **3.11** (matches CI and fal's cloudpickle requirement for containers).
- `fal==1.78.2` everywhere (harness/requirements.txt is the reference).
- Every install pinned (`==`, git SHA, or digest). No unpinned installs in any file, including README.
- `app.py` module scope may import only: stdlib, `fal`, `fal.container`, `fal.exceptions`, `fal.toolkit`, `fastapi`, `pydantic` (static CI executes app.py with only fal+httpx+pytest installed). numpy/torch/sam3d imports live inside functions/methods.
- Exactly one `fal.App` subclass in `app.py`; explicit `app_auth = "private"`; every **required** Input field carries `Field(examples=[...])` (the harness builds its smoke payload from `examples[0]`).
- `/` must be a plain JSON request/response endpoint (harness smoke); SSE lives at `/stream` exactly (fal_client.stream() appends that path).
- Container app: `App.requirements` stays empty; `requirements.txt` hand-mirrors the Dockerfile pips with a comment naming the Dockerfile authoritative.
- Event payload contract (frozen; used by Tasks 2, 4, 5, 6): see "Event schema" below.
- No performance claims in README that aren't backed by the spike's recorded measurements.
- `spike-sam3d/` (repo root) is **never committed** — tier-dir discovery ignores it, but keep it out of git entirely; delete it in Task 8.
- Commit after each task with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- All example files live in `04-genmedia-patterns/3d-progressive-rendering/` (short kebab-case, no slot prefix).

### Event schema (frozen contract)

Every SSE frame is `data: <compact JSON>\n\n`; heartbeats are the comment line `: keepalive\n\n`. Every stream ends with exactly one terminal event (`complete` or `error`).

```
{"type":"status","stage":"loading|preprocessing|finalizing","progress":0.03,"message":"..."}
{"type":"geometry","step":5,"total_steps":25,"progress":0.14,"voxels":"<b64>","count":8541,"grid_size":64}
{"type":"appearance","step":10,"total_steps":25,"progress":0.68,"voxels":"<b64>","count":8541,"grid_size":64}
{"type":"mesh_preview","vertices":"<b64 f32>","faces":"<b64 u32>","colors":"<b64 u8>","vertex_count":15000,"face_count":29996}
{"type":"complete","progress":1.0,"glb":{"url":"...","content_type":"model/gltf-binary"},"gaussian_splat":{"url":"...","content_type":"application/octet-stream"},"seed":42}
{"type":"error","code":"generation_failed|worker_died","message":"<sanitized, no traceback>"}
```

`voxels` is base64 of N×6 `uint8` rows `[x,y,z,r,g,b]`: x/y/z are **raw grid indices** (0..63 on the fixed 64³ grid — lossless in uint8, stable across frames; client dequantizes `(v + 0.5)/grid_size - 0.5`); r/g/b are 0-255 (geometry-stage frames use 128,128,128). Progress mapping: stage 1 → 0.05–0.50, stage 2 → 0.50–0.90, finalizing → 0.90–1.0.

---

### Task 1: Spike — validate `generate_iter` streaming on a real H100 deploy

The whole design is gated on this. It also warms fal's Dockerfile-content-keyed build cache (the final app uses the *same* Dockerfile file, so CI never pays the cold build).

**Files:**
- Create: `04-genmedia-patterns/3d-progressive-rendering/Dockerfile` (its final home — shared by spike and app)
- Create: `spike-sam3d/main.py` (repo root; NEVER `git add`)
- Create: `spike-sam3d/FINDINGS.md` (filled from run output; content later folded into README/design addendum, file itself not committed)

**Interfaces:**
- Produces: a validated Dockerfile; measured numbers (build ok, setup() seconds, full-gen seconds warm, per-step ss-decode ms, per-step slat-gs-decode ms, VRAM GB); structure facts (`is_mm_dit`, `x_t` type per stage, gaussian `_features_dc` shape vs voxel count N); go/no-go on the no-fork approach.

- [ ] **Step 1: Preconditions — HF access + fal secret**

Request access at https://huggingface.co/facebook/sam-3d-objects (gated: manual approval). Then verify the token can see the repo, and store it as a fal secret:

```bash
python3 -c "
from huggingface_hub import HfApi
import os
info = HfApi(token=os.environ['HF_TOKEN']).model_info('facebook/sam-3d-objects', revision='2e73555018d2741ccd486e56c24fac41155a1dc6')
print('OK', info.sha)
"
```

Expected: `OK 2e73555018d2741ccd486e56c24fac41155a1dc6`. Then:

```bash
fal secrets set HF_TOKEN "$HF_TOKEN"
```

If access is not yet approved (403), STOP — this is a hard external gate; report to Rehan rather than substituting a mirror.

- [ ] **Step 2: Write the Dockerfile**

Create `04-genmedia-patterns/3d-progressive-rendering/Dockerfile`:

```dockerfile
# The app's environment. fal builds this server-side (no local Docker, no
# registry push) and caches by the CONTENT of this file - a digest-pinned
# FROM makes every base change a visible edit. (nvidia/cuda tag resolved
# 2026-08-10.)
#
# The stack tracks Meta's tested baseline for sam-3d-objects (doc/setup.md):
# CUDA 12.1, torch 2.5.1+cu121, kaolin/pytorch3d/flash-attn via the repo's
# own pinned requirement files - installing the repo at a fixed SHA pins the
# whole model stack transitively.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
# Headless rendering for pyrender/pytorch3d texture ops.
ENV PYOPENGL_PLATFORM=osmesa
# H100 (sm90). Add e.g. "8.9" here if you switch machine_type.
ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"
# Index/links the upstream requirement files expect (doc/setup.md).
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# python3.11 must match the Python that runs `fal deploy`: fal ships the app
# as cloudpickled bytecode, which is CPython-minor-specific. curl is required
# by fal's resolver. The GL/osmesa libs back headless rendering.
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        curl ca-certificates git build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libegl1 libglew-dev libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --no-cache-dir pip==25.2

RUN python3.11 -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121

# The model code, pinned by SHA. Its requirements*.txt files pin the rest of
# the stack (spconv, xformers, kaolin, gsplat, pytorch3d, flash-attn, ...).
# Install order per upstream doc/setup.md ('.[p3d]' is a separate step because
# pytorch3d's torch dependency resolution is broken; '[dev]' is lint/test
# tooling we don't need). flash-attn and pytorch3d compile from source here -
# slow once, then cached by fal until this file changes.
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /opt/sam-3d-objects && \
    cd /opt/sam-3d-objects && git checkout f91db411c50efee93d8db7aeb323885650f6f722
RUN cd /opt/sam-3d-objects && \
    python3.11 -m pip install --no-cache-dir '.' && \
    MAX_JOBS=8 python3.11 -m pip install --no-cache-dir '.[p3d]' && \
    python3.11 -m pip install --no-cache-dir '.[inference]' && \
    ./patching/hydra

# fal last, so it resolves against what is already installed.
RUN python3.11 -m pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.
```

- [ ] **Step 3: Write the spike app**

Create `spike-sam3d/main.py`:

```python
"""Spike for 04.7 (SERV-1319): validate no-fork streaming via generate_iter.

Answers, on a real H100:
  1. Does the container build and the gated HF download work?
  2. What does generate_iter yield per stage (tensor vs dict)? is_mm_dit?
  3. Per-step cost: ss_decoder decode (stage 1), slat_decoder_gs decode (stage 2)?
  4. Does a generate_iter-driven run reproduce pipeline.run()'s voxel count (same seed)?
  5. Warm full-generation latency (for the harness 120s smoke budget) + VRAM.
  6. Gaussian _features_dc shape vs voxel count N (voxel-color mapping).

NEVER git add this directory.
"""
import json
import time
from pathlib import Path

import fal
from fal.container import ContainerImage
from fal.toolkit import FAL_PERSISTENT_DIR
from pydantic import BaseModel, Field

DOCKERFILE = (
    Path(__file__).parent.parent
    / "04-genmedia-patterns" / "3d-progressive-rendering" / "Dockerfile"
)
WEIGHTS_DIR = FAL_PERSISTENT_DIR / "sam-3d-objects" / "2e73555018d2741ccd486e56c24fac41155a1dc6"
HF_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
SAMPLE_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"


class ProbeInput(BaseModel):
    image_url: str = Field(default=SAMPLE_IMAGE, examples=[SAMPLE_IMAGE])
    seed: int = Field(default=42)
    stage2_probe_steps: int = Field(
        default=3, description="How many stage-2 steps to time gs-decode on."
    )


class SpikeApp(
    fal.App,
    name="sam3d-spike",
    keep_alive=300,
    kind="container",
    image=ContainerImage.from_dockerfile(str(DOCKERFILE), builder="depot"),
    max_concurrency=1,
):
    machine_type = "GPU-H100"
    app_auth = "private"
    request_timeout = 900

    def setup(self):
        import torch
        from huggingface_hub import snapshot_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        t0 = time.time()
        snapshot_download(
            "facebook/sam-3d-objects",
            revision=HF_REVISION,
            local_dir=str(WEIGHTS_DIR),
        )
        self.download_s = time.time() - t0

        # Mirrors the official notebook/inference.py Inference wrapper
        # (not pip-installed - it lives outside the sam3d_objects package).
        t0 = time.time()
        config_path = WEIGHTS_DIR / "checkpoints" / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        self.pipeline = instantiate(config)
        self.load_s = time.time() - t0
        torch.cuda.reset_peak_memory_stats()

    @fal.endpoint("/probe")
    def probe(self, input: ProbeInput) -> dict:
        import numpy as np
        import torch
        from fal.toolkit import Image as FalImage
        from sam3d_objects.pipeline.inference_utils import (
            downsample_sparse_structure,
            prune_sparse_structure,
        )
        from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

        pipeline = self.pipeline
        findings: dict = {
            "download_s": round(self.download_s, 1),
            "pipeline_load_s": round(self.load_s, 1),
            "is_mm_dit": pipeline.is_mm_dit(),
        }

        pil = FalImage(url=input.image_url).to_pil().convert("RGB")
        rgb = np.array(pil)
        mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, mask[..., None]], axis=-1)

        # --- Q5: reference full run (warm timing + reference voxel count) ---
        t0 = time.time()
        ref = pipeline.run(
            rgba, None, input.seed,
            stage1_only=False, with_mesh_postprocess=False,
            with_texture_baking=False, with_layout_postprocess=False,
            use_vertex_color=True,
        )
        findings["full_run_warm_s"] = round(time.time() - t0, 1)
        findings["ref_voxel_count"] = int(ref["coords"].shape[0])
        findings["ref_output_keys"] = sorted(ref.keys())
        findings["glb_is_none"] = ref.get("glb") is None

        # --- Q6: gaussian structure (voxel-color mapping) ---
        gs = ref["gs"]
        for attr in ("_features_dc", "_xyz", "_opacity"):
            v = getattr(gs, attr, None)
            if v is not None:
                findings[f"gs{attr}_shape"] = list(v.shape)

        # --- Q2-Q4: generate_iter-driven stage 1, timed per-step decode ---
        ss_input = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor)
        torch.manual_seed(input.seed)
        ss_gen = pipeline.models["ss_generator"]
        ss_dec = pipeline.models["ss_decoder"]
        ss_gen.no_shortcut = True
        ss_gen.reverse_fn.strength = pipeline.ss_cfg_strength
        ss_gen.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
        image = ss_input["image"]
        bs = image.shape[0]
        if pipeline.is_mm_dit():
            latent_shape = {
                k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_gen.reverse_fn.backbone.latent_mapping.items()
            }
        else:
            latent_shape = (bs,) + (4096, 8)
        cargs, ckw = pipeline.get_condition_input(
            pipeline.condition_embedders["ss_condition_embedder"],
            ss_input, pipeline.ss_condition_input_mapping,
        )
        decode_times, step_times, xt_types = [], [], set()
        coords = None
        t_prev = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.shape_model_dtype):
            for t, x_t, _ in ss_gen.generate_iter(latent_shape, image.device, *cargs, **ckw):
                step_times.append(time.time() - t_prev)
                xt_types.add(type(x_t).__name__)
                shape_latent = x_t["shape"] if isinstance(x_t, dict) else x_t
                td = time.time()
                ss = ss_dec(
                    shape_latent.permute(0, 2, 1).contiguous().view(bs, 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                decode_times.append(time.time() - td)
                t_prev = time.time()
        if pipeline.downsample_ss_dist > 0:
            coords = prune_sparse_structure(
                coords, max_neighbor_axes_dist=pipeline.downsample_ss_dist
            )
        coords, _ = downsample_sparse_structure(coords)
        findings["stage1_xt_types"] = sorted(xt_types)
        findings["stage1_steps"] = len(step_times)
        findings["stage1_step_ms_avg"] = round(1000 * sum(step_times) / len(step_times))
        findings["ss_decode_ms_avg"] = round(1000 * sum(decode_times) / len(decode_times))
        findings["iter_voxel_count"] = int(coords.shape[0])
        findings["coords_max"] = int(coords[:, 1:].max())  # expect <= 63
        findings["parity_ok"] = findings["iter_voxel_count"] == findings["ref_voxel_count"]

        # --- Q3 stage 2: time slat gs-decode on a few generate_iter steps ---
        slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        slat_gen = pipeline.models["slat_generator"]
        slat_gen.no_shortcut = True
        slat_gen.reverse_fn.strength = pipeline.slat_cfg_strength
        latent_shape2 = (bs,) + (coords.shape[0], 8)
        cargs2, ckw2 = pipeline.get_condition_input(
            pipeline.condition_embedders["slat_condition_embedder"],
            slat_input, pipeline.slat_condition_input_mapping,
        )
        cargs2 = cargs2 + (coords.cpu().numpy(),)
        gs_decode_times, slat_step_times = [], []
        n_probe = input.stage2_probe_steps
        t_prev = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
            for i, (t, x_t, _) in enumerate(
                slat_gen.generate_iter(latent_shape2, image.device, *cargs2, **ckw2)
            ):
                slat_step_times.append(time.time() - t_prev)
                if i < n_probe:
                    td = time.time()
                    st = sp.SparseTensor(coords=coords, feats=x_t[0]).to(image.device)
                    st = st * pipeline.slat_std.to(image.device) + pipeline.slat_mean.to(image.device)
                    g = pipeline.models["slat_decoder_gs"](st)[0]
                    fdc = getattr(g, "_features_dc", None)
                    findings.setdefault(
                        "probe_fdc_shape", None if fdc is None else list(fdc.shape)
                    )
                    gs_decode_times.append(time.time() - td)
                t_prev = time.time()
        findings["stage2_steps"] = len(slat_step_times)
        findings["stage2_step_ms_avg"] = round(
            1000 * sum(slat_step_times) / len(slat_step_times)
        )
        findings["slat_gs_decode_ms_avg"] = round(
            1000 * sum(gs_decode_times) / max(1, len(gs_decode_times))
        )

        findings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        print("[spike] FINDINGS:", json.dumps(findings, indent=2))
        return findings
```

- [ ] **Step 4: Run the spike**

```bash
fal run spike-sam3d/main.py::SpikeApp
```

Wait for the URL, then in a second terminal (auth headers come from `fal.sdk`):

```bash
python3 -c "
import httpx, json
from fal.sdk import get_credentials
url = input('app url: ').rstrip('/')
r = httpx.post(url + '/probe', json={}, headers=get_credentials().to_headers(), timeout=900)
r.raise_for_status()
print(json.dumps(r.json(), indent=2))
"
```

Expected: JSON findings with `parity_ok: true`, `coords_max <= 63`, all timings populated. The first run pays the container build (watch `fal run` logs; the build is cached against this Dockerfile content afterwards).

- [ ] **Step 5: Record findings and apply the decision matrix**

Write all returned numbers into `spike-sam3d/FINDINGS.md`. Then:

| Observation | Decision |
|---|---|
| `parity_ok` false or `generate_iter` unusable | STOP. Fall back per spec: pin the fork `rehan-remade/sam-3d-objects` to its current SHA, move it to an org account, adapt Task 4 to the fork's callback API. Ask Rehan before proceeding. |
| `full_run_warm_s` > 100 | In Task 8, add `HARNESS_REQUEST_TIMEOUT: "300"` to the deploy-check job env in `.github/workflows/check.yml`. |
| `slat_gs_decode_ms_avg` > 1500 | In Task 4, set `colors_every` default to a value putting per-request color-decode overhead under ~15% of stage-2 time (e.g. 10), and note it in README Cost & tuning. |
| `probe_fdc_shape[0]` == N or a clean multiple of N | Task 4 voxel-color mapping works as written. Otherwise use the gray-fallback branch (already in Task 4's code). |
| `coords_max` > 63 | Grid is bigger than 64³ — update `GRID_SIZE` in Task 2 and the schema note to the observed grid size. |
| setup() (download+load) > 500s on the warm run | Raise app `request_timeout`/document; if the *health* window (600s) is at risk, report to Rehan before merging. |

- [ ] **Step 6: Commit (Dockerfile only)**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/Dockerfile
git commit -m "feat(04-genmedia-patterns): SAM-3D container env, spike-validated (SERV-1319)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Encoders + SSE helpers (pure, static-CI-testable) — TDD

**Files:**
- Create: `04-genmedia-patterns/3d-progressive-rendering/app.py` (module-level helpers; the App class arrives in Task 3)
- Create: `04-genmedia-patterns/3d-progressive-rendering/test_app.py` (unit tests)

**Interfaces:**
- Produces (all module-level in `app.py`):
  - `GRID_SIZE: int = 64`, `HEARTBEAT_SECONDS: float = 15.0`
  - `encode_voxels(coords, colors=None) -> str` — coords: (N,3) int array-like 0..GRID_SIZE-1; colors: (N,3) float array-like in 0..1 or None (gray 128 fill). Returns base64 of N×6 uint8. **The color contract: float 0-1 in, scaled to 0-255 exactly once, here.**
  - `decode_voxels(payload: str) -> list[tuple[int,int,int,int,int,int]]` — pure stdlib; the reference client-side decoder.
  - `sse_format(event: dict) -> str`, `SSE_HEARTBEAT: str`
  - `put_progress(q, event) -> None` — non-blocking; on Full drops the OLDEST queued frame to make room (progress frames are disposable).
  - `put_terminal(q, event) -> None` — guaranteed delivery; evicts progress frames if needed; `event` may be `None` (end-of-stream sentinel).
  - `stream_frames(q, worker_done, heartbeat_s=HEARTBEAT_SECONDS)` — generator of SSE strings; heartbeat comments on silence; returns after the `None` sentinel; emits a `worker_died` error then returns if the worker set `worker_done` without a sentinel.
  - `class GenerationCancelled(Exception)`

- [ ] **Step 1: Write the failing tests**

Create `test_app.py`:

```python
"""Tests for 04.7 - 3D progressive rendering.

Two kinds, both run by CI:
  * unit tests over the pure encoding/SSE helpers - free, no deploy, run on
    every PR by the static job (numpy-dependent ones skip when numpy is
    absent);
  * live tests that consume the real SSE stream on an ephemeral deploy
    (harness/check.py) - added in Task 5.
"""
import base64
import importlib.util
import json
import queue
import threading
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "sam3d_stream_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


# --- decode_voxels: pure stdlib, runs everywhere -------------------------

def test_decode_voxels_roundtrips_handbuilt_payload():
    raw = bytes([1, 2, 3, 10, 20, 30, 63, 0, 63, 255, 128, 0])
    rows = app_module.decode_voxels(base64.b64encode(raw).decode())
    assert rows == [(1, 2, 3, 10, 20, 30), (63, 0, 63, 255, 128, 0)]


def test_decode_voxels_rejects_ragged_payload():
    with pytest.raises(ValueError):
        app_module.decode_voxels(base64.b64encode(b"\x01\x02\x03\x04").decode())


# --- encode_voxels: numpy path (skips on static CI without numpy) --------

def test_encode_decode_roundtrip_with_colors():
    np = pytest.importorskip("numpy")
    coords = np.array([[0, 0, 0], [63, 63, 63]], dtype=np.int64)
    colors = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]], dtype=np.float32)
    rows = app_module.decode_voxels(app_module.encode_voxels(coords, colors))
    assert rows[0] == (0, 0, 0, 0, 127, 255)
    assert rows[1] == (63, 63, 63, 255, 0, 63)


def test_encode_voxels_gray_fill_without_colors():
    np = pytest.importorskip("numpy")
    rows = app_module.decode_voxels(
        app_module.encode_voxels(np.array([[5, 6, 7]]), None)
    )
    assert rows == [(5, 6, 7, 128, 128, 128)]


def test_encode_voxels_clips_out_of_range_colors():
    np = pytest.importorskip("numpy")
    rows = app_module.decode_voxels(
        app_module.encode_voxels(np.array([[1, 1, 1]]), np.array([[-0.5, 2.0, 1.0]]))
    )
    assert rows == [(1, 1, 1, 0, 255, 255)]


# --- SSE framing -----------------------------------------------------------

def test_sse_format_is_one_data_frame():
    frame = app_module.sse_format({"type": "status", "progress": 0.5})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):]) == {"type": "status", "progress": 0.5}


def test_heartbeat_is_a_comment_line():
    assert app_module.SSE_HEARTBEAT.startswith(":")
    assert app_module.SSE_HEARTBEAT.endswith("\n\n")


# --- queue semantics: progress disposable, terminal guaranteed -------------

def test_put_progress_drops_oldest_when_full():
    q = queue.Queue(maxsize=2)
    for i in range(4):
        app_module.put_progress(q, {"type": "geometry", "step": i})
    steps = [q.get_nowait()["step"] for _ in range(2)]
    assert steps == [2, 3]  # oldest frames were dropped, newest survive


def test_put_terminal_evicts_progress_to_guarantee_delivery():
    q = queue.Queue(maxsize=1)
    app_module.put_progress(q, {"type": "geometry", "step": 0})
    app_module.put_terminal(q, {"type": "complete"})
    assert q.get_nowait() == {"type": "complete"}


# --- stream_frames: heartbeat, sentinel, dead-worker taxonomy --------------

def test_stream_frames_yields_events_then_stops_at_sentinel():
    q = queue.Queue()
    done = threading.Event()
    q.put({"type": "status"})
    q.put({"type": "complete"})
    q.put(None)
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    assert [json.loads(f[6:]) for f in frames] == [
        {"type": "status"}, {"type": "complete"},
    ]


def test_stream_frames_emits_heartbeat_on_silence():
    q = queue.Queue()
    done = threading.Event()

    def finish_later():
        import time
        time.sleep(0.15)
        q.put(None)

    threading.Thread(target=finish_later).start()
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    assert app_module.SSE_HEARTBEAT in frames


def test_stream_frames_reports_dead_worker_without_sentinel():
    q = queue.Queue()
    done = threading.Event()
    done.set()  # worker exited without ever enqueueing the sentinel
    frames = list(app_module.stream_frames(q, done, heartbeat_s=0.05))
    last = json.loads(frames[-1][6:])
    assert last["type"] == "error"
    assert last["code"] == "worker_died"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd 04-genmedia-patterns/3d-progressive-rendering && python -m pytest test_app.py -q
```

Expected: collection error (`app.py` does not exist yet).

- [ ] **Step 3: Write the helpers**

Create `app.py`:

```python
"""3D progressive rendering: stream SAM-3D reconstruction as it happens.

Catalog slot 04.7 (SERV-1319). One H100 app, two endpoints: `/` returns the
final GLB + Gaussian splat as URLs; `/stream` emits SSE while the two
diffusion stages run - voxels appear live, then the final assets arrive by
URL. The pattern to steal: model progress -> bounded queue -> SSE generator,
where progress frames are DISPOSABLE (drop-oldest) and the single terminal
event is GUARANTEED.
"""
import base64
import json
import queue
import secrets
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import FAL_PERSISTENT_DIR, File
from fal.toolkit import Image as FalImage
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Stage-1 voxel coords are indices in a fixed 64-cube grid, so uint8 is
# LOSSLESS - no per-frame renormalization, no jitter between frames.
GRID_SIZE = 64
HEARTBEAT_SECONDS = 15.0  # well under typical 60-100s proxy idle timeouts
SSE_HEARTBEAT = ": keepalive\n\n"


class GenerationCancelled(Exception):
    """Raised inside the worker when the client has disconnected."""


def encode_voxels(coords, colors=None) -> str:
    """Pack voxels as base64 of N x [x,y,z,r,g,b] uint8 rows (6 bytes/voxel).

    coords: (N,3) integer grid indices in [0, GRID_SIZE).
    colors: (N,3) floats in [0,1], or None for gray. The float->uint8 scaling
    happens HERE and only here - one encoder, one color contract.
    """
    import numpy as np

    coords = np.asarray(coords)
    packed = np.empty((len(coords), 6), dtype=np.uint8)
    packed[:, :3] = coords.astype(np.uint8)
    if colors is None:
        packed[:, 3:] = 128
    else:
        colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
        packed[:, 3:] = np.round(colors * 255).astype(np.uint8)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_voxels(payload: str) -> list[tuple[int, int, int, int, int, int]]:
    """Reference decoder (pure stdlib) - what a client does with `voxels`."""
    raw = base64.b64decode(payload)
    if len(raw) % 6:
        raise ValueError(f"voxel payload length {len(raw)} is not a multiple of 6")
    return [tuple(raw[i : i + 6]) for i in range(0, len(raw), 6)]


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def put_progress(q: queue.Queue, event: dict) -> None:
    """Enqueue a disposable progress frame. A slow client must never stall
    the GPU: when the queue is full, drop the OLDEST frame and move on."""
    while True:
        try:
            q.put_nowait(event)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def put_terminal(q: queue.Queue, event: dict | None) -> None:
    """Enqueue a terminal event (or the None end-of-stream sentinel) with
    guaranteed delivery, evicting progress frames if the queue is full."""
    while True:
        try:
            q.put(event, timeout=0.1)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def stream_frames(q: queue.Queue, worker_done: threading.Event,
                  heartbeat_s: float = HEARTBEAT_SECONDS):
    """Drain the queue into SSE frames. Heartbeats are SSE comments (clients'
    event parsers never see them). Ends at the None sentinel; if the worker
    died without one, emit a terminal error so 'finished' is never ambiguous
    with 'connection died'."""
    while True:
        try:
            event = q.get(timeout=heartbeat_s)
        except queue.Empty:
            if worker_done.is_set() and q.empty():
                yield sse_format({
                    "type": "error", "code": "worker_died",
                    "message": "generation worker exited unexpectedly",
                })
                return
            yield SSE_HEARTBEAT
            continue
        if event is None:
            return
        yield sse_format(event)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd 04-genmedia-patterns/3d-progressive-rendering && python -m pytest test_app.py -q
```

Expected: all pass (numpy tests skip if numpy absent locally — `pip install numpy==1.26.4` to run them).

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/app.py 04-genmedia-patterns/3d-progressive-rendering/test_app.py
git commit -m "feat(04-genmedia-patterns): voxel encoding + SSE queue semantics for 3d-progressive-rendering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: App class, `/` endpoint, requirements.txt

**Files:**
- Modify: `04-genmedia-patterns/3d-progressive-rendering/app.py` (append below the Task 2 helpers)
- Create: `04-genmedia-patterns/3d-progressive-rendering/requirements.txt`

**Interfaces:**
- Consumes: Task 1's Dockerfile (same directory), Task 2's helpers.
- Produces: `class GenerateInput(BaseModel)` (`image_url: str` required w/ examples, `mask_url: str | None`, `seed: int | None`, `stage1_steps: int|None`, `stage2_steps: int|None`); `class StreamInput(GenerateInput)` (+`geometry_every: int = 1`, `colors_every: int = 5`); `class AssetOutput(BaseModel)` (`glb: File`, `gaussian_splat: File`, `seed: int`); `class SAM3DStreamApp(fal.App)` with `self.pipeline` set in `setup()`, methods `_load_rgba(input) -> np.ndarray` and `_export_assets(outputs, request) -> tuple[File, File]`; endpoint `POST /` → `AssetOutput`.

- [ ] **Step 1: Append the models and App to app.py**

```python
# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

SAMPLE_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"
HF_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
WEIGHTS_DIR = FAL_PERSISTENT_DIR / "sam-3d-objects" / HF_REVISION
DOCKERFILE = Path(__file__).parent / "Dockerfile"


class GenerateInput(BaseModel):
    image_url: str = Field(
        description="Image containing the object to reconstruct in 3D.",
        examples=[SAMPLE_IMAGE],
    )
    mask_url: str | None = Field(
        default=None,
        description=(
            "Optional binary mask isolating the object (white = object). "
            "Without it the whole image is reconstructed as one object - "
            "fine for object-centric photos, poor for cluttered scenes."
        ),
    )
    seed: int | None = Field(
        default=None, description="Random seed; omit for a random one."
    )
    stage1_steps: int | None = Field(
        default=None, ge=4, le=50,
        description="Geometry diffusion steps (default: model config, 25).",
    )
    stage2_steps: int | None = Field(
        default=None, ge=4, le=50,
        description="Appearance diffusion steps (default: model config, 25).",
    )


class StreamInput(GenerateInput):
    geometry_every: int = Field(
        default=1, ge=1, le=10,
        description="Emit a geometry frame every N steps (1 = smoothest).",
    )
    colors_every: int = Field(
        default=5, ge=1, le=25,
        description=(
            "Emit a colored appearance frame every N steps. Each frame costs "
            "a Gaussian decode, so this is a latency knob, not just polish."
        ),
    )


class AssetOutput(BaseModel):
    glb: File = Field(description="Final textured mesh (vertex colors), GLB.")
    gaussian_splat: File = Field(description="Gaussian splat, PLY.")
    seed: int = Field(description="Seed actually used.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SAM3DStreamApp(
    fal.App,
    name="sam-3d-progressive",
    keep_alive=60,  # raise for interactive use; see README "Cost & tuning"
    kind="container",
    image=ContainerImage.from_dockerfile(str(DOCKERFILE), builder="depot"),
    max_concurrency=1,
    min_concurrency=0,
):
    # SAM-3D's own prerequisite is >=32 GB VRAM, which rules out most
    # smaller tiers; H100 keeps both diffusion stages interactive.
    machine_type = "GPU-H100"
    app_auth = "private"
    request_timeout = 600

    def setup(self):
        import torch
        from huggingface_hub import snapshot_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Gated checkpoint: requires one-time access approval on
        # huggingface.co/facebook/sam-3d-objects and an HF_TOKEN fal secret.
        # Pinned revision + /data cache: repeat cold starts skip the download.
        snapshot_download(
            "facebook/sam-3d-objects",
            revision=HF_REVISION,
            local_dir=str(WEIGHTS_DIR),
        )

        # Mirrors the official notebook/inference.py `Inference` wrapper
        # (that helper is not part of the pip-installed package).
        # FAL_PERSISTENT_DIR is not a real pathlib.Path - OmegaConf.load
        # rejects it with "Unexpected file type" (spike finding).
        config_path = Path(str(WEIGHTS_DIR)) / "checkpoints" / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        self.pipeline = instantiate(config)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[sam-3d-progressive] pipeline ready")

    # -- shared input/output plumbing --------------------------------------

    def _load_rgba(self, input: GenerateInput):
        """Fetch image (+ optional mask) and merge to the RGBA the pipeline
        expects. Runs BEFORE any StreamingResponse is returned, so bad inputs
        are honest HTTP 422s, never mid-stream error events."""
        import numpy as np

        try:
            rgb = np.array(FalImage(url=input.image_url).to_pil().convert("RGB"))
        except Exception as exc:
            raise FieldException(
                "image_url", "Could not fetch or decode the image."
            ) from exc

        if input.mask_url:
            try:
                mask_pil = FalImage(url=input.mask_url).to_pil().convert("L")
            except Exception as exc:
                raise FieldException(
                    "mask_url", "Could not fetch or decode the mask."
                ) from exc
            mask = (np.array(mask_pil.resize((rgb.shape[1], rgb.shape[0]))) > 127)
            mask = mask.astype(np.uint8) * 255
        else:
            mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        return np.concatenate([rgb, mask[..., None]], axis=-1)

    def _export_assets(self, outputs: dict, request: Request):
        """Final assets travel by URL, never inline - the terminal event and
        the JSON endpoint share this rule."""
        with TemporaryDirectory() as tmp:
            splat_path = str(Path(tmp) / "splat.ply")
            outputs["gs"].save_ply(splat_path)
            splat = File.from_path(splat_path, multipart=True, request=request)

            glb_path = str(Path(tmp) / "model.glb")
            outputs["glb"].export(glb_path)
            glb = File.from_path(
                glb_path, content_type="model/gltf-binary",
                multipart=True, request=request,
            )
        return glb, splat

    # -- endpoints ----------------------------------------------------------

    @fal.endpoint("/")
    def generate(self, input: GenerateInput, request: Request) -> AssetOutput:
        """Plain request/response: the whole pipeline, assets by URL.
        Use /stream to watch it happen."""
        rgba = self._load_rgba(input)
        seed = input.seed if input.seed is not None else secrets.randbits(31)
        outputs = self.pipeline.run(
            rgba, None, seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=input.stage1_steps,
            stage2_inference_steps=input.stage2_steps,
        )
        glb, splat = self._export_assets(outputs, request)
        return AssetOutput(glb=glb, gaussian_splat=splat, seed=seed)
```

- [ ] **Step 2: Create requirements.txt**

```
# Local-venv mirror. For a container example the AUTHORITATIVE environment is
# the Dockerfile in this folder (the model stack is pinned transitively by
# sam-3d-objects @ f91db411, installed there). These pins cover what you need
# locally: the unit tests and the pinned client/runtime versions.
fal==1.78.2
numpy==1.26.4
torch==2.5.1
torchvision==0.20.1
huggingface_hub==0.34.4
```

- [ ] **Step 3: Static checks + unit tests still pass**

```bash
python -m pytest 04-genmedia-patterns/3d-progressive-rendering/test_app.py -q
python harness/static_check.py --example 04-genmedia-patterns/3d-progressive-rendering
```

Expected: tests pass; static check passes (one App subclass, explicit auth, examples on required fields, README missing is the only acceptable failure at this point — if static_check hard-fails on the missing README, create a one-line placeholder `README.md` now and note that Task 7 replaces it).

- [ ] **Step 4: Live check of `/` on a real deploy**

```bash
fal run 04-genmedia-patterns/3d-progressive-rendering/app.py::SAM3DStreamApp
```

Then POST `{}`-defaults smoke from a second terminal:

```bash
python3 -c "
import httpx, json
from fal.sdk import get_credentials
url = input('app url: ').rstrip('/')
r = httpx.post(url + '/', json={'image_url': 'https://storage.googleapis.com/falserverless/UNO/crystal_ball.png', 'seed': 42}, headers=get_credentials().to_headers(), timeout=600)
r.raise_for_status()
out = r.json(); print(json.dumps(out, indent=2))
import urllib.request
assert urllib.request.urlopen(out['glb']['url']).read(4) == b'glTF', 'bad GLB magic'
print('GLB OK')
"
```

Expected: JSON with `glb.url`, `gaussian_splat.url`, `seed: 42`; `GLB OK`. Record the wall-clock of the request — this is the number the Task 1 decision matrix uses for the harness timeout call.

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/app.py 04-genmedia-patterns/3d-progressive-rendering/requirements.txt
git commit -m "feat(04-genmedia-patterns): SAM-3D generate endpoint on official pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Streaming driver + `/stream` endpoint

**Files:**
- Modify: `04-genmedia-patterns/3d-progressive-rendering/app.py`

**Interfaces:**
- Consumes: Task 2 helpers (`encode_voxels`, `put_progress`, `put_terminal`, `stream_frames`, `GenerationCancelled`, `GRID_SIZE`), Task 3 (`StreamInput`, `_load_rgba`, `_export_assets`, `self.pipeline`).
- Produces: module-level `stream_generate(pipeline, rgba, seed, params, emit, cancelled) -> dict` (mirrors `InferencePipeline.run()` but drives both generators via `generate_iter`; calls `emit(event_dict)` for progress; checks `cancelled: threading.Event` each step; returns the pipeline `outputs` dict); endpoint `POST /stream` → `text/event-stream`.

- [ ] **Step 1: Add `stream_generate` to app.py** (module level, after the helpers — it is the teaching centerpiece; keep it one screenful per stage)

```python
# ---------------------------------------------------------------------------
# The streaming driver. This is the whole idea of the example:
# the official pipeline's generators expose generate_iter(), a public
# per-solver-step iterator (generate() is literally "loop generate_iter,
# keep the last value") - so progressive output needs NO model fork, just
# this driver that mirrors InferencePipeline.run() stage by stage.
# ---------------------------------------------------------------------------


def stream_generate(pipeline, rgba, seed: int, params: "StreamInput",
                    emit, cancelled: threading.Event) -> dict:
    import numpy as np
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )

    SH_C0 = 0.28209479177387814  # SH degree-0 constant: rgb = 0.5 + C0 * f_dc

    def check_cancelled():
        if cancelled.is_set():
            raise GenerationCancelled()

    models = pipeline.models
    with pipeline.device:
        emit({"type": "status", "stage": "preprocessing", "progress": 0.03,
              "message": "Preparing pipeline inputs"})
        # InferencePipelinePointMap conditions stage 1 on a MoGe depth
        # pointmap - without it the condition embedder KeyErrors on
        # 'pointmap' (spike finding). Stage 2 does not take it.
        pointmap_dict = pipeline.compute_pointmap(rgba, None)
        ss_input = pipeline.preprocess_image(
            rgba, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        torch.manual_seed(seed)

        # ---- stage 1: geometry (sparse structure) -------------------------
        ss_gen, ss_dec = models["ss_generator"], models["ss_decoder"]
        ss_gen.no_shortcut = True
        ss_gen.reverse_fn.strength = pipeline.ss_cfg_strength
        ss_gen.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
        prev_steps = ss_gen.inference_steps
        if params.stage1_steps:
            ss_gen.inference_steps = params.stage1_steps
        total1 = ss_gen.inference_steps

        image = ss_input["image"]
        bs = image.shape[0]
        if pipeline.is_mm_dit():
            latent_shape = {
                k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_gen.reverse_fn.backbone.latent_mapping.items()
            }
        else:
            latent_shape = (bs,) + (4096, 8)
        cargs, ckw = pipeline.get_condition_input(
            pipeline.condition_embedders["ss_condition_embedder"],
            ss_input, pipeline.ss_condition_input_mapping,
        )

        coords = None
        step = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.shape_model_dtype):
            for _t, x_t, _ in ss_gen.generate_iter(
                latent_shape, image.device, *cargs, **ckw
            ):
                step += 1
                check_cancelled()
                if step % params.geometry_every and step != total1:
                    continue
                shape_latent = x_t["shape"] if isinstance(x_t, dict) else x_t
                ss = ss_dec(
                    shape_latent.permute(0, 2, 1).contiguous().view(bs, 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                grid_xyz = coords[:, 1:].cpu().numpy()
                emit({
                    "type": "geometry", "step": step, "total_steps": total1,
                    "progress": round(0.05 + 0.45 * step / total1, 3),
                    "voxels": encode_voxels(grid_xyz),
                    "count": int(len(grid_xyz)), "grid_size": GRID_SIZE,
                })
        ss_gen.inference_steps = prev_steps

        # Final stage-1 coords, pruned/downsampled exactly like run() does.
        if pipeline.downsample_ss_dist > 0:
            coords = prune_sparse_structure(
                coords, max_neighbor_axes_dist=pipeline.downsample_ss_dist
            )
        coords, _ = downsample_sparse_structure(coords)

        # ---- stage 2: appearance (structured latent) ----------------------
        slat_gen = models["slat_generator"]
        slat_gen.no_shortcut = True
        slat_gen.reverse_fn.strength = pipeline.slat_cfg_strength
        prev_steps2 = slat_gen.inference_steps
        if params.stage2_steps:
            slat_gen.inference_steps = params.stage2_steps
        total2 = slat_gen.inference_steps

        latent_shape2 = (bs,) + (coords.shape[0], 8)
        cargs2, ckw2 = pipeline.get_condition_input(
            pipeline.condition_embedders["slat_condition_embedder"],
            slat_input, pipeline.slat_condition_input_mapping,
        )
        cargs2 = cargs2 + (coords.cpu().numpy(),)
        grid_xyz = coords[:, 1:].cpu().numpy()
        n_vox = coords.shape[0]

        def voxel_colors(x_t) -> "np.ndarray | None":
            """Decode an intermediate slat to per-voxel float [0,1] colors via
            the Gaussian decoder's degree-0 SH. Costs one decode - which is
            why colors_every is a knob. Returns None if the gaussian layout
            doesn't map cleanly onto voxels (then frames stay gray)."""
            st = sp.SparseTensor(coords=coords, feats=x_t[0]).to(image.device)
            st = st * pipeline.slat_std.to(image.device) + pipeline.slat_mean.to(image.device)
            g = models["slat_decoder_gs"](st)[0]
            fdc = getattr(g, "_features_dc", None)
            if fdc is None:
                return None
            fdc = fdc.reshape(fdc.shape[0], -1)[:, :3]
            if fdc.shape[0] == n_vox:
                per_voxel = fdc
            elif fdc.shape[0] % n_vox == 0:
                per_voxel = fdc.view(n_vox, -1, 3).mean(dim=1)
            else:
                return None
            return (0.5 + SH_C0 * per_voxel).clamp(0, 1).float().cpu().numpy()

        slat = None
        step = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
            for _t, x_t, _ in slat_gen.generate_iter(
                latent_shape2, image.device, *cargs2, **ckw2
            ):
                step += 1
                check_cancelled()
                slat_latent = x_t
                if step % params.colors_every and step != total2:
                    continue
                colors = voxel_colors(x_t)
                emit({
                    "type": "appearance", "step": step, "total_steps": total2,
                    "progress": round(0.50 + 0.40 * step / total2, 3),
                    "voxels": encode_voxels(grid_xyz, colors),
                    "count": int(n_vox), "grid_size": GRID_SIZE,
                })
            slat = sp.SparseTensor(coords=coords, feats=slat_latent[0]).to(image.device)
            slat = slat * pipeline.slat_std.to(image.device) + pipeline.slat_mean.to(image.device)
        slat_gen.inference_steps = prev_steps2

        # ---- finalize: decode + GLB, exactly like run() --------------------
        emit({"type": "status", "stage": "finalizing", "progress": 0.90,
              "message": "Decoding mesh + gaussians, exporting GLB"})
        check_cancelled()
        outputs = pipeline.decode_slat(slat, pipeline.decode_formats)
        outputs = pipeline.postprocess_slat_output(
            outputs, with_mesh_postprocess=False, with_texture_baking=False,
            use_vertex_color=True,
        )

        glb = outputs.get("glb")
        if glb is not None and hasattr(glb, "vertices"):
            verts = np.asarray(glb.vertices, dtype=np.float32)
            faces = np.asarray(glb.faces, dtype=np.uint32)
            colors_u8 = None
            visual = getattr(glb, "visual", None)
            if visual is not None and getattr(visual, "vertex_colors", None) is not None:
                colors_u8 = np.asarray(visual.vertex_colors, dtype=np.uint8)[:, :3]
            # A frame, not an asset: cap the inline preview.
            if len(verts) <= 200_000:
                emit({
                    "type": "mesh_preview",
                    "vertices": base64.b64encode(verts.tobytes()).decode("ascii"),
                    "faces": base64.b64encode(faces.tobytes()).decode("ascii"),
                    "colors": None if colors_u8 is None
                    else base64.b64encode(colors_u8.tobytes()).decode("ascii"),
                    "vertex_count": int(len(verts)),
                    "face_count": int(len(faces)),
                })
        return outputs
```

- [ ] **Step 2: Add the `/stream` endpoint to `SAM3DStreamApp`**

```python
    @fal.endpoint("/stream")
    def stream(self, input: StreamInput, request: Request):
        """SSE progressive rendering. Every stream ends in exactly one
        terminal event: complete (asset URLs) or error (sanitized)."""
        from concurrent.futures import ThreadPoolExecutor

        rgba = self._load_rgba(input)  # bad inputs 422 BEFORE we stream
        seed = input.seed if input.seed is not None else secrets.randbits(31)

        q: queue.Queue = queue.Queue(maxsize=8)
        cancelled = threading.Event()
        worker_done = threading.Event()

        def worker():
            try:
                put_progress(q, {"type": "status", "stage": "loading",
                                 "progress": 0.01, "message": "Starting"})
                outputs = stream_generate(
                    self.pipeline, rgba, seed, input,
                    emit=lambda ev: put_progress(q, ev),
                    cancelled=cancelled,
                )
                glb, splat = self._export_assets(outputs, request)
                put_terminal(q, {
                    "type": "complete", "progress": 1.0,
                    "glb": {"url": glb.url, "content_type": "model/gltf-binary"},
                    "gaussian_splat": {"url": splat.url,
                                       "content_type": "application/octet-stream"},
                    "seed": seed,
                })
            except GenerationCancelled:
                pass  # client left; nobody is reading
            except Exception:
                import traceback
                traceback.print_exc()  # full detail server-side ONLY
                put_terminal(q, {"type": "error", "code": "generation_failed",
                                 "message": "3D generation failed; check app logs."})
            finally:
                worker_done.set()
                put_terminal(q, None)  # end-of-stream sentinel, guaranteed

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(worker)

        def frames():
            try:
                yield from stream_frames(q, worker_done)
            finally:
                # Runs on normal completion AND client disconnect: signal the
                # worker (it checks between steps) and release the thread.
                cancelled.set()
                executor.shutdown(wait=False, cancel_futures=True)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={
                # Optional fal-gateway nicety: plain synchronous calls to
                # /stream get the LAST event (complete/error) as the body.
                "x-fal-sse-last-event-as-body": "1",
                "Cache-Control": "no-cache",
            },
        )
```

- [ ] **Step 3: Unit tests still green; manual stream check on a live deploy**

```bash
python -m pytest 04-genmedia-patterns/3d-progressive-rendering/test_app.py -q
fal run 04-genmedia-patterns/3d-progressive-rendering/app.py::SAM3DStreamApp
```

Second terminal — consume the stream and print an event digest:

```bash
python3 -c "
import httpx, json
from fal.sdk import get_credentials
url = input('app url: ').rstrip('/')
seen = []
with httpx.stream('POST', url + '/stream',
                  json={'image_url': 'https://storage.googleapis.com/falserverless/UNO/crystal_ball.png', 'seed': 42},
                  headers=get_credentials().to_headers(), timeout=600) as r:
    r.raise_for_status()
    assert r.headers['content-type'].startswith('text/event-stream')
    for line in r.iter_lines():
        if line.startswith('data: '):
            ev = json.loads(line[6:])
            seen.append(ev['type'])
            print(ev['type'], ev.get('step'), ev.get('progress'), ev.get('count'))
print('sequence:', seen)
"
```

Expected: `status` → repeated `geometry` (rising step/count) → repeated `appearance` → `mesh_preview` (unless vertex cap hit) → `complete` with two URLs. Confirm appearance frames are NOT all-gray (spot-check one `voxels` payload's color bytes ≠ 128) — if they are, apply the Task 1 decision-matrix row for `probe_fdc_shape`.

- [ ] **Step 4: Disconnect test (the bug the old example had)**

Start a stream as above but Ctrl-C the client mid-generation. In the `fal run` logs, confirm the worker stops (no further `geometry`/`appearance` decode logging within a few steps) and the app serves a subsequent request normally. Expected: next request succeeds; no wedged worker.

- [ ] **Step 5: Commit**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/app.py
git commit -m "feat(04-genmedia-patterns): SSE streaming via official generate_iter - no model fork

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Live SSE tests (sets the repo's streaming-test precedent)

**Files:**
- Modify: `04-genmedia-patterns/3d-progressive-rendering/test_app.py` (append)

**Interfaces:**
- Consumes: `app_client` fixture (harness/pytest_plugin.py — provides `.url`, `.run()`), deployed endpoints from Tasks 3–4, `decode_voxels` from Task 2.

- [ ] **Step 1: Append live tests**

```python
# --- live tests (harness only: app_client skips without HARNESS_APP_URL) ---

import httpx  # noqa: E402  (module-level: harness deps include httpx)

SAMPLE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"


def _fal_headers():
    from fal.sdk import get_credentials
    return get_credentials().to_headers()


def test_generate_returns_valid_assets(app_client):
    out = app_client.run("/", {"image_url": SAMPLE, "seed": 42})
    assert out["seed"] == 42
    glb_head = b""
    with httpx.stream("GET", out["glb"]["url"], timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            glb_head += chunk
            if len(glb_head) >= 4:
                break
    assert glb_head[:4] == b"glTF"
    assert httpx.head(out["gaussian_splat"]["url"], timeout=60).status_code < 400


def test_stream_event_contract(app_client):
    """The streaming contract: monotonic progress, geometry strictly before
    appearance, decodable voxel frames, exactly one terminal event carrying
    asset URLs."""
    events = []
    with httpx.stream(
        "POST", app_client.url + "/stream",
        json={"image_url": SAMPLE, "seed": 42, "geometry_every": 5, "colors_every": 10},
        headers=_fal_headers(), timeout=600,
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    terminals = [t for t in types if t in ("complete", "error")]
    assert terminals == ["complete"], f"stream ended badly: {types}"
    assert types[-1] == "complete"

    geometry = [e for e in events if e["type"] == "geometry"]
    appearance = [e for e in events if e["type"] == "appearance"]
    assert geometry and appearance
    last_geometry = max(i for i, t in enumerate(types) if t == "geometry")
    assert last_geometry < types.index("appearance")

    progresses = [e["progress"] for e in events if "progress" in e]
    assert progresses == sorted(progresses), "progress went backwards"

    frame = geometry[-1]
    rows = app_module.decode_voxels(frame["voxels"])
    assert len(rows) == frame["count"] > 0
    assert all(0 <= r[i] < frame["grid_size"] for r in rows[:100] for i in range(3))

    final = events[-1]
    with httpx.stream("GET", final["glb"]["url"], timeout=60) as r:
        r.raise_for_status()
        assert next(r.iter_bytes(4))[:4] == b"glTF"


def test_stream_rejects_bad_image_before_streaming(app_client):
    resp = httpx.post(
        app_client.url + "/stream",
        json={"image_url": "https://storage.googleapis.com/falserverless/does-not-exist.png"},
        headers=_fal_headers(), timeout=120,
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "image_url"]
```

- [ ] **Step 2: Verify unit tests still pass and live tests skip locally**

```bash
python -m pytest 04-genmedia-patterns/3d-progressive-rendering/test_app.py -q
```

Expected: unit tests pass, live tests skip (`no live app under test`).

- [ ] **Step 3: Full harness check (real ephemeral deploy)**

```bash
python harness/check.py --example 04-genmedia-patterns/3d-progressive-rendering
```

Expected: deploy → health → `/` smoke → all tests pass → teardown confirmed. Watch total test wall-clock: the two full generations (smoke + `/` test + stream test) must fit the 600s pytest cap — if not, drop `test_generate_returns_valid_assets` (the harness smoke already covers `/`) and keep the stream test.

- [ ] **Step 4: Commit**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/test_app.py
git commit -m "test(04-genmedia-patterns): SSE contract tests for 3d-progressive-rendering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: viewer.html — single-file progressive voxel viewer

**Files:**
- Create: `04-genmedia-patterns/3d-progressive-rendering/viewer.html`

**Interfaces:**
- Consumes: the frozen event schema; a deployed app URL + FAL key pasted by the user.

- [ ] **Step 1: Write viewer.html**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAM-3D progressive viewer</title>
<style>
  body { margin: 0; font: 14px system-ui; background: #111; color: #eee; }
  #bar { display: flex; gap: 8px; padding: 10px; flex-wrap: wrap; align-items: center; }
  #bar input { padding: 6px; background: #222; color: #eee; border: 1px solid #444; }
  #url { width: 320px; } #key { width: 200px; } #img { width: 280px; }
  #status { padding: 0 10px 8px; color: #9ad; }
  canvas { display: block; }
  a { color: #9ad; }
</style>
</head>
<body>
<div id="bar">
  <input id="url" placeholder="deployed app URL (…fal.run/you/sam-3d-progressive)">
  <input id="key" placeholder="FAL key id:secret" type="password">
  <input id="img" value="https://storage.googleapis.com/falserverless/UNO/crystal_ball.png">
  <button id="go">Generate</button>
  <span id="dl"></span>
</div>
<div id="status">Paste your endpoint + key (local dev only: the key is visible
to this page - never ship this pattern to real users), then Generate.</div>
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(1.2, 0.9, 1.2);
const renderer = new THREE.WebGLRenderer({ antialias: true });
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(2, 3, 1); scene.add(dir);

function resize() {
  const h = innerHeight - document.getElementById("bar").offsetHeight
    - document.getElementById("status").offsetHeight;
  renderer.setSize(innerWidth, h);
  camera.aspect = innerWidth / h; camera.updateProjectionMatrix();
}
addEventListener("resize", resize); resize();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

let voxelMesh = null, meshPreview = null;
const MAX_INSTANCES = 70000;
const proto = new THREE.BoxGeometry(1, 1, 1);
const mat = new THREE.MeshLambertMaterial();
const dummy = new THREE.Object3D();

// Real instancing: ONE InstancedMesh, per-instance matrix + color. (The old
// docs page nested <mesh> children inside instancedMesh - that renders
// nothing instanced at all.)
function showVoxels(rows, gridSize) {
  if (meshPreview) { scene.remove(meshPreview); meshPreview = null; }
  if (!voxelMesh) {
    voxelMesh = new THREE.InstancedMesh(proto, mat, MAX_INSTANCES);
    voxelMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(voxelMesh);
  }
  const s = 1 / gridSize;
  const n = Math.min(rows.length, MAX_INSTANCES);
  dummy.scale.set(s * 0.9, s * 0.9, s * 0.9);
  for (let i = 0; i < n; i++) {
    const [x, y, z, r, g, b] = rows[i];
    // dequantize grid index -> centered world coords
    dummy.position.set((x + 0.5) / gridSize - 0.5, (y + 0.5) / gridSize - 0.5,
                       (z + 0.5) / gridSize - 0.5);
    dummy.updateMatrix();
    voxelMesh.setMatrixAt(i, dummy.matrix);
    voxelMesh.setColorAt(i, new THREE.Color(r / 255, g / 255, b / 255));
  }
  voxelMesh.count = n;
  voxelMesh.instanceMatrix.needsUpdate = true;
  if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
}

function showMeshPreview(ev) {
  const verts = new Float32Array(b64bytes(ev.vertices).buffer);
  const faces = new Uint32Array(b64bytes(ev.faces).buffer);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  let m;
  if (ev.colors) {
    const cols = b64bytes(ev.colors);
    const colF = new Float32Array(cols.length);
    for (let i = 0; i < cols.length; i++) colF[i] = cols[i] / 255;
    geo.setAttribute("color", new THREE.BufferAttribute(colF, 3));
    m = new THREE.MeshLambertMaterial({ vertexColors: true });
  } else {
    m = new THREE.MeshLambertMaterial({ color: 0x8899aa });
  }
  geo.computeVertexNormals();
  if (voxelMesh) { scene.remove(voxelMesh); voxelMesh = null; }
  meshPreview = new THREE.Mesh(geo, m);
  // model space is roughly [-0.5, 0.5] voxel coords; mesh preview shares it
  scene.add(meshPreview);
}

function b64bytes(b64) {
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeVoxels(b64) {
  const bytes = b64bytes(b64), rows = [];
  for (let i = 0; i + 5 < bytes.length; i += 6) rows.push(bytes.subarray(i, i + 6));
  return rows;
}

const status = (m) => document.getElementById("status").textContent = m;

document.getElementById("go").onclick = async () => {
  const url = document.getElementById("url").value.replace(/\/+$/, "");
  const key = document.getElementById("key").value.trim();
  document.getElementById("dl").innerHTML = "";
  status("connecting…");

  // EventSource can't POST or set headers, so: fetch + manual SSE parse.
  // Events arrive in arbitrary chunk boundaries - buffer and split on the
  // blank line, exactly what the old docs snippet forgot.
  const resp = await fetch(url + "/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Key " + key },
    body: JSON.stringify({ image_url: document.getElementById("img").value }),
  });
  if (!resp.ok) { status("HTTP " + resp.status + ": " + await resp.text()); return; }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      if (!chunk.startsWith("data: ")) continue; // comment heartbeats etc.
      const ev = JSON.parse(chunk.slice(6));
      if (ev.type === "geometry" || ev.type === "appearance") {
        showVoxels(decodeVoxels(ev.voxels), ev.grid_size);
        status(`${ev.type} ${ev.step}/${ev.total_steps} - ${ev.count} voxels`);
      } else if (ev.type === "mesh_preview") {
        showMeshPreview(ev); status("mesh preview");
      } else if (ev.type === "status") {
        status(ev.message);
      } else if (ev.type === "complete") {
        status("done - seed " + ev.seed);
        document.getElementById("dl").innerHTML =
          `<a href="${ev.glb.url}">GLB</a> · <a href="${ev.gaussian_splat.url}">splat</a>`;
      } else if (ev.type === "error") {
        status("error: " + ev.message);
      }
    }
  }
};
</script>
</body>
</html>
```

- [ ] **Step 2: Verify against a live deploy**

```bash
fal run 04-genmedia-patterns/3d-progressive-rendering/app.py::SAM3DStreamApp
python3 -m http.server -d 04-genmedia-patterns/3d-progressive-rendering 8000
```

Open http://localhost:8000/viewer.html, paste the app URL and a FAL key, Generate. Expected: gray voxels accumulate during geometry, colorize during appearance, swap to the mesh preview, then GLB/splat download links appear. Confirm the browser console shows no parse errors across the whole stream.

- [ ] **Step 3: Commit**

```bash
git add 04-genmedia-patterns/3d-progressive-rendering/viewer.html
git commit -m "feat(04-genmedia-patterns): single-file Three.js progressive viewer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: README (§4-A) + tier README row

**Files:**
- Create: `04-genmedia-patterns/3d-progressive-rendering/README.md` (replace any Task 3 placeholder)
- Modify: `04-genmedia-patterns/README.md` (04.7 row: link the folder)

- [ ] **Step 1: Write the example README**

Follow `templates/example/README.md` section order exactly. Content requirements (write full prose, not bullets-as-prose; substitute the spike's measured numbers where marked):

````markdown
# 3D progressive rendering: watch SAM-3D build a model live

3D generation takes long enough that a spinner is a product problem: the user
stares at nothing for the better part of a minute. SAM-3D Objects happens to
be built from two diffusion stages whose intermediate states are *drawable* -
a voxel cloud that sharpens, then colorizes. This example streams those
states over SSE while the model runs, so your UI renders progress instead of
promising it. fal's hosted `fal-ai/sam-3/3d-objects` gives you the same model
request/response; deploy this when you want the progressive experience.

## What you get

<!-- GIF/video: voxels appearing + colorizing + final mesh, viewer.html capture -->
Progressive voxel frames during both diffusion stages, a vertex-colored mesh
preview, then the final assets by URL:

```json
{"type":"geometry","step":12,"total_steps":25,"progress":0.27,"voxels":"<base64>","count":8541,"grid_size":64}
{"type":"appearance","step":20,"total_steps":25,"progress":0.82,"voxels":"<base64>","count":8541,"grid_size":64}
{"type":"complete","progress":1.0,"glb":{"url":"https://…/model.glb"},"gaussian_splat":{"url":"https://…/splat.ply"},"seed":42}
```

## Run it

One-time: request access to the gated weights at
huggingface.co/facebook/sam-3d-objects (SAM License), then store a token:

```bash
fal secrets set HF_TOKEN "hf_…"
```

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py::SAM3DStreamApp
```

Then either open the bundled viewer (`python -m http.server 8000`, open
viewer.html, paste the printed URL + a FAL key - local dev only), or consume
the stream in ~10 lines of Python:

```python
import httpx, json
from fal.sdk import get_credentials

with httpx.stream(
    "POST", "<your app URL>/stream",
    json={"image_url": "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"},
    headers=get_credentials().to_headers(), timeout=600,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            print(event["type"], event.get("progress"))
```

## How it works

Three ideas, in the order they appear in app.py:

1. **No fork.** The official pipeline's generators expose `generate_iter()`,
   a public per-solver-step iterator (`generate()` is just "loop it, keep the
   last value"). `stream_generate()` mirrors `InferencePipeline.run()` stage
   by stage and emits a frame per step. Stage-1 frames decode the
   intermediate latent to a 64³ occupancy grid; stage-2 frames decode
   per-voxel colors through the Gaussian head every `colors_every` steps.
2. **Progress frames are disposable; the terminal event is guaranteed.** The
   GPU worker and the SSE generator meet at a bounded queue: `put_progress`
   drops the oldest frame when the client is slow (the GPU never waits on the
   network), `put_terminal` always delivers. Every stream ends in exactly one
   `complete` or `error` - and disconnects cancel generation between steps
   instead of leaking a wedged worker. Heartbeats are SSE *comments*, sent
   every 15s so proxies don't kill quiet phases.
3. **Small frames inline, big assets by URL.** A voxel frame is 6 bytes per
   voxel (raw 64³ grid indices fit uint8 losslessly - `(v+0.5)/64-0.5` to
   dequantize; colors are scaled float 0-1 → byte exactly once, in one
   encoder). The GLB and splat never travel inline - the `complete` event
   carries fal CDN URLs (see Working with Files).

Concept docs: [Streaming Endpoints](https://docs.fal.ai/serverless/development/streaming) ·
[handling inputs & outputs](https://docs.fal.ai/serverless/development/handle-inputs-and-outputs) ·
[custom containers](https://docs.fal.ai/serverless/development/customize-environment-containers)

## Deploy

```bash
fal deploy app.py
```

Once registered, the fal clients stream it natively (they append `/stream`):

```python
import fal_client

for event in fal_client.stream("<username>/sam-3d-progressive",
                               arguments={"image_url": "https://…/photo.png"}):
    print(event["type"], event.get("progress"))
```

```ts
import { fal } from "@fal-ai/client";

const stream = await fal.stream("<username>/sam-3d-progressive", {
  input: { image_url: "https://…/photo.png" },
});
for await (const event of stream) console.log(event.type, event.progress);
```

## Cost & tuning

<!-- Fill from spike FINDINGS.md - no unmeasured claims:
     cold start (build cached) ≈ Xs setup; warm generation ≈ Ys on H100.
     Worked example at current H100 $/s pricing: one generation + 60s
     keep_alive tail. Knobs table: keep_alive (interactive sessions vs batch),
     geometry_every/colors_every (frame rate vs per-step decode cost — each
     appearance frame costs one Gaussian decode ≈ Zms), stage1/2_steps
     (quality vs latency), max_concurrency=1 rationale (one pipeline, one
     GPU). machine_type: SAM-3D's stated prerequisite is ≥32 GB VRAM (spike
     measured peak W GB), which is why this stays on H100. -->

## Next steps

- [Stream progressive results with SSE](../../03-streaming-realtime/) - the
  minimal version of this pattern (03.1)
- [LOD processor for 3D assets](../../05-media-processing/) - post-process
  the GLB this example produces (05.2)
- Hosted [fal-ai/sam-3/3d-objects](https://fal.ai/models/fal-ai/sam-3/3d-objects)
  for request/response without deploying; TRELLIS.2-4B / Hunyuan3D 3.x if you
  want maximum asset fidelity over progressive UX

---
SAM-3D Objects weights and code are Meta's, under the
[SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE);
this example downloads the gated checkpoint from the official
`facebook/sam-3d-objects` repository.
````

- [ ] **Step 2: Link the folder in the tier README**

In `04-genmedia-patterns/README.md`, change the 04.7 row's Example cell to:

```
| `04.7` | 3D progressive rendering (SAM-3D) — [`3d-progressive-rendering/`](3d-progressive-rendering/) | `needs-deploy` | [SERV-1319](https://linear.app/features-and-labels/issue/SERV-1319) |
```

- [ ] **Step 3: Verify + commit**

```bash
python harness/static_check.py --example 04-genmedia-patterns/3d-progressive-rendering
git add 04-genmedia-patterns/3d-progressive-rendering/README.md 04-genmedia-patterns/README.md
git commit -m "docs(04-genmedia-patterns): 3d-progressive-rendering README + tier row

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final validation + cleanup

**Files:**
- Modify (only if the Task 1 decision matrix triggered it): `.github/workflows/check.yml`
- Delete: `spike-sam3d/` (was never committed)

- [ ] **Step 1: Apply the timeout decision (conditional)**

If Task 3 Step 4 measured the warm `/` generation above 100s, add to the deploy-check job's `env:` in `.github/workflows/check.yml`:

```yaml
      HARNESS_REQUEST_TIMEOUT: "300"
```

(with a comment: `# 04.7 runs a full two-stage 3D generation in its smoke`). If ≤100s, skip this step.

- [ ] **Step 2: Full check suite, from scratch**

```bash
python -m pytest -q
python harness/check.py --example 04-genmedia-patterns/3d-progressive-rendering
```

Expected: repo-wide unit tests green (live tests skip); harness run fully green including teardown confirmation. The container build must be a cache hit (the Dockerfile is byte-identical to the spike's) — if the harness log shows a full rebuild, the Dockerfile changed since Task 1: re-verify it still builds, and note the cache is now warm for CI.

- [ ] **Step 3: Fold spike numbers into README, delete the spike**

Fill every `<!-- Fill from spike -->` marker in README.md with FINDINGS.md numbers, then:

```bash
rm -rf spike-sam3d/
git status  # confirm nothing spike-related is tracked
git add 04-genmedia-patterns/3d-progressive-rendering/README.md
git commit -m "docs(04-genmedia-patterns): measured numbers for 3d-progressive-rendering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Hand off**

Use superpowers:finishing-a-development-branch: PR against `main` titled
`feat(04-genmedia-patterns): 3D progressive rendering (SAM-3D) (SERV-1319)`, body linking the spec, the ticket, and the harness-green run. Follow-ups that are explicitly **not** in this PR (note them in the PR body): the fal-docs page refresh (companion PR per SERV-1266 IA conventions), the org-owned hosted endpoint decision, and the companion frontend repo home.

## Plan Self-Review (done at write time)

- **Spec coverage:** spike gate → Task 1; two-endpoint split → Tasks 3–4; queue semantics/cancellation/heartbeat/error taxonomy → Tasks 2, 4; encoding contract → Task 2; container conventions + cache pre-warm → Task 1; SSE test precedent → Task 5; viewer → Task 6; §4-A README + license attribution + tier row → Task 7; CI timeout decision + cleanup + PR → Task 8. Docs-repo work deferred (spec says post-merge).
- **Known-risk notes for executors:** `x_t` structure for mm-dit checkpoints and gaussian `_features_dc` layout are spike-verified facts — Task 4's code handles tensor/dict and clean/unclean color mappings, but trust the spike output over this plan if they disagree; `downsample_sparse_structure` before stage 2 is required (slat latent shape depends on the downsampled coords). The `x-fal-sse-last-event-as-body` header is optional hardening, not the pattern.
- **Type consistency:** event field names (`type/step/total_steps/progress/voxels/count/grid_size`), helper names (`encode_voxels/decode_voxels/put_progress/put_terminal/stream_frames`), and class names (`SAM3DStreamApp`, `GenerateInput`, `StreamInput`, `AssetOutput`) are identical across Tasks 2–6.
