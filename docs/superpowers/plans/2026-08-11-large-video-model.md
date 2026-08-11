# Example 02.3 — Large Video Model with Preview Streaming: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `02-scaling-cold-starts/large-video-model/` — LTX-2.3 22B distilled on one GPU-H200 with parallel weight loading, GPU-sizing reasoning, and intermediate-step tiny-VAE previews streamed over SSE — CI deploy-check green.

**Architecture:** One `fal.App` with a JSON `/` endpoint (harness-smoked) and a `/stream` SSE sibling. A `callback_on_step_end` x0-extrapolates mid-schedule latents; a single-worker decode thread runs the vendored TAEHV tiny VAE and packs half-res mp4 data-URI events; the final step decodes with the full VAE, muxes audio, uploads to CDN. `loader.py` carries the parallel-download + page-cache-pre-read recipe (the SERV-1303 artifact).

**Tech Stack:** fal 1.78.2, diffusers 0.39.0 (`LTX2Pipeline`), torch 2.12.1, PyAV (`encode_video`), vendored `taehv.py` (MIT, madebyollin), httpx-sse-style raw SSE in tests.

**Spec:** `docs/superpowers/specs/2026-08-11-large-video-model-design.md` (approved 2026-08-11).

## Global Constraints

- Python 3.11 (CI); official Lightricks packages are OFF LIMITS (need ≥3.12) — diffusers path only.
- `App.requirements` authoritative, every spec `==`-pinned; `requirements.txt` an exact mirror (CI-enforced; no `git+` deps — vendor instead).
- `app_auth = "private"`; explicit `machine_type` with a justification comment.
- Every required `Input` field carries `Field(examples=[...])` — examples[0] is both the README sample and the CI smoke payload; it must complete warm in < 120 s.
- README follows `templates/example/README.md` §4-A section order exactly.
- No performance claims except ONE stamped measured table (02.2 precedent), values from the validated deploy.
- Never cite internal registry code anywhere; public sources only.
- Model/weights (pinned): `diffusers/LTX-2.3-Distilled-Diffusers` @ `432e0d3c2d1769aaa4d295f9243f7062bf6b47ee` (95 GB, ungated). Tiny VAE: madebyollin/taehv branch `2026_03_11_taeltx23_wide` @ `32ac0146b11007cda5a57b60a3b35653361fb8a4` (`taehv.py` + `taeltx2_3_wide.pth`, 60 MB) and `taeltx2_3.pth` (24 MB, same content on main @ `e589fddc076e77f5ba8cd6baabe4ba3260b261cd`).
- Distilled call contract (from the model card + `diffusers/pipelines/ltx2/utils.py`): `sigmas=DISTILLED_SIGMA_VALUES, num_inference_steps=8, guidance_scale=1.0`, LTX-2.3 behavior via `use_cross_timestep=True`.
- Callback contract (diffusers 0.39.0 `pipeline_ltx2.py:1409-1421`): `callback_on_step_end(pipe, i, t, callback_kwargs)` → MUST return a dict; `callback_kwargs["latents"]` is video latents, post-step, packed `[B, S, D]`, at `pipe.scheduler.sigmas[i+1]`; audio latents are NOT reachable.
- Mid-step decode recipe: `LTX2Pipeline._unpack_latents(lat, F_lat, H_lat, W_lat, 1, 1)` → `[B,128,F,H,W]`; `LTX2Pipeline._denormalize_latents(lat, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)`; permute NCTHW→NTCHW; `TAEHV.decode_video()` → NTCHW RGB [0,1], first 7 frames trimmed.
- Latent geometry: `H_lat = height//32`, `W_lat = width//32`, `F_lat = (frames-1)//8 + 1`; height/width divisible by 32; frames on the 8k+1 grid.

---

### Task 1: Spike (GATE — no implementation until this passes)

**Files:**
- Create: `<scratchpad>/spike/spike_app.py` (throwaway — NOT committed to the repo)
- Create: `<scratchpad>/spike/spike_stream_client.py` (throwaway)
- Modify: `docs/superpowers/plans/2026-08-11-large-video-model.md` (append results table)

**Interfaces:**
- Produces: go/no-go decision; pinned values for later tasks: `TAEHV_VARIANT` ("wide" or "standard"), smoke shape (`resolution`, `num_frames`), measured setup/generation timings, peak VRAM, resolved dependency pins, confirmed distilled kwargs.

- [ ] **Step 1: Auth preflight**

Run: `fal auth whoami` (or confirm `FAL_KEY` is exported). If auth is broken (expired refresh token — this happened 2026-08-05), STOP and ask Rehan for a key before burning time.

- [ ] **Step 2: Write the spike app**

`<scratchpad>/spike/spike_app.py` — a compact fal app answering all five gates. Complete file:

```python
"""SERV-1303 spike: LTX-2.3 distilled on H200 + taehv mid-step previews. THROWAWAY."""
import base64, io, json, os, queue, subprocess, tempfile, time
from pathlib import Path

import fal
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
MODEL_REV = "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"
MODEL_DIR = "/data/models/ltx-2.3-distilled-diffusers"
TAEHV_RAW = "https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4"
TAE_WEIGHTS = {
    "standard": f"{TAEHV_RAW}/taeltx2_3.pth",
    "wide": f"{TAEHV_RAW}/taeltx2_3_wide.pth",
}


class In(BaseModel):
    prompt: str = Field(examples=["a red fox trotting through fresh snow at golden hour"])
    height: int = 512
    width: int = 768
    num_frames: int = 57
    variant: str = "wide"  # taehv variant to preview with


class Out(BaseModel):
    timings: dict
    peak_vram_gb: float
    event_sizes_kb: list


class SpikeApp(fal.App, keep_alive=600):
    machine_type = "GPU-H200"
    app_auth = "private"
    startup_timeout = 1800
    requirements = [
        "fal==1.78.2", "torch==2.12.1", "diffusers==0.39.0", "transformers==5.3.0",
        "accelerate==1.14.0", "av==16.0.1", "hf-transfer==0.1.9",
    ]

    def setup(self):
        import torch, urllib.request
        from huggingface_hub import snapshot_download
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "1000000000000")
        os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
        self.t = {}
        t0 = time.time()
        try:
            snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, local_files_only=True)
        except Exception:
            snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, max_workers=32)
        self.t["download_s"] = time.time() - t0
        t0 = time.time()
        subprocess.run(f"find {MODEL_DIR} -type f | xargs -P 32 -I {{}} cat {{}} > /dev/null",
                       shell=True, check=True)
        self.t["page_cache_s"] = time.time() - t0
        tae_dir = Path("/data/models/taehv"); tae_dir.mkdir(parents=True, exist_ok=True)
        for name, url in TAE_WEIGHTS.items():
            dest = tae_dir / Path(url).name
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
        # vendored taehv.py is shipped next to this file via local_python_modules
        from taehv import TAEHV
        from diffusers import LTX2Pipeline
        t0 = time.time()
        self.pipe = LTX2Pipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
        self.t["pipe_load_s"] = time.time() - t0
        self.taes = {
            name: TAEHV(str(tae_dir / Path(url).name)).to("cuda", torch.bfloat16)
            for name, url in TAE_WEIGHTS.items()
        }
        torch.cuda.reset_peak_memory_stats()

    def _decode_preview(self, tae, packed, h, w, f):
        import torch
        from diffusers import LTX2Pipeline
        with torch.inference_mode():
            lat = LTX2Pipeline._unpack_latents(packed, (f - 1) // 8 + 1, h // 32, w // 32, 1, 1)
            lat = LTX2Pipeline._denormalize_latents(
                lat, self.pipe.vae.latents_mean, self.pipe.vae.latents_std,
                self.pipe.vae.config.scaling_factor)
            rgb = tae.decode_video(lat.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                                   parallel=True, show_progress_bar=False)
            return (rgb[0].float().clamp(0, 1) * 255).to(torch.uint8).cpu()  # [T,3,h,w]

    @fal.endpoint("/")
    def run(self, r: In) -> Out:
        import numpy as np, torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video
        sizes, sheets, state = [], [], {"prev": None, "prev_sigma": None}
        tae = self.taes[r.variant]

        def cb(pipe, i, t, kw):
            lat = kw["latents"]
            sigma = float(pipe.scheduler.sigmas[i + 1])
            t0 = time.time()
            if state["prev"] is not None and sigma < state["prev_sigma"]:
                x0 = lat - sigma * (lat - state["prev"]) / (sigma - state["prev_sigma"])
                for tag, src in (("raw", lat), ("x0", x0)):
                    frames = self._decode_preview(tae, src, r.height, r.width, r.num_frames)
                    mid = frames[len(frames) // 2].permute(1, 2, 0).numpy()
                    sheets.append((i, tag, mid))
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        encode_video(list(frames.permute(0, 2, 3, 1).numpy()), 24.0, f.name)
                        sizes.append(round(os.path.getsize(f.name) / 1024))
            self.t[f"preview_step{i}_s"] = round(time.time() - t0, 2)
            state["prev"], state["prev_sigma"] = lat.clone(), sigma
            return {}

        t0 = time.time()
        video, audio = self.pipe(
            prompt=r.prompt, height=r.height, width=r.width, num_frames=r.num_frames,
            sigmas=DISTILLED_SIGMA_VALUES, num_inference_steps=8, guidance_scale=1.0,
            use_cross_timestep=True, output_type="np", return_dict=False,
            callback_on_step_end=cb,
        )
        self.t["generate_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        out = "/data/spike_final.mp4"
        encode_video(video[0], 24.0, out, audio=torch.tensor(audio[0]).float().cpu()
                     if not torch.is_tensor(audio[0]) else audio[0].float().cpu(),
                     audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate)
        self.t["final_encode_s"] = round(time.time() - t0, 1)
        # contact sheet: mid-frame of every (step, raw/x0) preview, saved for eyeballing
        from PIL import Image
        for i, tag, arr in sheets:
            Image.fromarray(arr).save(f"/data/spike_preview_step{i}_{tag}_{r.variant}.png")
        import torch as _t
        return Out(timings=self.t, peak_vram_gb=round(_t.cuda.max_memory_allocated() / 2**30, 1),
                   event_sizes_kb=sizes)

    @fal.endpoint("/stream")
    def stream(self, r: In) -> StreamingResponse:
        # gate 5: does fal.run hold a long SSE stream with MB events?
        def gen():
            blob = base64.b64encode(os.urandom(700_000)).decode()  # ~0.9 MB event
            for i in range(10):
                time.sleep(20)  # 10 events over ~200s — longer than any proxy idle guess
                yield f'data: {{"i": {i}, "blob": "{blob}"}}\n\n'
            yield 'data: {"type": "done"}\n\n'
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"x-fal-sse-last-event-as-body": "1"})
```

- [ ] **Step 3: Fetch the vendorable taehv.py next to the spike app**

```bash
cd <scratchpad>/spike
curl -fsSL https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4/taehv.py -o taehv.py
grep -n "WideMemBlock" taehv.py   # must hit — wide support lives on this branch
```

Add `local_python_modules = ["taehv"]` to `SpikeApp` if not already present.

- [ ] **Step 4: Write the SSE client probe**

`<scratchpad>/spike/spike_stream_client.py`:

```python
"""Gate 5 driver: consume the spike /stream through the real fal.run gateway."""
import json, os, sys, time
import httpx

url = sys.argv[1].rstrip("/") + "/stream"
headers = {"Authorization": f"Key {os.environ['FAL_KEY']}"}
t0 = time.time()
with httpx.stream("POST", url, json={"prompt": "x"}, headers=headers, timeout=600) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            print(f"+{time.time()-t0:6.1f}s event {ev.get('i', ev.get('type'))} "
                  f"({len(line)//1024} KB)")
print("STREAM COMPLETE — gateway held the connection")
```

- [ ] **Step 5: Run the spike**

```bash
cd <scratchpad>/spike && fal run spike_app.py
# in a second terminal, against the printed URL:
python spike_stream_client.py https://fal.run/<user>/<app>   # gate 5
# then POST / twice (cold + warm) with variant=wide and variant=standard:
curl -sX POST <url>/ -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox trotting through fresh snow at golden hour", "variant": "wide"}' | jq .
```

Download the `/data/spike_preview_step*_{raw,x0}_{wide,standard}.png` sheets (add a tiny `/file` endpoint or re-run with `fal run --debug` shell access) and eyeball: from which step is the x0 preview "watchable"?

- [ ] **Step 6: Record results + decide**

Append to this plan file a `## Spike results (YYYY-MM-DD)` section filling exactly:

| Gate | Measurement | Pass? |
|---|---|---|
| 1. Pipeline + callback on 0.39.0 | loads / callback fires / kwargs accepted | |
| 2. Preview legibility | first watchable step (x0, wide) vs (x0, standard) | |
| 3. VRAM + timings | peak GB; cold download_s / page_cache_s / pipe_load_s; warm generate_s @ 768×512×57f | |
| 4. Preview cost | preview_step*_s vs per-step budget; event KB | |
| 5. Gateway SSE | 10×0.9MB events over ~200 s arrive incrementally | |

Decisions to record: `TAEHV_VARIANT`, smoke shape (must keep warm `/` invoke < 100 s), whether `HARNESS_STARTUP_TIMEOUT` needs raising, resolved pins (`pip freeze` the spike env for `huggingface_hub`, `av`, `numpy`, `pillow` exact versions). Apply pre-agreed fallbacks from the spec if a gate fails (latent2rgb → B200 → stop and re-spec on WAN 2.2). Tear down the spike app (`fal apps delete` or let ephemeral expire) — verify no runner is left billing.

- [ ] **Step 7: Commit the plan update**

```bash
git add docs/superpowers/plans/2026-08-11-large-video-model.md
git commit -m "docs(02.3): spike results recorded — gate decision (SERV-1303)"
```

---

### Task 2: Scaffold folder + loader.py

**Files:**
- Create: `02-scaling-cold-starts/large-video-model/loader.py`
- Create: `02-scaling-cold-starts/large-video-model/requirements.txt` (placeholder mirror; final pins in Task 4)
- Create: `02-scaling-cold-starts/large-video-model/README.md` (template copy, filled in Task 8)
- Create: `02-scaling-cold-starts/large-video-model/test_app.py` (template copy, real tests in Tasks 4–5)

**Interfaces:**
- Produces: `loader.tune_hf_env() -> None`, `loader.fetch_model() -> str` (returns model dir), `loader.fetch_taehv() -> pathlib.Path` (returns weights path), `loader.warm_page_cache(directory: str, parallelism: int = 32) -> float` (returns seconds). Constants: `loader.MODEL_REPO`, `loader.MODEL_REV`, `loader.MODEL_DIR`, `loader.TAEHV_WEIGHTS_URL`, `loader.TAEHV_WEIGHTS_SHA` — Task 4's `setup()` imports these.

- [ ] **Step 1: Copy the template**

```bash
mkdir -p 02-scaling-cold-starts/large-video-model
cp templates/example/README.md templates/example/test_app.py 02-scaling-cold-starts/large-video-model/
```

- [ ] **Step 2: Write loader.py (complete file)**

The ticket's deliverable: a self-contained parallel-loading module. Substitute `TAEHV_WEIGHTS_URL` with the spike-chosen variant (`taeltx2_3_wide.pth` shown; swap filename if the spike chose standard).

```python
"""Fast weight loading for 20+ GB models on fal.

The three-step recipe (each step timed and printed):
  1. tune_hf_env()      - HF Xet parallel range-gets, before any hub import
  2. fetch_model()      - snapshot_download to /data; warm boots skip the
                          network entirely via local_files_only=True
  3. warm_page_cache()  - parallel pre-read so from_pretrained hits RAM,
                          not cold NVMe/network storage

fal's /data is content-addressed in 4 MB chunks across storage tiers, so
32 parallel readers beat one sequential reader by a wide margin. See:
https://fal.ai/docs -> serverless -> optimizations -> parallel file loading.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
MODEL_REV = "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"  # pin: repo state 2026-05-21
MODEL_DIR = "/data/models/ltx-2.3-distilled-diffusers"

# Tiny video VAE for previews - madebyollin/taehv, MIT.
# Pinned to the branch that carries the LTX-2.3 "wide" decoder weights.
_TAEHV_REF = "32ac0146b11007cda5a57b60a3b35653361fb8a4"
TAEHV_WEIGHTS_URL = (
    f"https://raw.githubusercontent.com/madebyollin/taehv/{_TAEHV_REF}/taeltx2_3_wide.pth"
)
TAEHV_WEIGHTS_SHA = "<fill from spike: sha256 of the downloaded file>"
_TAEHV_DIR = Path("/data/models/taehv")


def tune_hf_env() -> None:
    """Parallelize Hugging Face Xet downloads. Call before importing huggingface_hub."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "1000000000000")  # 1 TB on /data
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # non-Xet repos


def fetch_model() -> str:
    """Download the pinned model snapshot to /data (skip metadata calls when warm).

    A cached-but-online snapshot_download still spends ~45 s on hub metadata
    round-trips; local_files_only=True returns in under a second.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    t0 = time.time()
    try:
        path = snapshot_download(
            MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, local_files_only=True
        )
        print(f"[loader] model cache hit in {time.time() - t0:.1f}s")
    except (LocalEntryNotFoundError, FileNotFoundError):
        print(f"[loader] cold download of {MODEL_REPO} (~95 GB) ...")
        path = snapshot_download(
            MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, max_workers=32
        )
        print(f"[loader] downloaded in {time.time() - t0:.1f}s")
    return path


def fetch_taehv() -> Path:
    """Fetch the pinned tiny-VAE weights to /data with atomic rename + sha check."""
    dest = _TAEHV_DIR / Path(TAEHV_WEIGHTS_URL).name
    if dest.exists():
        return dest
    _TAEHV_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=_TAEHV_DIR, delete=False) as tmp:
        urllib.request.urlretrieve(TAEHV_WEIGHTS_URL, tmp.name)
        digest = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
        if digest != TAEHV_WEIGHTS_SHA:
            os.unlink(tmp.name)
            raise RuntimeError(f"taehv weights sha mismatch: {digest}")
        os.rename(tmp.name, dest)  # atomic on /data: concurrent workers see all-or-nothing
    return dest


def warm_page_cache(directory: str, parallelism: int = 32) -> float:
    """Pre-read every file into the OS page cache with parallel readers.

    /data serves 4 MB content-addressed chunks; parallel readers pull from
    multiple cache nodes at once (sequential ~1 GB/s vs parallel ~6-15 GB/s).
    """
    t0 = time.time()
    subprocess.run(
        f"find {directory} -type f | xargs -P {parallelism} -I {{}} cat {{}} > /dev/null",
        shell=True,
        check=True,
    )
    elapsed = time.time() - t0
    print(f"[loader] page cache warm in {elapsed:.1f}s ({directory})")
    return elapsed
```

- [ ] **Step 3: Placeholder requirements.txt**

```
fal==1.78.2
# mirrored from App.requirements in Task 4 (CI enforces exact match)
```

- [ ] **Step 4: Commit**

```bash
git add 02-scaling-cold-starts/large-video-model/
git commit -m "feat(02.3): scaffold large-video-model + parallel loader (SERV-1303)"
```

---

### Task 3: Vendor taehv.py

**Files:**
- Create: `02-scaling-cold-starts/large-video-model/taehv.py`

**Interfaces:**
- Produces: `TAEHV(checkpoint_path: str)` — `torch.nn.Module`; `decode_video(x: Tensor[NTCHW latents], parallel=True, show_progress_bar=False) -> Tensor[NTCHW RGB, [0,1]]`. Task 5 consumes exactly this.

- [ ] **Step 1: Fetch the pinned file**

```bash
cd 02-scaling-cold-starts/large-video-model
curl -fsSL https://raw.githubusercontent.com/madebyollin/taehv/32ac0146b11007cda5a57b60a3b35653361fb8a4/taehv.py -o taehv.py
grep -n "WideMemBlock" taehv.py && grep -n "taeltx" taehv.py   # both must hit
```

- [ ] **Step 2: Prepend provenance header**

Add at the very top of `taehv.py` (keep the rest byte-identical):

```python
# Vendored from https://github.com/madebyollin/taehv
# commit 32ac0146b11007cda5a57b60a3b35653361fb8a4 (branch 2026_03_11_taeltx23_wide)
# License: MIT (c) Ollin Boer Bohan - see repository LICENSE.
# Vendored because no PyPI package exists and this repo pins all requirements.
```

- [ ] **Step 3: Commit**

```bash
git add taehv.py && git commit -m "feat(02.3): vendor taehv tiny-VAE decoder (MIT, pinned)"
```

---

### Task 4: app.py invoke path + smoke contract

**Files:**
- Create: `02-scaling-cold-starts/large-video-model/app.py`
- Modify: `02-scaling-cold-starts/large-video-model/test_app.py` (replace template test)
- Modify: `02-scaling-cold-starts/large-video-model/requirements.txt` (real mirror)

**Interfaces:**
- Consumes: `loader` functions/constants (Task 2), `TAEHV` (Task 3).
- Produces: `Input` (prompt/resolution/num_frames/seed/preview_every), `Output` (video: File, seed: int, timings: dict), `App._generate(request, preview_cb) -> tuple[str, int, dict]` (final mp4 path, seed, timings) — Task 5 wraps this. Endpoint `/`.

Substitute the smoke shape below (`57` frames, `"768x512"`) with the spike's chosen values if different.

- [ ] **Step 1: Write the failing test**

Replace `test_app.py` content:

```python
"""Smoke tests - run by the harness against the live ephemeral app.

'/' is the harness-smoked JSON endpoint; '/stream' (Task 5) is exercised
with a raw SSE client because the harness only speaks request/response.
"""
import base64
import json
import os

import httpx


def test_invoke_contract(app_client):
    response = app_client.run("/", {"prompt": "a red fox trotting through fresh snow at golden hour"})
    assert response["video"]["url"].startswith("https://")
    assert isinstance(response["seed"], int)
    assert response["timings"]["generate_s"] > 0
```

- [ ] **Step 2: Verify it fails/skips correctly**

Run: `python -m pytest 02-scaling-cold-starts/large-video-model/test_app.py -q`
Expected: `1 skipped` (no live app - fixture skips; the harness will run it for real).

- [ ] **Step 3: Write app.py (complete file)**

```python
"""Serve a 20+ GB video model - LTX-2.3 22B distilled on one H200.

Catalog slot 02.3 (SERV-1303). The three lessons:
  1. loader.py - parallel download + page-cache pre-read for ~95 GB of weights
  2. GPU sizing - ~70 GB resident bf16 needs H200-class VRAM (see README math)
  3. /stream    - intermediate-step tiny-VAE previews over SSE (Task 5)
"""
import time
from typing import Literal

import fal
from fal.toolkit import File
from pydantic import BaseModel, Field

import loader

# height, width - both divisible by 32 (pipeline requirement)
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "768x512": (512, 768),
    "512x768": (768, 512),
    "1024x576": (576, 1024),
}


def snap_frames(n: int) -> int:
    """LTX generates on an 8k+1 frame grid; snap down so users never 422 on it."""
    return max(9, ((n - 1) // 8) * 8 + 1)


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. Longer, cinematic prompts work well.",
        examples=["a red fox trotting through fresh snow at golden hour"],
        max_length=2000,
    )
    resolution: Literal["768x512", "512x768", "1024x576"] = Field(
        default="768x512", description="Output size (width x height)."
    )
    num_frames: int = Field(
        default=57, ge=9, le=121,
        description="Frame count at 24 fps; snapped to the model's 8k+1 grid.",
    )
    seed: int | None = Field(default=None, description="Reproducible seed; random if unset.")
    preview_every: int = Field(
        default=1, ge=1, le=4,
        description="Stream a preview every N denoising steps (/stream only).",
    )


class Output(BaseModel):
    video: File = Field(description="Final video with audio, hosted on the fal CDN.")
    seed: int = Field(description="Seed actually used.")
    timings: dict[str, float] = Field(description="Server-side stage timings, seconds.")


class App(fal.App, keep_alive=300):  # 5 min: video sessions come in bursts; idle H200 bills
    machine_type = "GPU-H200"  # ~70 GB resident bf16 weights - see README "GPU sizing"
    app_auth = "private"
    startup_timeout = 1800  # first-ever boot downloads ~95 GB to /data
    local_python_modules = ["loader", "taehv"]
    # Authoritative deploy-time deps; requirements.txt mirrors exactly (CI-enforced).
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        "torch==2.12.1",
        "diffusers==0.39.0",  # LTX2Pipeline landed in 0.37; 0.39 is current stable
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "av==16.0.1",  # diffusers.utils.encode_video (mp4 mux, incl. audio)
        "hf-transfer==0.1.9",
        "huggingface_hub==<fill from spike pip freeze>",
        "numpy==<fill from spike pip freeze>",
    ]

    def setup(self) -> None:
        """Load -> warm: every stage timed; these numbers are the example's story."""
        import torch
        from diffusers import LTX2Pipeline
        from taehv import TAEHV

        self.timings: dict[str, float] = {}
        loader.tune_hf_env()
        t0 = time.time()
        model_dir = loader.fetch_model()
        tae_path = loader.fetch_taehv()
        self.timings["fetch_s"] = round(time.time() - t0, 1)
        self.timings["page_cache_s"] = round(loader.warm_page_cache(model_dir), 1)

        t0 = time.time()
        self.pipe = LTX2Pipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
        self.tiny_vae = TAEHV(str(tae_path)).to("cuda", torch.bfloat16)
        self.timings["load_s"] = round(time.time() - t0, 1)

        # Warm up at the smoke shape (Input defaults) so the first request pays
        # no compile/autotune tax - autotune is shape-dependent, so warming at a
        # smaller shape would not help the real requests.
        t0 = time.time()
        self._generate(Input(prompt="warmup"), preview_cb=None)
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        print(f"[setup] {self.timings}")

    def _generate(self, request: Input, preview_cb) -> tuple[str, int, dict[str, float]]:
        """Run one distilled 8-step generation; returns (mp4 path, seed, timings)."""
        import tempfile

        import torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        height, width = RESOLUTIONS[request.resolution]
        num_frames = snap_frames(request.num_frames)
        seed = request.seed if request.seed is not None else int(time.time_ns() % 2**31)
        timings: dict[str, float] = {}

        t0 = time.time()
        try:
            video, audio = self.pipe(
                prompt=request.prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                sigmas=DISTILLED_SIGMA_VALUES,  # distilled schedule - fixed 8 steps
                num_inference_steps=8,
                guidance_scale=1.0,  # distilled = CFG off
                use_cross_timestep=True,  # LTX-2.3 conditioning
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="np",
                return_dict=False,
                callback_on_step_end=preview_cb,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise fal.exceptions.RequestCancelledException(  # replace in Task 5 Step 4 if
                # a typed 422 helper is preferred; keep worker alive either way
                "Out of GPU memory - lower num_frames or resolution and retry."
            )
        timings["generate_s"] = round(time.time() - t0, 1)

        t0 = time.time()
        out_path = tempfile.mktemp(suffix=".mp4")
        waveform = audio[0]
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(waveform)
        encode_video(
            video[0],
            24.0,
            out_path,
            audio=waveform.float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
        )
        timings["encode_s"] = round(time.time() - t0, 1)
        return out_path, seed, timings

    @fal.endpoint("/")
    def run(self, request: Input) -> Output:
        path, seed, timings = self._generate(request, preview_cb=None)
        return Output(video=File.from_path(path), seed=seed, timings=timings)
```

Note: confirm the OOM exception type against fal 1.78.2 during the harness run — if `fal.exceptions.RequestCancelledException` is wrong for a 500-with-message, raise `fastapi.HTTPException(status_code=500, detail=...)` instead. Keep `torch.cuda.empty_cache()` in the handler either way.

- [ ] **Step 4: Mirror requirements.txt exactly** (same lines as `App.requirements`, one per line).

- [ ] **Step 5: Static check + live verify**

```bash
python harness/static_check.py
python harness/check.py --example 02-scaling-cold-starts/large-video-model
```

Expected: static passes (layout, pins, payload derivable); deploy-check deploys, `/health` OK, smoke 2xx, `test_invoke_contract` passes, teardown confirmed. This is the slow loop (~20–40 min cold) — fix forward within the run when possible.

- [ ] **Step 6: Commit**

```bash
git add 02-scaling-cold-starts/large-video-model/
git commit -m "feat(02.3): LTX-2.3 22B distilled invoke endpoint, deploy-checked (SERV-1303)"
```

---

### Task 5: Preview machinery + /stream + SSE test

**Files:**
- Modify: `02-scaling-cold-starts/large-video-model/app.py`
- Modify: `02-scaling-cold-starts/large-video-model/test_app.py`

**Interfaces:**
- Consumes: `App._generate(request, preview_cb)`, `self.tiny_vae` (Task 4), `TAEHV.decode_video` (Task 3).
- Produces: `/stream` SSE endpoint; event schema `{"type": "preview", "step", "total_steps", "progress", "video", "width", "height", "frames"}` and `{"type": "result", "video": {...}, "seed", "timings"}`.

- [ ] **Step 1: Extend the test first**

Append to `test_app.py`:

```python
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
                events.append(json.loads(line[len("data: "):]))

    previews = [e for e in events if e["type"] == "preview"]
    assert len(previews) >= 2, f"expected >=2 previews, got {len(previews)}"
    steps = [p["step"] for p in previews]
    assert steps == sorted(set(steps)), "preview steps must be strictly increasing"
    header = base64.b64decode(previews[0]["video"].split(",", 1)[1])[:64]
    assert previews[0]["video"].startswith("data:video/mp4;base64,")
    assert b"ftyp" in header, "preview payload is not an mp4"
    assert events[-1]["type"] == "result"
    assert events[-1]["video"]["url"].startswith("https://")
```

Run: `python -m pytest 02-scaling-cold-starts/large-video-model/test_app.py -q` → `2 skipped`.

- [ ] **Step 2: Add the preview session to app.py**

Insert after the `snap_frames` function (module level):

```python
class _PreviewSession:
    """Per-request preview state: x0 extrapolation + latest-wins decode handoff.

    The callback runs on the denoise thread and must never block: it stores
    the newest latents (dropping any undecoded predecessor) and returns.
    A single decode worker drains it; the SSE generator drains `events`.
    """

    def __init__(self, app: "App", height: int, width: int, num_frames: int, every: int):
        import queue as _queue
        import threading

        self.app, self.height, self.width, self.num_frames = app, height, width, num_frames
        self.every = every
        self.events: "_queue.Queue[dict]" = _queue.Queue()
        self.done = threading.Event()
        self._latest: tuple[int, int, object] | None = None  # (step, total, x0_latents)
        self._lock = threading.Lock()
        self._prev: tuple[object, float] | None = None  # (latents, sigma)
        self._worker = threading.Thread(target=self._decode_loop, daemon=True)
        self._worker.start()

    # -- denoise thread ------------------------------------------------------
    def on_step_end(self, pipe, i, t, callback_kwargs):
        try:
            latents = callback_kwargs["latents"]
            sigma = float(pipe.scheduler.sigmas[i + 1])
            total = len(pipe.scheduler.sigmas) - 1
            if self._prev is not None and i % self.every == 0 and i < total - 1:
                prev_latents, prev_sigma = self._prev
                # Flow-matching latents move on straight paths: extrapolate to sigma=0
                # from two consecutive states for a clean-sample estimate.
                x0 = latents - sigma * (latents - prev_latents) / (sigma - prev_sigma)
                with self._lock:
                    self._latest = (i, total, x0.detach().clone())
            self._prev = (latents.detach().clone(), sigma)
        except Exception as exc:  # previews must never break generation
            print(f"[preview] callback error at step {i}: {exc!r}")
        return {}

    # -- decode worker thread --------------------------------------------------
    def _decode_loop(self):
        import time as _time

        while not self.done.is_set():
            with self._lock:
                job, self._latest = self._latest, None
            if job is None:
                _time.sleep(0.02)
                continue
            step, total, x0 = job
            try:
                self.events.put(self._render_event(step, total, x0))
            except Exception as exc:
                print(f"[preview] decode error at step {step}: {exc!r}")

    def _render_event(self, step: int, total: int, x0) -> dict:
        import base64
        import os
        import tempfile

        import torch
        from diffusers import LTX2Pipeline
        from diffusers.utils import encode_video

        pipe, tae = self.app.pipe, self.app.tiny_vae
        f_lat = (self.num_frames - 1) // 8 + 1
        with torch.inference_mode():
            lat = LTX2Pipeline._unpack_latents(
                x0, f_lat, self.height // 32, self.width // 32, 1, 1
            )
            lat = LTX2Pipeline._denormalize_latents(
                lat, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
            )
            rgb = tae.decode_video(  # NTCHW in, NTCHW [0,1] out, first 7 frames trimmed
                lat.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                parallel=True, show_progress_bar=False,
            )
            rgb = torch.nn.functional.interpolate(  # half-res previews: ~4x smaller events
                rgb[0].float(), scale_factor=0.5, mode="bilinear", align_corners=False
            )
            frames = (rgb.clamp(0, 1).permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        path = tempfile.mktemp(suffix=".mp4")
        try:
            encode_video(list(frames), 24.0, path)
            payload = base64.b64encode(open(path, "rb").read()).decode()
        finally:
            if os.path.exists(path):
                os.unlink(path)
        return {
            "type": "preview",
            "step": step,
            "total_steps": total,
            "progress": round((step + 1) / total, 3),
            "video": f"data:video/mp4;base64,{payload}",
            "width": frames.shape[2],
            "height": frames.shape[1],
            "frames": int(frames.shape[0]),
        }

    def close(self):
        self.done.set()
        self._worker.join(timeout=5)
```

- [ ] **Step 3: Add the /stream endpoint to App**

Insert below the `run` endpoint, plus the two imports at the top of the file (`import asyncio`, `import json`, and `from fastapi.responses import StreamingResponse`):

```python
    @fal.endpoint("/stream")
    async def stream(self, request: Input) -> StreamingResponse:
        height, width = RESOLUTIONS[request.resolution]
        session = _PreviewSession(
            self, height, width, snap_frames(request.num_frames), request.preview_every
        )

        async def event_source():
            loop = asyncio.get_running_loop()
            job = loop.run_in_executor(
                None, lambda: self._generate(request, preview_cb=session.on_step_end)
            )
            try:
                while True:
                    while not session.events.empty():
                        yield f"data: {json.dumps(session.events.get())}\n\n"
                    if job.done():
                        break
                    await asyncio.sleep(0.05)
                path, seed, timings = job.result()  # re-raises generation errors
                final = Output(video=File.from_path(path), seed=seed, timings=timings)
                yield f"data: {json.dumps({'type': 'result', **json.loads(final.model_dump_json())})}\n\n"
            finally:
                session.close()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"x-fal-sse-last-event-as-body": "1"},
        )
```

- [ ] **Step 4: Live verify (fal run loop first, then harness)**

```bash
fal run 02-scaling-cold-starts/large-video-model/app.py
# second terminal - raw SSE probe against the printed URL:
python - <<'EOF'
import httpx, json, os, sys
url = "<printed-url>/stream"
with httpx.stream("POST", url, json={"prompt": "a red fox in snow"},
                  headers={"Authorization": f"Key {os.environ['FAL_KEY']}"}, timeout=600) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            print(ev["type"], ev.get("step"), len(line) // 1024, "KB")
EOF
```

Expected: several `preview` lines with increasing steps, then one `result`. Then the full loop:

```bash
python harness/check.py --example 02-scaling-cold-starts/large-video-model
```

Expected: smoke + both tests pass, teardown confirmed.

- [ ] **Step 5: Commit**

```bash
git add 02-scaling-cold-starts/large-video-model/
git commit -m "feat(02.3): intermediate-step tiny-VAE previews over SSE (SERV-1303)"
```

---

### Task 6: client.py

**Files:**
- Create: `02-scaling-cold-starts/large-video-model/client.py`

**Interfaces:**
- Consumes: the deployed app's `/stream` SSE schema (Task 5).

- [ ] **Step 1: Write client.py (complete file)**

```python
"""Watch a generation live from the terminal.

Usage:
    pip install fal-client==1.0.0
    export FAL_KEY=...
    python client.py <owner>/large-video-model "a red fox trotting through fresh snow"

Writes preview_latest.mp4 as each preview arrives (open it in a looping
player) and final.mp4 (with audio) when generation completes.
"""
import base64
import pathlib
import sys
import urllib.request

import fal_client


def main() -> None:
    app_id, prompt = sys.argv[1], sys.argv[2]
    for event in fal_client.stream(app_id, {"prompt": prompt}, path="/stream"):
        if event["type"] == "preview":
            data = base64.b64decode(event["video"].split(",", 1)[1])
            pathlib.Path("preview_latest.mp4").write_bytes(data)
            bar = "#" * round(20 * event["progress"]) + "-" * (20 - round(20 * event["progress"]))
            print(f"[{bar}] step {event['step'] + 1}/{event['total_steps']}"
                  f" - preview_latest.mp4 updated ({len(data) // 1024} KB)")
        elif event["type"] == "result":
            urllib.request.urlretrieve(event["video"]["url"], "final.mp4")
            print(f"done: final.mp4 (seed {event['seed']}, {event['timings']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify against a live `fal run` URL** — `fal_client.stream` targets deployed app ids; against an ephemeral URL use the Task 5 raw-httpx probe instead, then run `client.py` for real in Task 9 after `fal deploy`. For now: `python -c "import ast; ast.parse(open('02-scaling-cold-starts/large-video-model/client.py').read())"`.

- [ ] **Step 3: Commit** — `git add client.py && git commit -m "feat(02.3): terminal streaming client"`

---

### Task 7: viewer.html

**Files:**
- Create: `02-scaling-cold-starts/large-video-model/viewer.html`

- [ ] **Step 1: Write viewer.html (complete file)**

```html
<!doctype html>
<!-- Demo viewer for the /stream endpoint. DEMO ONLY: pasting a FAL key into a
     browser page is fine on your machine, never in a shipped frontend - proxy
     through your backend instead. -->
<meta charset="utf-8">
<title>large-video-model - live preview</title>
<style>
  body { font: 15px system-ui; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  input, button, textarea { font: inherit; padding: .5rem; }
  #row { display: flex; gap: .5rem; margin-bottom: .5rem; } #row > * { flex: 1; }
  video { width: 100%; background: #000; border-radius: 8px; min-height: 320px; }
  #status { color: #666; margin: .5rem 0; font-variant-numeric: tabular-nums; }
  progress { width: 100%; }
</style>
<h1>Watch it generate</h1>
<div id="row">
  <input id="app" placeholder="app URL, e.g. https://fal.run/you/large-video-model">
  <input id="key" placeholder="FAL key (demo only!)" type="password">
</div>
<textarea id="prompt" rows="2">a red fox trotting through fresh snow at golden hour</textarea>
<div id="row"><button id="go">Generate</button></div>
<progress id="bar" value="0" max="1" hidden></progress>
<div id="status">idle</div>
<video id="v" autoplay muted loop playsinline></video>
<script>
const $ = id => document.getElementById(id);
$("go").onclick = async () => {
  const status = $("status"), bar = $("bar"), video = $("v");
  bar.hidden = false; bar.value = 0; status.textContent = "requesting...";
  const res = await fetch($("app").value.replace(/\/$/, "") + "/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": "Key " + $("key").value },
    body: JSON.stringify({ prompt: $("prompt").value }),
  });
  if (!res.ok) { status.textContent = "HTTP " + res.status + ": " + await res.text(); return; }
  const reader = res.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n"); buffer = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find(l => l.startsWith("data: "));
      if (!line) continue;
      const ev = JSON.parse(line.slice(6));
      if (ev.type === "preview") {
        bar.value = ev.progress;
        status.textContent = `step ${ev.step + 1}/${ev.total_steps} - live tiny-VAE preview`;
        video.src = ev.video;              // data:video/mp4;base64,... - swaps in place
      } else if (ev.type === "result") {
        bar.value = 1; video.src = ev.video.url; video.muted = false; video.loop = false;
        status.textContent = `done - seed ${ev.seed} (final has audio)`;
      }
    }
  }
};
</script>
```

- [ ] **Step 2: Manual verify** — open the file in a browser against the Task 5 `fal run` URL (or the Task 9 deployed URL); confirm previews swap live and the final plays with audio.

- [ ] **Step 3: Commit** — `git add viewer.html && git commit -m "feat(02.3): browser live-preview viewer"`

---

### Task 8: README + catalog wiring

**Files:**
- Modify: `02-scaling-cold-starts/large-video-model/README.md` (full rewrite of the template copy)
- Modify: `02-scaling-cold-starts/README.md` (02.3 row → link the folder)
- Modify: `llms.txt` (add to "Live examples")

- [ ] **Step 1: Write the README** following the template's §4-A order with this content map (write real prose, not these bullets):

- Intro: serving a 20+ GB model well + watching it generate.
- **What you get**: final mp4 (embed/link), a GIF of `viewer.html` mid-generation (record during Task 9's deployed run), the smoke-request JSON.
- **Run it**: template's pinned block, then `python client.py` + `viewer.html` usage.
- **How it works**: 3 subsections — `loader.py` (why parallel beats sequential on /data: 4 MB content-addressed chunks; the local_files_only warm path), **GPU sizing** (the ~70 GB bf16 math over the machine-types table: A100-40/L40-48 no, H100-80 no activation headroom, H200-141 yes — THE teaching section), previews (x0 extrapolation + tiny VAE + latest-wins threading; link the taehv repo).
- **Deploy**: `fal deploy`, `fal_client.stream()` snippet mirroring client.py, note `/stream` goes through the synchronous gateway (`fal.run`), not the queue.
- **Cost & tuning**: keep_alive rationale; ONE stamped measured table (from Task 9's deployed run: cold fetch / page-cache / load / warmup / warm generate at smoke shape; labeled with date + config); `preview_every` and resolution knobs; the fp8/smaller-GPU direction as untested cost reduction.
- **Licensing**: LTX-2 Community License ($10M ARR gate, AI-disclosure) — not Apache; link `Lightricks/LTX-2.3` LICENSE; taehv MIT attribution.
- **Next steps**: 03.1 (SSE for images) when it lands, 04.5 (multi-GPU video), parallel-file-loading + machine-types docs links.

- [ ] **Step 2: Tier README row** — in `02-scaling-cold-starts/README.md`, change the 02.3 row's Example cell to link `[Serve a 20+ GB video model...](large-video-model/)` and readiness to reflect "shipped" per the table style used by tier 00.

- [ ] **Step 3: llms.txt** — add under "Live examples": `- [02.3 large-video-model](02-scaling-cold-starts/large-video-model/README.md): LTX-2.3 22B distilled on one H200 - parallel loading for ~95 GB of weights, GPU-sizing math, and tiny-VAE intermediate previews streamed over SSE`

- [ ] **Step 4: Commit** — `git add -A && git commit -m "docs(02.3): README, tier table, llms.txt (SERV-1303)"`

---

### Task 9: Full validation + numbers + PR

**Files:**
- Modify: `.github/workflows/check.yml` (only if spike/Task 4 showed cold boot > 500 s: add `HARNESS_STARTUP_TIMEOUT: "1500"` to the deploy job env with a comment naming 02.3)
- Modify: `02-scaling-cold-starts/large-video-model/README.md` (stamped numbers + GIF)

- [ ] **Step 1: Full harness run** — `python harness/check.py --example 02-scaling-cold-starts/large-video-model` twice (cold-ish + warm): green, teardown confirmed both times.
- [ ] **Step 2: Registered deploy** — `fal deploy 02-scaling-cold-starts/large-video-model/app.py`; run `client.py` and `viewer.html` against it; record the GIF + the stamped numbers table; one 121-frame 1024x576 generation to confirm the envelope.
- [ ] **Step 3: Fill README numbers + GIF, commit** — `git commit -am "docs(02.3): stamped measured numbers from validated deploy"`
- [ ] **Step 4: Push + PR** targeting `main`, body: problem, evidence (deploy-check runs, stamped numbers), licensing note, spike results link, `Closes SERV-1303` reference (Linear DoD still needs the docs page — note it stays open per the merged ≠ Done convention). End with the repo's PR footer.
- [ ] **Step 5: Update SERV-1303** — comment with PR link + status; leave In Progress until the fal-docs page phase is scheduled separately.

## Self-review checklist (run after writing)

- Spec coverage: loader ✓ (T2), taehv ✓ (T3), invoke+sizing ✓ (T4), previews+SSE ✓ (T5), clients ✓ (T6-7), README/catalog ✓ (T8), validation/rollout ✓ (T9), spike gates ✓ (T1).
- All exact pins present except three marked `<fill from spike ...>` — deliberate spike outputs, listed in Task 1 Step 6.
- Type consistency: `_generate` tuple, `_PreviewSession` API, event schema names match across T4/T5/T6/T7 tests and clients.
