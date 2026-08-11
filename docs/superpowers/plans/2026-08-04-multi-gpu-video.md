# 04.5 Multi-GPU Frontier Video (fal.distributed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship example `04-genmedia-patterns/multi-gpu-video/` — WAN 2.2 T2V-A14B generating one video sharded across 4× GPU-H100 with `fal.distributed` + diffusers context parallelism — validated by a real deploy, per spec `docs/superpowers/specs/2026-08-04-multi-gpu-frontier-video-design.md` (SERV-1316).

**Architecture:** A `DistributedWorker` per GPU loads the full `WanPipeline` (bf16) and enables diffusers Ulysses sequence parallelism on both MoE expert transformers (`enable_parallelism(ContextParallelConfig(ulysses_degree=4))`); `DistributedRunner` in the `fal.App` orchestrates spawn/NCCL-init/invoke. All ranks run the same generation with the same seed; rank 0 writes the mp4 and hands back a file path; the endpoint uploads it to the CDN. A throwaway spike deploy (Task 2) gates the diffusers-CP path vs the upstream-Wan fallback (Task 2B).

**Tech Stack:** fal 1.78.2 (`fal.distributed`), diffusers 0.39.0 experimental context parallelism, WanPipeline (dual-transformer MoE + `boundary_ratio`), torch 2.12.1 NATIVE SDPA attention (verified CP-capable — no flash-attn needed), HF weights on the account-wide `/data` cache.

## Global Constraints

- Python **3.11** everywhere (client and runner must match — fal ships cloudpickled code).
- Pins (verified available 2026-08-04; these exact strings go in BOTH `App.requirements` and `requirements.txt` — the two must mirror byte-for-byte): `fal==1.78.2`, `torch==2.12.1`, `diffusers==0.39.0`, `transformers==5.3.0`, `accelerate==1.14.0`, `pyzmq==27.1.0`, `hf-transfer==0.1.9`, `ftfy==6.3.1`, `imageio==2.37.4`, `imageio-ffmpeg==0.6.0`.
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers` @ revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7` (HF main sha, resolved 2026-08-04; Apache-2.0, ungated — verified via HF API).
- `machine_type = "GPU-H100"` and `num_gpus = 4` must be **class attributes** (class kwargs raise `TypeError` at wrap time). `app_auth = "private"` explicit.
- `DistributedWorker.__call__` must accept the force-injected `streaming` kwarg; `runner.invoke()` returns `{"error": str(e)}` on worker exceptions instead of raising — always check.
- Every required `Input` field needs `Field(examples=[...])`; the harness smoke POST is built from them and must finish < 120 s.
- Heavy imports (`torch`, `diffusers`, `huggingface_hub`) only inside `setup()`/`__call__`/endpoint bodies — the free static job runpy-executes `app.py` with only `fal`/`httpx`/`pytest` installed. (`from fal.distributed import ...` at module level is safe — verified lazy.)
- Never name helper files `app_*.py` (each would be deployed as a variant). The spike lives in the scratchpad, never in the repo.
- Public-safety: no internal tooling names, registry paths, private forks, or bot-token patterns in any committed file. No performance claims except the single stamped table produced in Task 5.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Repo root (worktree): `/home/relic/fal-serverless-examples/.claude/worktrees/multi-gpu-frontier-video-93bcc9`. Scratchpad: `/tmp/claude-1000/-home-relic-fal-serverless-examples--claude-worktrees-multi-gpu-frontier-video-93bcc9/a8c24ba5-b1bc-4a8b-832d-ba7ad2597d54/scratchpad`.

---

### Task 1: Tooling preflight (venv, auth, toolkit API)

**Files:**
- Create: `<repo>/.venv/` (gitignored — verify; if not, add to `.gitignore` in this task)

**Interfaces:**
- Produces: `.venv/bin/python`, `.venv/bin/fal`, `.venv/bin/pytest` used by every later task; confirmed `File.from_path` signature used by Task 3's `app.py`.

- [ ] **Step 1: Create the venv and install harness deps**

```bash
cd /home/relic/fal-serverless-examples/.claude/worktrees/multi-gpu-frontier-video-93bcc9
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -r harness/requirements.txt
.venv/bin/python -c "import fal; print(fal.__version__)"
```
Expected: `1.78.2`. Verify `.venv` is ignored: `git check-ignore -q .venv && echo ignored` → `ignored` (add a `.gitignore` line if not).

- [ ] **Step 2: Verify fal auth works on this machine**

```bash
.venv/bin/fal apps list 2>&1 | head -5
```
Expected: a table (or empty list) — NOT an auth error. `~/.fal/auth0_token` exists; if this fails, stop and ask Rehan to `fal auth login`.

- [ ] **Step 3: Record the exact toolkit signatures Task 3 relies on**

```bash
.venv/bin/python - <<'EOF'
import inspect
from fal.toolkit import File
print("File.from_path", inspect.signature(File.from_path))
EOF
```
Expected: something like `(path, content_type=None, repository=..., ...)`. Note the exact parameter names; Task 3's endpoint uses `File.from_path(<path>, content_type="video/mp4")` — adjust that call if the signature differs.

- [ ] **Step 4: Read the harness client auth pattern (reused by the spike client)**

Read `harness/pytest_plugin.py` and note how it builds auth headers (`fal.sdk.get_credentials()` → `.to_headers()` or equivalent) and posts to the app URL. Task 2's `spike_client.py` copies this pattern exactly.

---

### Task 2: Spike probe deploy — THE GATE (approach A validation)

Throwaway probe. Decides: does diffusers-CP work on the A14B dual-transformer under `DistributedRunner`; which VRAM strategy; what smoke config fits < 100 s. **Never committed to the repo** (scratchpad only). Spec authorizes a same-day pivot to Task 2B on failure.

**Files:**
- Create: `<scratchpad>/spike/spike_app.py`
- Create: `<scratchpad>/spike/spike_client.py`
- Modify: `docs/superpowers/specs/2026-08-04-multi-gpu-frontier-video-design.md` (record outcome)
- Modify: `/home/relic/.claude/projects/-home-relic-fal-serverless-examples/memory/example-04-5-multi-gpu-video.md` (record outcome)

**Interfaces:**
- Produces: the **gate decision record** — chosen strategy (`offload` | `resident`), chosen smoke config (`num_frames`, `num_inference_steps` at 480p, measured seconds), measured 720p×81f×27-step wall-clock and per-rank peak VRAM, confirmed diffusers pin. Task 3 consumes these as constants.

- [ ] **Step 1: Write `spike_app.py`**

```python
"""04.5 spike probe - throwaway, never committed (SERV-1316).

Measures whether diffusers 0.39.0 context parallelism runs WAN 2.2 A14B
on 4x H100 under fal.distributed, which offload strategy fits, and how
long the candidate smoke/production configs take.
"""
import os
import time

import fal
from fal.distributed import DistributedRunner, DistributedWorker
from pydantic import BaseModel, Field

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
NUM_GPUS = 4


class SpikeWorker(DistributedWorker):
    def setup(self, strategy: str = "offload", **kwargs):
        import torch
        from diffusers import ContextParallelConfig, WanPipeline

        pipe = WanPipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
        )
        cp = ContextParallelConfig(ulysses_degree=self.world_size)
        pipe.transformer.enable_parallelism(config=cp)
        pipe.transformer_2.enable_parallelism(config=cp)
        pipe.vae.enable_tiling()
        if strategy == "offload":
            # Sequenced text_encoder->transformer->transformer_2->vae swap:
            # only the active component is resident (~30 GB peak weights).
            pipe.enable_model_cpu_offload(device=self.device)
        else:  # "resident" - everything on GPU, VRAM ceiling probe
            pipe.to(self.device)
        self.pipe = pipe
        self.rank_print(f"setup done, strategy={strategy}")

    def __call__(self, streaming: bool = False, **kwargs):
        import torch
        from diffusers.utils import export_to_video

        torch.cuda.reset_peak_memory_stats(self.device)
        generator = torch.Generator(device="cpu").manual_seed(kwargs.get("seed", 7))
        t0 = time.perf_counter()
        result = self.pipe(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs.get("negative_prompt", ""),
            height=kwargs.get("height", 480),
            width=kwargs.get("width", 832),
            num_frames=kwargs.get("num_frames", 33),
            num_inference_steps=kwargs.get("num_inference_steps", 8),
            guidance_scale=kwargs.get("guidance_scale", 3.5),
            guidance_scale_2=kwargs.get("guidance_scale_2", 4.0),
            generator=generator,
            output_type="np",
        )
        generation_seconds = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.rank_print(f"gen={generation_seconds:.1f}s peak_vram={peak_gb:.1f}GB")
        if self.rank != 0:
            return {}
        frames = result.frames[0]
        path = f"/tmp/spike-{int(time.time())}.mp4"
        t1 = time.perf_counter()
        export_to_video(frames, path, fps=16)
        return {
            "video_path": path,
            "generation_seconds": round(generation_seconds, 1),
            "export_seconds": round(time.perf_counter() - t1, 1),
            "rank0_peak_vram_gb": round(peak_gb, 1),
            "frames_shape": list(frames.shape),
        }


class SpikeInput(BaseModel):
    prompt: str = Field(examples=["a red panda napping on a mossy branch, soft light"])
    height: int = 480
    width: int = 832
    num_frames: int = 33
    num_inference_steps: int = 8
    seed: int = 7


class SpikeOutput(BaseModel):
    generation_seconds: float
    export_seconds: float
    rank0_peak_vram_gb: float
    endpoint_seconds: float
    video_size_mb: float
    frames_shape: list[int]


class SpikeApp(fal.App, keep_alive=600):
    machine_type = "GPU-H100"
    num_gpus = NUM_GPUS
    app_auth = "private"
    startup_timeout = 1800
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "pyzmq==27.1.0",
        "hf-transfer==0.1.9",
        "ftfy==6.3.1",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    async def setup(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        # Parent downloads once into the account-wide /data HF cache so the
        # 4 workers don't race the Hub.
        snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        print(f"[spike] snapshot_download {time.perf_counter() - t0:.0f}s")

        strategy = os.environ.get("SPIKE_STRATEGY", "offload")
        t1 = time.perf_counter()
        self.runner = DistributedRunner(worker_cls=SpikeWorker, world_size=NUM_GPUS)
        await self.runner.start(timeout=1800, strategy=strategy)
        print(f"[spike] runner.start {time.perf_counter() - t1:.0f}s (strategy={strategy})")

    @fal.endpoint("/")
    async def probe(self, request: SpikeInput) -> SpikeOutput:
        t0 = time.perf_counter()
        result = await self.runner.invoke(request.model_dump())
        if "error" in result:
            raise RuntimeError(f"worker error: {result['error']}")
        size_mb = os.path.getsize(result["video_path"]) / 1024**2
        return SpikeOutput(
            generation_seconds=result["generation_seconds"],
            export_seconds=result["export_seconds"],
            rank0_peak_vram_gb=result["rank0_peak_vram_gb"],
            endpoint_seconds=round(time.perf_counter() - t0, 1),
            video_size_mb=round(size_mb, 1),
            frames_shape=result["frames_shape"],
        )
```

- [ ] **Step 2: Write `spike_client.py`** (auth pattern copied from `harness/pytest_plugin.py` as read in Task 1 Step 4)

```python
"""POST probe payloads at the ephemeral spike app. Usage:
   python spike_client.py <app_url> <json_payload> [timeout_seconds]"""
import json
import sys
import time

import httpx
from fal.sdk import get_credentials

url = sys.argv[1].rstrip("/")
payload = json.loads(sys.argv[2])
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
headers = get_credentials().to_headers()

# health first
for _ in range(600):
    try:
        r = httpx.get(url + "/health", headers=headers, timeout=10)
        if r.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)
else:
    sys.exit("never became healthy")

t0 = time.perf_counter()
r = httpx.post(url + "/", json=payload, headers=headers, timeout=timeout)
print(f"HTTP {r.status_code} in {time.perf_counter() - t0:.1f}s")
print(json.dumps(r.json(), indent=2))
```
(If Task 1 Step 4 found a different header call than `get_credentials().to_headers()`, use what the harness actually does.)

- [ ] **Step 3: Deploy the spike ephemerally**

```bash
cd <scratchpad>/spike
<repo>/.venv/bin/fal run spike_app.py
```
Run in background; watch the log for the app URL (line like `https://fal.run/...` or `Access your exposed service at ...`). Startup does the FIRST-EVER A14B download to `/data` (~120 GB) — expect many minutes; `startup_timeout=1800` covers it. Record `snapshot_download` and `runner.start` timings from the log.

- [ ] **Step 4: Probe the offload strategy, in this order** (each via `spike_client.py <url> '<json>'`)

1. Warmup/correctness: `{"prompt": "a red panda napping on a mossy branch", "num_frames": 9, "num_inference_steps": 2}` — expect HTTP 200, sane `frames_shape` `[9, 480, 832, 3]`.
2. Candidate smoke config, twice (second run = warm number): `{"prompt": "a red panda napping on a mossy branch, soft light", "num_frames": 33, "num_inference_steps": 8, "seed": 7}` — record both `endpoint_seconds`.
3. Production config: `{"prompt": "a red panda napping on a mossy branch, soft light", "height": 720, "width": 1280, "num_frames": 81, "num_inference_steps": 27, "seed": 7}` with timeout 3600 — record `generation_seconds`, `rank0_peak_vram_gb`.
4. Sequential-stability: repeat probe 2 once more — three total invokes with no NCCL hang/desync.

- [ ] **Step 5: Decision matrix**

- Offload works and warm smoke `endpoint_seconds` < 100 s → **strategy=offload**, smoke config = 480p/33f/8st. Done.
- Offload works but smoke 100–120 s → shrink in this order until < 100 s: `num_inference_steps` 8→6, then `num_frames` 33→25. Record the final config.
- Offload OOMs or conflicts with CP hooks (traceback mentioning hooks/device mismatch) → redeploy with `SPIKE_STRATEGY=resident` (env var on the `fal run` invocation), rerun probes; resident peak VRAM must stay < 78 GB at the chosen configs (if 720p OOMs but 480p fits, 720p moves to "documented, needs H200" and the README says so honestly).
- CP itself fails (dispatch/enable_parallelism errors) → bounded rescue, max 3 attempts: (1) `pipe.transformer.set_attention_backend("native")` + same for `transformer_2` before `enable_parallelism`, (2) `ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)`, (3) pin `diffusers==0.36.0` instead and redeploy. Still failing → **pivot to Task 2B**, record why.

- [ ] **Step 6: Tear down and record**

Ctrl-C / kill the `fal run` process; confirm no runner is left (re-run `fal apps list` — ephemeral runners don't show there, so also check the dashboard or REST if in doubt; the run dying with the process is the expected behavior).

Append to the spec (`docs/superpowers/specs/2026-08-04-multi-gpu-frontier-video-design.md`), under Landing checklist, a new block:

```markdown
## Spike outcome (2026-08-XX)

- Verdict: approach A CONFIRMED / PIVOTED to B (reason)
- Strategy: offload | resident · diffusers pin: 0.39.0 | 0.36.0
- Smoke config: 480p × <frames>f × <steps> steps ≈ <s> s warm (target < 100 s)
- 720p × 81f × 27 steps: <s> s, rank-0 peak VRAM <gb> GB
- setup: snapshot_download <s> s (first boot only), runner.start <s> s
```

- [ ] **Step 7: Commit the spec update + update the memory file**

```bash
cd <repo>
git add docs/superpowers/specs/2026-08-04-multi-gpu-frontier-video-design.md
git commit -m "docs: record 04.5 spike outcome (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
Mirror the outcome into `/home/relic/.claude/projects/-home-relic-fal-serverless-examples/memory/example-04-5-multi-gpu-video.md`. Report the verdict to Rehan in chat (pivot is pre-authorized; don't block).

---

### Task 2B: CONTINGENCY — pivot to upstream Wan sequence parallelism (only if Task 2 fails)

Skip entirely if Task 2 confirmed approach A.

**Files:**
- Modify: `<scratchpad>/spike/spike_app.py` (worker internals only)
- Modify: spec Key-decisions row "Implementation" + memory file

**Interfaces:**
- Produces: the same gate decision record as Task 2 (strategy/config/timings), with worker internals = upstream Wan. Task 3's worker `setup()`/`__call__` bodies swap per Step 1; everything else in the plan is unchanged (same payload keys, same `{"video_path": ...}` return).

- [ ] **Step 1: Replace SpikeWorker internals with the upstream-Wan path**

Resolve the current `Wan-Video/Wan2.2` main sha: `curl -s https://api.github.com/repos/Wan-Video/Wan2.2/commits/main | head -20` — pin it as `WAN_REPO_SHA`. Worker `setup()` replaces the diffusers block with:

```python
import subprocess, sys
from pathlib import Path

repo_dir = Path("/data/repos/Wan2.2")
if not repo_dir.exists():
    subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.2.git", str(repo_dir)], check=True)
subprocess.run(["git", "-C", str(repo_dir), "checkout", WAN_REPO_SHA], check=True)
sys.path.insert(0, str(repo_dir))

import wan
from wan.configs import WAN_CONFIGS
from huggingface_hub import snapshot_download

ckpt = snapshot_download("Wan-AI/Wan2.2-T2V-A14B")  # official (non-diffusers) layout
self.model = wan.WanT2V(
    config=WAN_CONFIGS["t2v-A14B"],
    checkpoint_dir=ckpt,
    device_id=self.rank,
    rank=self.rank,
    t5_fsdp=True,
    dit_fsdp=True,
    use_sp=True,          # Ulysses sequence parallel across the runner's process group
    t5_cpu=False,
    convert_model_dtype=True,
)
```
and `__call__` uses `self.model.generate(prompt, size=(width, height), frame_num=num_frames, sampling_steps=steps, guide_scale=..., seed=seed, offload_model=False)` → rank 0 saves via the repo's `save_video` util → same `{"video_path": ...}` contract. Check the upstream repo's `generate.py` for exact kwarg names at execution time (they drift between releases) and whether `flash_attn` is a hard import — if it is, add the torch-2.12/cu-matching wheel to requirements and record it.

- [ ] **Step 2: Rerun Task 2 Steps 3–7** with the same probes, decision matrix (minus CP rescue), and recording. Update the spec Key-decisions "Implementation" row to name approach B and why.

---

### Task 3: Example folder — input surface first (TDD), then the full app

**Files:**
- Create: `04-genmedia-patterns/multi-gpu-video/test_app.py`
- Create: `04-genmedia-patterns/multi-gpu-video/app.py`
- Create: `04-genmedia-patterns/multi-gpu-video/requirements.txt`
- Create: `04-genmedia-patterns/multi-gpu-video/README.md` (skeleton now, full content in Task 6)

**Interfaces:**
- Consumes: Task 2's gate record (strategy, smoke config values). The code below assumes `offload` + 33f/8st; substitute the recorded values.
- Produces: `Input` (fields: `prompt: str`, `negative_prompt: str = ""`, `resolution: Literal["480p","720p"] = "480p"`, `num_frames: int = 33`, `num_inference_steps: int = 8`, `guidance_scale: float = 3.5`, `guidance_scale_2: float = 4.0`, `seed: int | None = None`), `Output` (`video: File`, `seed: int`, `generation_seconds: float`), `RESOLUTIONS: dict[str, tuple[int, int]]`, app class `MultiGpuVideo`. Tasks 4–6 rely on these exact names.

- [ ] **Step 1: Write the failing unit tests** — `04-genmedia-patterns/multi-gpu-video/test_app.py`:

```python
"""Free unit tests (static job) + one live harness test for 04.5.

The live test runs only under harness/check.py (the app_client fixture
skips otherwise), so plain `pytest` costs nothing. One live generation
total - each one spins 4 GPUs.
"""
import httpx
import pytest


# ---- free unit tests (no GPU, no deploy) ----

def test_num_frames_rule_rejects_off_grid():
    from app import Input

    with pytest.raises(ValueError):
        Input(prompt="x", num_frames=34)


def test_num_frames_rule_accepts_grid_values():
    from app import Input

    for n in (9, 17, 33, 49, 81):
        assert Input(prompt="x", num_frames=n).num_frames == n


def test_resolution_map():
    from app import RESOLUTIONS

    assert RESOLUTIONS["480p"] == (480, 832)
    assert RESOLUTIONS["720p"] == (720, 1280)


def test_defaults_are_the_ci_smoke_config():
    from app import Input

    i = Input(prompt="x")
    assert i.resolution == "480p"
    assert i.num_frames == 33
    assert i.num_inference_steps == 8


# ---- live test (harness only) ----

def test_generate_smoke(app_client):
    response = app_client.run(
        "/",
        {
            "prompt": "a red panda napping on a mossy branch, soft light",
            "resolution": "480p",
            "num_frames": 33,
            "num_inference_steps": 8,
            "seed": 7,
        },
    )
    assert response["seed"] == 7
    assert response["generation_seconds"] > 0
    url = response["video"]["url"]
    assert url.startswith("https://")
    head = httpx.get(url, timeout=60).content[:64]
    assert b"ftyp" in head[:16], f"not an mp4: {head!r}"
```
(If Task 2 changed the smoke config, use the recorded frames/steps here and in `Input` defaults/examples below — they must stay identical.)

- [ ] **Step 2: Run the unit tests to verify they fail**

```bash
cd <repo>
.venv/bin/python -m pytest 04-genmedia-patterns/multi-gpu-video/test_app.py -q
```
Expected: `4 failed, 1 skipped` — failures with `ModuleNotFoundError: No module named 'app'` (pytest's default import mode puts the test file's directory on sys.path, so `import app` resolves once app.py exists; the live test skips outside the harness). If collection itself misbehaves from the repo root, run from inside the folder: `cd 04-genmedia-patterns/multi-gpu-video && <repo>/.venv/bin/python -m pytest test_app.py -q`.

- [ ] **Step 3: Write `app.py`** (full file; worker internals per Task 2's verdict — this is the approach-A/offload version):

```python
"""Multi-GPU video inference - WAN 2.2 A14B sharded across 4 GPUs with fal.distributed.

One frontier text-to-video model, one video, four GPUs. DistributedRunner
spawns a worker per GPU and initializes torch.distributed (NCCL) before
setup() runs; diffusers context parallelism (Ulysses sequence parallelism)
then splits every denoising step's sequence across the ranks, so all four
GPUs cooperate on the same clip.

Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers - Apache-2.0, ungated. The MoE pair
of 14B expert transformers is ~70 GB in bf16: this is what "needs multi-GPU"
actually looks like.
"""
import os
import random
import time
import uuid
from typing import Literal

import fal
from fal.distributed import DistributedRunner, DistributedWorker
from fal.toolkit import File
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
NUM_GPUS = 4

# WAN generates at 16 fps; sizes are (height, width).
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 832),
    "720p": (720, 1280),
}


class Input(BaseModel):
    prompt: str = Field(
        description="What happens in the video. Long, specific prompts steer WAN best.",
        examples=["a red panda napping on a mossy branch, soft morning light"],
    )
    negative_prompt: str = Field(
        default="",
        description="What to avoid in the video.",
        examples=[""],
    )
    resolution: Literal["480p", "720p"] = Field(
        default="480p",
        description="480p (832x480) for fast iteration; 720p (1280x720) for final quality.",
        examples=["480p"],
    )
    num_frames: int = Field(
        default=33,
        ge=9,
        le=81,
        description="Clip length in frames at 16 fps; must satisfy (n - 1) % 4 == 0. 81 is ~5 s.",
        examples=[33],
    )
    num_inference_steps: int = Field(
        default=8,
        ge=2,
        le=40,
        description="Denoising steps. 8 is a fast preview; WAN's native quality default is 27.",
        examples=[8],
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance for the high-noise expert.",
    )
    guidance_scale_2: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance for the low-noise expert.",
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted. Echoed back; same seed + same inputs reproduces the clip.",
    )

    @field_validator("num_frames")
    @classmethod
    def _frames_on_grid(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError(
                "num_frames must satisfy (num_frames - 1) % 4 == 0, e.g. 9, 17, 33, 49, 81"
            )
        return v


class Output(BaseModel):
    video: File = Field(description="The generated clip (mp4, 16 fps) on the fal CDN.")
    seed: int = Field(description="The seed that produced this video.")
    generation_seconds: float = Field(
        description="Wall-clock seconds the four GPUs spent denoising this clip."
    )


class WanWorker(DistributedWorker):
    """One process per GPU. torch.distributed is already initialized here."""

    def setup(self, **kwargs) -> None:
        import torch
        from diffusers import ContextParallelConfig, WanPipeline

        # Every rank loads the full pipeline (sequence parallelism shards the
        # sequence, not the weights) from the /data HF cache the app warmed.
        pipe = WanPipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
        )

        # Ulysses attention across the runner's process group: each denoising
        # step's sequence is split over world_size ranks.
        cp = ContextParallelConfig(ulysses_degree=self.world_size)
        pipe.transformer.enable_parallelism(config=cp)
        pipe.transformer_2.enable_parallelism(config=cp)

        pipe.vae.enable_tiling()
        # The A14B MoE experts run in disjoint denoising stages; sequenced CPU
        # offload keeps only the active component resident (~30 GB of 80 GB).
        pipe.enable_model_cpu_offload(device=self.device)
        self.pipe = pipe

    def __call__(self, streaming: bool = False, **kwargs):
        # `streaming` is force-injected by the runner - accept and ignore it.
        import torch
        from diffusers.utils import export_to_video

        # Context parallelism needs identical inputs on every rank, so the
        # endpoint picks one seed and all ranks use it (unlike data
        # parallelism, where each rank would take a different seed).
        generator = torch.Generator(device="cpu").manual_seed(kwargs["seed"])
        t0 = time.perf_counter()
        result = self.pipe(
            prompt=kwargs["prompt"],
            negative_prompt=kwargs["negative_prompt"],
            height=kwargs["height"],
            width=kwargs["width"],
            num_frames=kwargs["num_frames"],
            num_inference_steps=kwargs["num_inference_steps"],
            guidance_scale=kwargs["guidance_scale"],
            guidance_scale_2=kwargs["guidance_scale_2"],
            generator=generator,
            output_type="np",
        )
        generation_seconds = round(time.perf_counter() - t0, 1)

        if self.rank != 0:
            # Ranks cooperated via NCCL inside the pipeline call; only rank 0
            # materializes the artifact and replies.
            return {}

        path = f"/tmp/wan22-{uuid.uuid4().hex}.mp4"
        export_to_video(result.frames[0], path, fps=16)
        return {"video_path": path, "generation_seconds": generation_seconds}


class MultiGpuVideo(fal.App, keep_alive=300):
    machine_type = "GPU-H100"
    num_gpus = NUM_GPUS  # class attribute, not a class kwarg
    app_auth = "private"
    startup_timeout = 1800  # first-ever boot downloads ~120 GB into /data
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "pyzmq==27.1.0",  # fal.distributed's transport; not a dependency of fal itself
        "hf-transfer==0.1.9",
        "ftfy==6.3.1",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    async def setup(self) -> None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import snapshot_download

        # Download once from the parent so the 4 workers don't race the Hub;
        # they load from the account-wide /data cache.
        snapshot_download(MODEL_ID, revision=MODEL_REVISION)

        self.runner = DistributedRunner(worker_cls=WanWorker, world_size=NUM_GPUS)
        await self.runner.start(timeout=1800)

        # Warm up the whole distributed path before the first request: worker
        # errors surface as an "error" key, not an exception.
        warmup = await self.runner.invoke(
            {
                "prompt": "warmup",
                "negative_prompt": "",
                "height": 480,
                "width": 832,
                "num_frames": 9,
                "num_inference_steps": 2,
                "guidance_scale": 3.5,
                "guidance_scale_2": 4.0,
                "seed": 0,
            }
        )
        assert "error" not in warmup, f"warmup failed: {warmup}"
        os.remove(warmup["video_path"])

    @fal.endpoint("/")
    async def generate(self, request: Input) -> Output:
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        height, width = RESOLUTIONS[request.resolution]
        result = await self.runner.invoke(
            {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "height": height,
                "width": width,
                "num_frames": request.num_frames,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "guidance_scale_2": request.guidance_scale_2,
                "seed": seed,
            }
        )
        if "error" in result:
            # 5xx: not billed, and the platform retries on healthy hardware.
            raise HTTPException(status_code=500, detail=f"generation failed: {result['error']}")

        video = File.from_path(result["video_path"], content_type="video/mp4")
        os.remove(result["video_path"])
        return Output(
            video=video,
            seed=seed,
            generation_seconds=result["generation_seconds"],
        )
```
(Adjust `File.from_path(...)` kwargs to the signature recorded in Task 1 Step 3. If Task 2 chose `resident`, replace the `enable_model_cpu_offload` line with `pipe.to(self.device)`. If Task 2B pivoted, swap the worker's `setup`/`__call__` bodies per Task 2B Step 1 — payload keys and return contract stay identical.)

- [ ] **Step 4: Write `requirements.txt`** — byte-mirror of `App.requirements`, one per line with the same comments stripped:

```
fal==1.78.2
torch==2.12.1
diffusers==0.39.0
transformers==5.3.0
accelerate==1.14.0
pyzmq==27.1.0
hf-transfer==0.1.9
ftfy==6.3.1
imageio==2.37.4
imageio-ffmpeg==0.6.0
```

- [ ] **Step 5: Write the README skeleton** (real title + slot + one honest paragraph; full §4-A content lands in Task 6):

```markdown
# Multi-GPU video inference with fal.distributed

Serve WAN 2.2 A14B - a 2x14B-expert text-to-video model whose weights alone
outgrow a single GPU's memory budget - across 4 GPUs with Ulysses sequence
parallelism: one prompt, one video, four GPUs cooperating on every denoising
step.

> Catalog slot `04.5` ([SERV-1316](https://linear.app/features-and-labels/issue/SERV-1316))

_Full walkthrough landing with the validated deploy - see app.py._
```

- [ ] **Step 6: Run the unit tests to verify they pass**

```bash
cd <repo>/04-genmedia-patterns/multi-gpu-video && <repo>/.venv/bin/python -m pytest test_app.py -q
```
Expected: `4 passed, 1 skipped` (the live test skips outside the harness).

- [ ] **Step 7: Run the static checks**

```bash
cd <repo> && .venv/bin/python harness/static_check.py
```
Expected: PASS for `04-genmedia-patterns/multi-gpu-video` (required files, pinned deps, mirror equality, examples on required fields, app_auth present).

- [ ] **Step 8: Commit**

```bash
git add 04-genmedia-patterns/multi-gpu-video/
git commit -m "feat(04-genmedia-patterns): multi-gpu-video app + tests (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Live deploy-check green (the DoD gate)

**Files:**
- Modify: `04-genmedia-patterns/multi-gpu-video/app.py` (only if fixes needed)

**Interfaces:**
- Consumes: Task 3's app; Task 2's warmed `/data` cache (same account → no first-boot download).
- Produces: a green `harness/check.py` run — deploy + health + smoke + tests + confirmed teardown. Numbers for Task 5's table (smoke wall-clock from the harness log).

- [ ] **Step 1: Run the harness against the example**

```bash
cd <repo>
.venv/bin/python harness/check.py --example 04-genmedia-patterns/multi-gpu-video 2>&1 | tee <scratchpad>/harness-run-1.log
```
Expected sequence in the log: app URL within 900 s → health OK within 600 s (cache is warm from Task 2; setup = 4×load + CP init + warmup) → smoke POST 200 < 120 s → `pytest test_app.py` all green (5 passed) → teardown CONFIRMED. Exit code 0.

- [ ] **Step 2: Iterate on failures** (each fix = its own commit; rerun Step 1 after each)

Known failure modes and the fix order:
- Health timeout (> 600 s in setup): measure which phase via log prints; if worker load dominates, this is real — options: keep, and note that CI needs `HARNESS_STARTUP_TIMEOUT` raised in `check.yml` env (repo-level change — flag to Rehan, don't do unilaterally); or drop the in-setup warmup (moves cost to first request — worse; last resort).
- Smoke > 120 s: shrink the `Input` defaults/examples AND `test_app.py` smoke payload AND README numbers together (steps 8→6, then frames 33→25 — keep all three in lockstep).
- Teardown unconfirmed: rerun once (transient REST lag); if reproducible, inspect `check.py` log for the runner state and stop — do not leave 4×H100 running; verify death in the dashboard before continuing.

- [ ] **Step 3: Commit any fixes**

```bash
git add 04-genmedia-patterns/multi-gpu-video/
git commit -m "fix(04-genmedia-patterns): multi-gpu-video deploy-check fixes (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Registered deploy, 720p manual validation, stamped numbers, demo asset

**Files:**
- Create: `<scratchpad>/validate_720p.py`
- Modify: spec (Spike outcome block gains the final stamped numbers)

**Interfaces:**
- Consumes: Task 4's green app.
- Produces: the README's stamped-numbers values, the demo clip CDN URL + response JSON for "What you get", and a registered deploy for the playground link. This closes the ticket's "validated deployment" DoD clause.

- [ ] **Step 1: Deploy registered**

```bash
cd <repo>/04-genmedia-patterns/multi-gpu-video
<repo>/.venv/bin/fal deploy app.py
```
Record the printed app alias/URL (expected `<username>/multi-gpu-video` — check the exact alias the CLI prints; the README Deploy section must use the real one).

- [ ] **Step 2: Write and run the 720p validation client**

```python
"""One-time manual validation: the full 720p production config via the queue."""
import json

import fal_client

handle = fal_client.submit(
    "<alias-from-step-1>",
    arguments={
        "prompt": "a red panda napping on a mossy branch, soft morning light",
        "resolution": "720p",
        "num_frames": 81,
        "num_inference_steps": 27,
        "seed": 7,
    },
)
result = handle.get()
print(json.dumps(result, indent=2))
```
```bash
uv pip install -p <repo>/.venv/bin/python fal-client
<repo>/.venv/bin/python <scratchpad>/validate_720p.py | tee <scratchpad>/validate-720p.json
```
Expected: JSON with `video.url`, `seed: 7`, `generation_seconds`. Download the clip, watch it (send to Rehan via chat), keep the URL.

- [ ] **Step 3: Also capture the smoke-config clip** (same script, 480p/33f/8st) — this is the fast-preview sample for the README.

- [ ] **Step 4: Assemble the stamped table** (goes in README Task 6; values from Task 2/4/5 logs):

| Config | Wall-clock | Notes |
|---|---|---|
| 480p · 33 frames · 8 steps | `<s>` s | CI smoke config, measured `<date>` |
| 720p · 81 frames · 27 steps | `<s>` s | production config, measured `<date>` |

Stamp line: `Measured <date> on 4× GPU-H100, fal==1.78.2, diffusers==<pin>, app revision <git sha>. One-time measurement, not a guarantee.`

- [ ] **Step 5: Update the spec's Spike-outcome block with final numbers; commit**

```bash
git add docs/superpowers/specs/2026-08-04-multi-gpu-frontier-video-design.md
git commit -m "docs: 04.5 validated-deploy numbers (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: README (§4-A, full)

**Files:**
- Modify: `04-genmedia-patterns/multi-gpu-video/README.md` (replace skeleton)

**Interfaces:**
- Consumes: Task 5's clip URLs, response JSON, stamped table, registered alias.

- [ ] **Step 1: Write the full README** in exactly this section order (mirrors `templates/example/README.md`):

1. `# Multi-GPU video inference with fal.distributed` + problem paragraph (frontier model too big/slow for one GPU; no torchrun scripts, no ready-file handshakes — two classes and an `invoke`) + `> Catalog slot 04.5 (SERV-1316)` blockquote.
2. `## What you get` — OUTPUT FIRST: link/embed the 720p demo clip (Task 5 URL), then the response JSON (video/seed/generation_seconds).
3. `## Run it` — `pip install fal==1.78.2` → `fal auth login` → `fal run app.py`; note what startup does (first-ever boot downloads ~120 GB to `/data`; later boots load from cache).
4. `## How it works` — H3 subsections: **One worker per GPU** (DistributedRunner spawn, NCCL init before `setup()`, READY handshake vs the manual pattern this replaces — generic description + upstream Wan repo's own torchrun instructions as the public "before"); **Sequence parallelism, not data parallelism** (one video faster vs more videos; same seed on every rank; link `serverless/distributed/overview` docs); **The MoE VRAM budget** (~70 GB bf16 per rank replicated; the offload line; why `num_gpus=4` H100 works); **The error contract** (`{"error": ...}` + warmup assert; single-flight runner → `max_concurrency` honesty).
5. `## Deploy` — `fal deploy app.py`, playground link for the registered alias, Python (`fal_client.submit`) + JS (`fal.subscribe`) + curl queue examples, and the **manual validation protocol** (the Task 5 720p run, documented as the repeatable procedure).
6. `## Cost & tuning` — multi-GPU bills `gpu_count × duration` (4× the H100 rate for this app); `keep_alive=300` trade-off; when 2/4/8 GPUs make sense (Ulysses degree must divide attention heads); the stamped numbers table + stamp line; smoke vs production config knobs.
7. `## Next steps` — 04.2 (text-to-video fundamentals), `serverless/distributed/streaming` (SSE previews), 02.1 (cold starts).

Rules: no performance claims beyond the stamped table; no internal tooling mentions; every command copy-pasteable; `fal run app.py` / `fal deploy app.py` paths relative to the example folder.

- [ ] **Step 2: Self-check the README against the app** — every field name, default, and command in the README must match `app.py` exactly (`resolution` values, frames rule, alias). Diff mentally section by section.

- [ ] **Step 3: Commit**

```bash
git add 04-genmedia-patterns/multi-gpu-video/README.md
git commit -m "docs(04-genmedia-patterns): multi-gpu-video README (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Landing — catalog wiring, Linear, final verification

**Files:**
- Modify: `04-genmedia-patterns/README.md` (slot table row)
- Modify: `llms.txt` (Live examples list)
- Modify: root `README.md` (only if it states per-tier live counts — check)

**Interfaces:**
- Consumes: everything prior.
- Produces: the branch ready for PR + Linear reflecting reality.

- [ ] **Step 1: Wire the catalog**

In `04-genmedia-patterns/README.md`, the 04.5 row gains the folder link: `| `04.5` | [Multi-GPU inference on a frontier video model (fal.distributed)](multi-gpu-video/) | ... |` (match the exact table format of the merged tiers). Add the `llms.txt` line following its existing one-line-per-example format. Check root `README.md` for any tier-04 count/status text to update.

- [ ] **Step 2: Full free verification**

```bash
cd <repo>
.venv/bin/python harness/static_check.py
.venv/bin/python -m pytest -q
git status --short
```
Expected: static PASS, all unit tests green + live tests skipped, no untracked example files.

- [ ] **Step 3: Linear** — set SERV-1316 to In Progress (if not already) and post the scope comment: model + revision, 4×H100 + parallelism decision, spike verdict, validated-deploy evidence (harness green + 720p run), source lives in fal-serverless-examples (supersedes the ticket's fal-demos mention per CONTEXT.md), fal-docs page = separate later phase.

- [ ] **Step 4: Commit + wrap**

```bash
git add 04-genmedia-patterns/README.md llms.txt README.md
git commit -m "docs: wire 04.5 multi-gpu-video into catalog (SERV-1316)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
Then run superpowers:verification-before-completion, and superpowers:finishing-a-development-branch (PR: `feat(04-genmedia-patterns): multi-GPU video inference with fal.distributed (SERV-1316)`; ask Rehan whether the spec/plan docs stay on the branch or get dropped, per the standing owner's-call).
