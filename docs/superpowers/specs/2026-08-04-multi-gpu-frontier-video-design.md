# Design: 04.5 — Multi-GPU inference on a frontier video model (fal.distributed)

**Catalog slot** 04.5 · [SERV-1316](https://linear.app/features-and-labels/issue/SERV-1316) · Folder: `04-genmedia-patterns/multi-gpu-video/`

**Status**: approved design 2026-08-04, pre-implementation. Working doc for the branch; drop or keep at PR time (owner's call).

**Owner decisions (Rehan, 2026-08-04)**: ships in the Aug-7 release slice · target config **4× GPU-H100** · approach = **diffusers-native context parallelism, with the upstream-Wan sequence-parallel/FSDP path as pre-agreed fallback, gated by a day-1 spike deploy** · invoke-only endpoint (no SSE) · one stamped measured-numbers table allowed · the "manual multi-GPU" before-story is told generically only.

## Problem & story

Frontier video models are too big or too slow for one GPU, and the traditional serving answer is a pile of orchestration hacks: a torchrun subprocess, a ready-file handshake, an HTTP hop to a rank-0 server, and hand-rolled broadcast loops that desync the process group the first time a worker throws. `fal.distributed` replaces all of that with two classes — `DistributedRunner` spawns one worker per GPU, initializes NCCL before your code runs, and gives you `await runner.invoke(payload)` from a normal `fal.App` endpoint.

04.5 shows the real thing: **WAN 2.2 T2V-A14B (MoE, 2×14B experts, Apache-2.0, ungated — the same model family fal serves in production) generating one 720p video sharded across 4 GPUs with Ulysses sequence parallelism.** It replaces the SDXL data-parallel walkthrough as the `fal.distributed` showcase: sequence parallelism makes *one* generation faster, which is what you want when the model is the bottleneck — data parallelism (the old page's pattern) only makes *more* outputs.

Platform features taught: `num_gpus` × `machine_type`, `DistributedRunner`/`DistributedWorker` lifecycle (spawn → NCCL init → READY handshake), rank-0 result semantics, the `{"error": ...}` return contract, single-flight invocation, multi-GPU billing math.

## Load-bearing facts (verified 2026-08-04, fal 1.78.2 source + diffusers wheels + HF API)

1. **`fal.distributed` is process orchestration only** — no parallelism primitives. It spawns workers via `torch.multiprocessing.spawn` (cloudpickle-patched, so worker classes can live in `app.py`), sets `RANK`/`WORLD_SIZE`/`MASTER_*` env, and calls `dist.init_process_group(backend="nccl", device_id=cuda:rank)` **before** `setup()` runs — any sequence-parallel library can attach to the ready-made default group. Requests: parent → rank 0 over ZMQ; rank 0 rebroadcasts via `dist.broadcast_object_list`; **all ranks execute `__call__`, only rank 0's return value comes back**.
2. **API gotchas the example must handle** (all verified in source): worker exceptions do **not** raise on the caller side — `invoke()` returns `{"error": str(e)}` and the endpoint must check; the runner force-injects a `streaming` kwarg into every payload, so `__call__` must accept it; an invocation lock serializes requests (one in flight per runner); `pyzmq` is required at runtime but is **not** a dependency of the `fal` package — it must be pinned in `App.requirements`; PIL objects returned through `invoke()` get JPEG-re-encoded, so video is handed off by file path instead.
3. **`num_gpus` works through the existing harness with zero changes**: `fal.App` ClassVar flowing to `MachineRequirements` on both the ephemeral-run path CI uses and the deploy path. It must be a class *attribute* — passing it as a class kwarg raises `TypeError` at wrap time.
4. **Model**: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, Apache-2.0, ungated. Weights per rank under sequence parallelism (SP replicates weights; it shards the sequence): 2 expert transformers ≈ 28.6 GB bf16 each + UMT5-xxl encoder 11.4 GB + VAE ≈ 1 GB ≈ **~70 GB** — H100 (80 GB) needs an offload strategy (§App). Generation constraints: `num_frames` must satisfy `(n−1) % 4 == 0`; native defaults 81 frames @ 16 fps, 720p = 1280×720; the two experts take separate guidance scales. Pin the HF revision current at implementation time.
5. **diffusers ships experimental context parallelism**: `ContextParallelConfig(ulysses_degree=...)` + `ModelMixin.enable_parallelism()`, with a built-in `_cp_plan` for `WanTransformer3DModel` (verified in the 0.36.0 wheel). The API is explicitly experimental; **0.39.0 (repo-standard pin) is tried first by the spike, 0.36.0 is the fallback pin** if the API moved. Nobody has validated this path at fal — hence the spike gate.
6. **Harness limits** (defaults, no CI overrides): 900 s to URL, 600 s to healthy (app-side `startup_timeout` does **not** raise this), **120 s synchronous smoke POST to `/`** built from `Field(examples=[...])`, 600 s test budget with 120 s per in-test request, teardown must be REST-confirmed or the run fails (the money-leak guard — extra important at 4 GPUs). Multi-GPU bills `gpu_count × duration`. The `/data` HF cache is account-wide: dev/spike deploys pre-warm it, so CI boots load from cache.

## Files

```
04-genmedia-patterns/multi-gpu-video/
  app.py            # WanWorker (DistributedWorker) + MultiGPUVideoApp (fal.App) — one deployable app
  requirements.txt  # exact mirror of App.requirements, fully pinned
  test_app.py       # free unit tests (input validation) + live tests via app_client fixture
  README.md         # §4-A order
```

The spike probe app is throwaway — run from a scratch branch, never committed.

## App (app.py)

Class settings: `machine_type = "GPU-H100"` · `num_gpus = 4` · `app_auth = "private"` · `max_concurrency = 1` (the runner is single-flight; say so honestly) · `startup_timeout = 1200` · `keep_alive = 300`. Requirements pinned: `fal==1.78.2`, `torch==2.12.1`, `diffusers` (0.39.0 or 0.36.0, spike decides), `transformers`, `accelerate`, `pyzmq`, `hf-transfer`, `ftfy`, `imageio[ffmpeg]`-class export deps — exact set fixed during implementation, mirrored in requirements.txt.

**`WanWorker.setup()`**: `snapshot_download` (pinned revision) into the `/data` HF cache → load `WanPipeline` bf16 → `enable_parallelism(ContextParallelConfig(ulysses_degree=self.world_size))` on **both** expert transformers → VRAM strategy per the ladder below.

**`WanWorker.__call__(streaming=False, **kwargs)`**: all ranks run the same `pipe(...)` call with the **same seed** (context parallelism requires identical inputs across ranks — the README contrasts this with the old data-parallel per-rank-seed pattern); rank 0 exports the mp4 to a uuid'd path under `/tmp` and returns `{"video_path": ..., "timings": ...}`; other ranks return `{}`.

**VRAM ladder on H100 (spike resolves, in order):**
(a) UMT5 encoder on CPU → ~58 GB resident, ~21 GB activation headroom at 720p / Ulysses degree 4;
(b) additionally group-offload the inactive MoE expert (the experts run in disjoint denoising stages) → ~30 GB resident (active expert + VAE);
(c) fallback **approach B**: upstream `Wan-Video/Wan2.2` repo at a pinned commit under the same `DistributedRunner`, `use_sp=True` + `dit_fsdp`/`t5_fsdp` — FSDP shards weights to ~14 GB/rank and VRAM stops being a constraint. Same folder, README skeleton, endpoint, and CI shape either way; only the worker internals change.

**Endpoint `POST /`** (single endpoint): `prompt`, `negative_prompt`, `resolution` enum (`480p`/`720p`), `num_frames` (bounded, `(n−1) % 4 == 0` enforced with a 422 `FieldException`), `num_inference_steps` (bounded), `seed`. `Field(examples=[...])` = a **small real generation** (480p, ~33 frames, ~8 steps) targeted to finish inside the 120 s CI smoke — spike confirms; shrink frames/steps if not. Output: `video: File` (CDN), echoed seed/config, measured wall-clock as response metadata. Endpoint checks the `{"error": ...}` contract and re-raises as 5xx (not billed; platform retries on healthy hardware). `App.setup()` ends with a tiny warm-up `invoke` asserting no error key.

The documented production config — 720p, 81 frames, 27 steps — runs through the queue client (`fal_client.subscribe`), never raw sync HTTP.

## CI & tests

- **Free (static job, every PR)**: required-files/pin-mirror checks; unit tests for frame-count/resolution/bounds validation. Heavy imports live inside `setup()`/endpoint bodies so the static runpy pass stays cheap.
- **Live (deploy-check)**: harness deploys the real 4×H100 app ephemerally → health through `runner.start()` → smoke `POST /` with the examples config → live tests assert the video URL resolves, mp4 magic bytes, echoed seed, no error key. Rides the pre-warmed `/data` cache to stay inside the 600 s health window.
- **One-time manual validation protocol** (README Deploy section, closes the DoD): full 720p×81f×27-step generation via `fal_client.subscribe` against a registered deploy, output linked/embedded in the README.

## README (§4-A order)

Problem paragraph (model too big/slow for one GPU; serving it without torchrun scripts and ready-file hacks) → `> Catalog slot 04.5 (SERV-1316)` blockquote → **What you get**: sample clip + response JSON first → **Run it** → **How it works**: worker lifecycle, why every rank runs the same call, sequence- vs data-parallelism (when each is right, link distributed overview docs), the error-contract and single-flight gotchas, `num_gpus` wiring → **Deploy**: playground + Python/JS/curl clients + the manual validation protocol → **Cost & tuning**: multi-GPU bills `gpu_count × duration`, keep_alive math, 2 vs 4 vs 8 GPUs, why `max_concurrency = 1` → **Next steps**: 04.2 (text-to-video), distributed streaming docs, 02.1 (cold starts).

**Numbers policy**: exactly one stamped table from real validation runs (date, machine config, fal/diffusers versions; wall-clock for the smoke config and the 720p config) per the 02.2 "backed by a reproducible run" precedent. No other performance claims; no unbacked speedup marketing.

**Before-story policy**: the manual-orchestration pain is described generically and via the upstream Wan repo's public torchrun instructions. No internal code quoted, linked, or attributed; no internal tooling named.

## Non-goals

LTX-class text-to-video walkthrough (**04.2**) · SSE preview streaming (**03.1** + distributed streaming docs) · per-user LoRAs (**04.3**) · LoRA training (**04.4**) · cold-start optimization (**02.1**) · autoscaling/cost modeling (**02.2**) · I2V/S2V flavors · distributed *training* (DDP) · production inference-optimization stacks (quantization/caching) · the fal-docs page rewrite (separate later phase branched from **current** fal-docs `origin/main` — local clone is stale and on a feature branch; the ticket's gallery-card, cross-link, and redirect fixes happen there).

## Key decisions log

| Decision | Choice | Why |
|---|---|---|
| Model | Wan 2.2 T2V-A14B-Diffusers | Apache-2.0, ungated, genuinely needs multi-GPU (~70 GB/rank), same family fal serves; 5B fits one GPU (weak story), LTX owned by 04.2, Hunyuan license-restricted |
| Parallelism | Ulysses sequence parallel (one video faster) | The differentiated answer to SERV-1316's open question; data parallelism is what the old SDXL page already taught |
| GPU config | 4× GPU-H100 | Owner call: bigger parallelism demo at manageable cost; H100 VRAM gap handled by offload ladder or FSDP fallback |
| Implementation | diffusers-native CP, spike-gated, upstream-Wan SP+FSDP fallback | Pip-pinnable ~150-line teaching code if it works; prod-shaped proven path if it doesn't; decided by one probe deploy, same-day pivot |
| Endpoint surface | Single invoke-only `POST /` | Scope discipline for the Aug-7 slice; streaming is 03.1's story |
| Video handoff | File path from rank 0, endpoint uploads via `File` | Avoids the runner's JPEG re-encode of pickled images; worker and app share the container FS |
| CI smoke | Small real generation via Field examples (480p/~33f/~8 steps) | Must fit the 120 s sync smoke; still exercises the full distributed path |
| Numbers | One stamped measured table | 02.2 precedent; only claims backed by a real run |
| Spec fate | Committed on branch, owner decides at PR time | 04.3 precedent (dropped) vs 04.4 precedent (status-line caveat) |

## Landing checklist

- Spike deploy first (gate) — record outcome + chosen diffusers pin here and in the Linear scope comment
- Tier README slot-table row → folder link; `llms.txt` line; root README tier count if stated
- Linear: SERV-1316 → In Progress + scope-note comment (model, 4×H100, sequence-parallel decision, source lives in this repo superseding fal-demos per CONTEXT.md)
- Pre-warm `/data` HF cache from dev deploys before any CI deploy-check runs
- PR: `feat(04-genmedia-patterns): multi-GPU video inference with fal.distributed (SERV-1316)`
- fal-docs page + card + cross-link/redirect fixes: separate later phase from current fal-docs main
