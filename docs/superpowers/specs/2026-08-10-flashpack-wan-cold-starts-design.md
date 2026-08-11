# Design: 02.1 · Cut cold starts by Nx — FlashPack + /data + compiled caches (WAN 2.1)

**Date**: 2026-08-10 · **Ticket**: [SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286) · **Gate**: [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271) (see "SERV-1271 evidence" below)

## Problem

Cold starts on GPU serverless are dominated by three costs paid on every worker boot: downloading weights from the Hub, deserializing safetensors through CPU onto the GPU, and `torch.compile` kernel compilation. fal has a lever for each — `/data` persistent storage, [FlashPack](https://github.com/fal-ai/flashpack) disk→GPU tensor streaming, and `fal.toolkit.synchronized_inductor_cache` — and this example demonstrates all three on one model with a measured before/after.

This is a **clean-room rework** of closed draft PR #5 (SD-Turbo), per owner direction on 2026-08-10: rebuild around the WAN 2.1 flagship example that the FlashPack repo itself showcases (`tests/test_wan_pipeline.py`), starting from `templates/example/`, re-deriving infrastructure only where needed. Knowledge of PR #5's landmines is retained (they are platform facts, not code): see "Known traps".

## Model

`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` — text-to-video, the exact model and integration pattern of FlashPack's flagship test.

- HF repo ships fp32: text encoder (UMT5-XXL) ~22.7 GB in 5 shards, transformer ~5.7 GB in 2 shards, VAE ~0.5 GB → **~29 GB download** on a baseline boot.
- Served dtypes (flagship's choices): transformer bf16, text encoder bf16, VAE fp32 → **~15 GB FlashPack pack**.
- The size asymmetry (29 GB fetched+cast vs 15 GB streamed) is what makes the before/after story strong.

## Approved decisions

1. **All three levers** (owner choice over FlashPack-only scope): baseline vs optimized apps, measured before/after.
2. **Clean-room rewrite** (owner choice): start from `templates/example/`, not PR #5's code.
3. **Structure A** (owner choice): two apps + shared `wan_flashpack.py` module shipped via `local_python_modules`.

## Files

```
02-scaling-cold-starts/cut-cold-starts/
├── README.md          # §4-A template order
├── app.py             # OPTIMIZED: /data pack + FlashPack load + shared inductor cache
├── app_baseline.py    # BASELINE: plain from_pretrained + torch.compile, no levers
├── wan_flashpack.py   # the 4 flagship classes, verbatim from FlashPack's WAN test
├── bench.py           # measured before/after via real ephemeral deploys
├── test_app.py        # runs against BOTH variants (harness enforces identical behavior)
└── requirements.txt   # mirrors App.requirements exactly
```

- Baseline is named `app_baseline.py` because `harness/check.py::discover_variants` globs `app*.py` and deploys each variant independently against the same `test_app.py` (the 01.1 `app_container.py` precedent). `wan_flashpack.py` deliberately does not match the glob.
- Tier README (`02-scaling-cold-starts/README.md`) row 02.1 gains the folder link; SERV-1271 stays noted until Rehan closes it.

## `wan_flashpack.py` — the flagship module

The four classes exactly as upstream defines them:

- `FlashPackWanTransformer3DModel(WanTransformer3DModel, FlashPackDiffusersModelMixin)` with `flashpack_ignore_prefixes = ["rope"]`
- `FlashPackAutoencoderKLWan(AutoencoderKLWan, FlashPackDiffusersModelMixin)`
- `FlashPackUMT5EncoderModel(UMT5EncoderModel, FlashPackTransformersModelMixin)` with `flashpack_ignore_names = ["encoder.embed_tokens.weight"]`
- `FlashPackWanPipeline(WanPipeline, FlashPackDiffusionPipeline)` — `__init__` signature matched to the **pinned** diffusers version (upstream's signature includes `transformer_2`/`boundary_ratio`/`expand_timesteps`; adjust to what the pin actually accepts)

Both apps set `local_python_modules = ["wan_flashpack"]`. This gives the classes a real importable module name so transformers' `sys.modules[cls.__module__]` introspection works on fal (the crash PR #5 hit with classes defined inside path-loaded app files).

## `app.py` (optimized) — `setup()` stages, each wall-clock timed

Constants: `PACK_DIR = /data/examples/cut-cold-starts/wan21-t2v-13b-bf16-v1`, `INDUCTOR_CACHE_KEY = "cut-cold-starts-wan21-13b/v1"`, `READY_MARKER` inside `PACK_DIR` carrying the inductor key in its name. Bump `v1` whenever model/dtype/pins change — stale packs and kernel caches must never be loaded across pins.

1. **imports** — deferred heavy imports, timed.
2. **download + pack (first boot per account only)** — if `PACK_DIR` missing: `snapshot_download(MODEL_ID)`, build components at serving dtypes via the flagship classes, assemble `FlashPackWanPipeline`, `save_pretrained_flashpack(<local tmp dir>)` — **local disk first**, then one bulk `copytree` to a `/data` temp dir and atomic `rename` to `PACK_DIR`. Loser of a concurrent-boot race cleans up; `rename` failures only swallowed when `PACK_DIR` exists afterwards; `.build-*` dirs older than 1 h swept.
3. **load weights (every boot)** — `FlashPackWanPipeline.from_pretrained_flashpack(str(PACK_DIR), device_map="cuda")`. Immediately after: meta-tensor materialization check over all params/buffers of transformer/vae/text_encoder — fail the boot loudly on pack/model mismatch, **before** any `.to()` masks it.
4. **compile + warmup** — inside `synchronized_inductor_cache(INDUCTOR_CACHE_KEY)`: `pipe.transformer = torch.compile(pipe.transformer, dynamic=True)`, then 1-step generations at the min and max allowed frame counts (17 and 81) at the fixed 832×480 resolution. Kernels sync to /data on context exit; later boots unpack and warmup is cache-hit execution.
5. **readiness marker** — written atomically (tmp + rename) only after steps 2–4 all succeeded this boot or previously. `built_caches_this_boot = not READY_MARKER.is_file()` decided at entry; a crash mid-setup leaves no marker so the next boot reports itself honestly.

App config: `machine_type = "GPU-H100"` (pinned — inductor caches are per GPU type), `keep_alive=60`, `startup_timeout` sized for the one-time first boot (29 GB download + ~15 GB pack publish + compile; ≥ 1800 s), `app_auth = "private"` (repo convention across all merged examples).

## `app_baseline.py` — no levers

`WanPipeline.from_pretrained(MODEL_ID, torch_dtype=bf16)` (VAE fp32 per flagship), `.to("cuda")`, then the **same** `torch.compile` + warmup but with no shared cache. Every boot pays download + CPU deserialization + cast + full compile. Both apps end in the identical serving state — the comparison isolates boot cost, not output quality. Same endpoints, same schema, same pins (identical `requirements` so both share one cached env and the benchmark isolates the levers).

## Endpoints (identical on both apps)

- `POST /` — `Input`: `prompt` (str), `num_frames` (int, default 81, constrained to 4k+1 in [17, 81]), `num_inference_steps` (int, default 28, [1, 40]), `seed` (int | None, echoed back), `negative_prompt` (str | None). Output: `video` (`fal.toolkit` CDN-hosted mp4, 832×480 @ 16 fps), `seed`, `generation_ms`. Resolution fixed at 832×480 (flagship's shape) so warmup covers all compiled shapes.
- `POST /setup-report` — `{variant, gpu, built_caches_this_boot, stages: [{name, seconds}], total_seconds}`. `bench.py` and the README table read from this; runner-state granularity via in-app stage timers (the REST API only exposes PENDING/RUNNING for ephemeral runners — PR #5 finding).

## Known traps (platform facts, carried as knowledge)

1. **Never chunk-write packs directly onto `/data`** — measured ~0.1 MB/s in July validation; pack locally, bulk-copy, atomic rename. Reads from `/data` are fine (~544 MB/s measured then).
2. **transformers 5.x mixin crash** — `sys.modules[cls.__module__]` KeyError for classes defined in path-loaded app files. Expected fix: `wan_flashpack.py` + `local_python_modules` (API confirmed present in fal 1.78.2: `fal/app.py:553`). **Spike validates before any real build.** Fallback: text encoder via direct `assign_from_file` onto a meta-device skeleton (PR #5's proven workaround), diffusers components keep the mixin path.
3. **Meta tensors ride through `.to()` unchanged** — materialization check must run immediately after FlashPack assignment.
4. **`save_pretrained_flashpack`** requires target dirs to pre-exist in some paths; `pack_to_file` has a required positional `target_dtype` (pass `None` to keep dtypes).

## Pins

`fal==1.78.2` (repo standard), `flashpack==0.4.0` (public wheel — **identical to repo main** except one CI-only commit, verified 2026-08-10 via `compare/v0.4.0...main`), plus diffusers / transformers / torch / accelerate pinned at spike-validated versions (starting candidates: the PR-#5-validated `diffusers==0.39.0`, `transformers==5.3.0`, `torch==2.12.1`, `accelerate==1.14.0`; the `WanPipeline` subclass `__init__` must match whatever diffusers pin lands). `requirements.txt` mirrors `App.requirements` exactly; both apps pin identically.

## Spike (before implementation proper)

1. **Mixin-on-fal spike**: tiny fal app (no model weights) that defines a trivial `FlashPackTransformersModelMixin` subclass in `wan_flashpack.py`, ships it via `local_python_modules`, and instantiates it in `setup()`. Pass → flagship pattern verbatim. Fail → `assign_from_file` fallback for the text encoder, documented in README.
2. **Signature check**: confirm the pinned diffusers `WanPipeline.__init__` signature and adjust the subclass; confirm `from_pretrained_flashpack(..., device_map="cuda")` kwargs against flashpack 0.4.0.

## Testing & benchmarking

- **CI (`harness/check.py`)**: discovers both variants, ephemerally deploys each, health-checks, smokes `POST /`, runs `test_app.py` against each live app. `test_app.py`: schema assertions + one tiny seeded generation (17 frames, 4 steps) asserting a video URL/data-URI comes back and the seed echoes. This satisfies the DoD's "validated deployment" (per the 2026-07-21 descope, "measured numbers via the harness" reads as CI deploy-check).
- **`bench.py`** (fresh, minimal): for each variant — ephemeral deploy, wait healthy (wall-clock spawn→healthy), `POST /setup-report`, teardown. Runs: baseline ×2, optimized first boot ×1 (cache build, one-time), optimized steady state ×2. Prints the markdown table the README embeds. No leaked runners (verify list before/after).
- **README numbers**: only from real dated `bench.py` runs, with GPU/torch/date stated. Never hardcoded or extrapolated (CONTEXT.md fact #2).

## README (§4-A order)

Problem → sample output (video still/gif + response JSON) → Run it → How it works (the three levers, each mapped to its `setup()` stage and timed row; link FlashPack repo since no fal concept doc exists yet) → Deploy → Measured before/after table + measurement details → Cost & tuning (H100 choice, keep_alive, first-boot economics, when NOT to bother: small models/rare deploys) → Next steps (02.2 autoscaling, 02.3 20+ GB models, FlashPack repo).

## Deliverables & follow-ups

1. Example folder as above + tier README row update, on branch `claude/flashpack-example-2-1-fc7cdc`, PR to `main` linking SERV-1286.
2. Findings comment on SERV-1271: wheel==main parity evidence, WAN flagship pattern working (or not) on fal via `local_python_modules`, any new gaps — Rehan closes the decision ticket himself.
3. Out of scope here: the fal-docs page (separate repo, IA gated by SERV-1266) and the FlashPack concept doc — both remain SERV-1286 docs-side follow-ups.

## Risks

| Risk | Mitigation |
|---|---|
| Mixin crash reproduces even via `local_python_modules` | Spike first; `assign_from_file` fallback keeps the example shippable |
| WAN generation makes CI smoke slow (video is minutes, not seconds) | Smoke uses 17 frames / 4 steps seeded (~tens of seconds of transformer time on H100); `test_app.py` timeout headroom checked against harness `TEST_TIMEOUT_S` |
| First optimized boot is very long (download+pack+compile) | `startup_timeout ≥ 1800 s`; README frames it as one-time-per-account; readiness marker keeps reporting honest |
| Baseline boot variance (Hub throughput, host) | bench runs baseline ×2 and publishes both numbers + date, like PR #5 did |
| dynamic=True still recompiles at unseen frame counts | Warmup at both frame-count extremes; frames constrained to [17, 81] 4k+1 |
