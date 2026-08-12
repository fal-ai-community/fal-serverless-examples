# Cut cold starts by Nx: FlashPack + /data + compiled caches

Every GPU serverless cold start pays three costs before a model can serve a single request: pulling weights off the Hub, deserializing them through CPU into GPU memory, and compiling kernels for the shapes it will run. On this example's model (WAN 2.1 text-to-video), the third cost dominates — `torch.compile` alone eats roughly three-quarters of a from-scratch boot, dwarfing the download. Anyone who scales to zero between bursts, autoscales under bursty traffic, or just doesn't want every fresh worker to re-pay that tax hits this problem. This example deploys the same model twice — once with every cold-start lever pulled, once with none — so the before/after is a real measured deploy, not a claim.

> Catalog slot `02.1` ([SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286)). Model: [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).

## What you get

Measured 2026-08-12 (UTC) on GPU-H100: `setup()` drops from ~275 s on every single cold start (baseline) to a ~96–133 s steady state after a one-time first boot (optimized) — a **2.1–2.9× reduction**, driven almost entirely by not recompiling kernels from scratch. Full table and how that number is derived: [Measured cold starts](#measured-cold-starts).

![Sample output: calico cat surfing, 17 frames @ 16 fps](assets/sample-output.gif)

This is the exact output of the smoke request below (`num_frames=17`, `num_inference_steps=8`, `seed=42` — the same seed reproduces this exact video); the full-resolution still is in [`assets/sample-output.jpg`](assets/sample-output.jpg).

A real smoke response from a live deploy of `app.py` (`prompt` + `seed` from the endpoint's `Field` examples, `num_frames=17`, `num_inference_steps=8` — the fast/cheap shape used for CI and local checks):

```json
{
  "video": {
    "url": "https://v3b.fal.media/files/b/0aa5fdda/dvvzDUMrGuV76CNzhfubf_video.mp4",
    "content_type": "application/octet-stream",
    "file_name": "video.mp4",
    "file_size": 122630
  },
  "seed": 42,
  "generation_ms": …
}
```

Quoted verbatim from `.superpowers/sdd/bench-runs/attempt5-step5.log` (the "smoke response:" line) — the log line is truncated right after `"generation_ms"`, so that field is elided above rather than invented; per `TextToVideoOutput`, it always reports server-side model time in milliseconds (denoise + decode, excluding upload). The `url` is a real fal CDN link from that run, but fal CDN files have limited retention — treat it as illustrative, not a live asset; don't expect it to resolve.

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py        # optimized: /data + FlashPack + compiled caches
```

`fal run app_baseline.py` serves the identical schema and model the slow way — no cold-start lever pulled — so you can feel the difference between the two `setup()` times directly instead of just reading the table below.

## How it works

**`setup()` — three levers, each mapped to a row in the [measured table](#measured-cold-starts):**

**Lever 1 — `/data` persistent storage (once per account).** The Hub repo ships ~29 GB of fp32 shards. The very first boot for a given `PACK_VERSION` downloads them, rebuilds the pipeline at serving dtypes (bf16 transformer + text encoder, fp32 VAE — the same dtypes the baseline casts to, so both apps serve identical pipelines), and packs it with FlashPack — that's the `download + build FlashPack pack (first boot only)` stage (179.79 s in the table). The pack is written to **local disk first**, then bulk-copied to a `/data` temp directory and atomically renamed into place — that's `publish pack to /data (first boot only)` (27.92 s). Chunked writes straight to `/data` measured ~0.1 MB/s in a July 2026 validation (reads are fine, ~544 MB/s then), and the atomic rename means concurrently booting workers can never observe a half-written pack. `PACK_DIR` is versioned (`PACK_VERSION = "v1"`) — bump it whenever `MODEL_ID`, serving dtypes, or the torch pin change, so a new pin can never load an old pack or old kernels. Every boot after the first, for any worker, on any later deploy, skips this lever entirely and goes straight to lever 2.

**Lever 2 — FlashPack loading (every boot).** `FlashPackWanPipeline.from_pretrained_flashpack(str(PACK_DIR), device_map="cuda", ...)` streams tensors `/data → GPU` directly, skipping safetensors' CPU deserialize-then-move step. That's the `load weights (FlashPack, /data -> GPU)` row: 6.78 s on the first boot (pack just built, still local-ish) and 15.45 s / 7.66 s on steady-state boots, against the baseline's `download + load weights (Hub, safetensors -> CPU -> GPU)` at 46.09 s / 41.48 s — 2.7–6.0× faster across the steady-state runs (41.5–46.1 s → 7.7–15.5 s). One integration wrinkle lives here: transformers ties `UMT5EncoderModel`'s `encoder.embed_tokens.weight` to `shared.weight`, and FlashPack's loader assigns packed tensors by *replacing* each `Parameter` object, which silently breaks that tie (the pack deliberately omits the tied name via `flashpack_ignore_names`). `setup()`'s load stage works around it: it loads the text encoder itself, re-ties it with `set_input_embeddings`, and passes the finished module into the pipeline loader as a kwarg — a component supplied that way is used as-is and skips the broken path. See the comment block starting at `text_encoder = classes["FlashPackUMT5EncoderModel"].from_pretrained_flashpack(...)` in [`app.py`](app.py)'s load stage, and the guard right after it that fails loudly on any leftover meta tensor instead of crashing obscurely later.

**Lever 3 — `synchronized_inductor_cache` (compiled kernels).** `torch.compile(self.pipe.transformer, dynamic=True)` runs inside `fal.toolkit.synchronized_inductor_cache(INDUCTOR_CACHE_KEY)`: the first boot compiles for real and syncs kernels to `/data` on context exit (`compile + warmup (shared inductor cache)`, 198.75 s — nearly identical to the baseline's uncached 202.9 s / 208.19 s, because it's doing the same work); every later boot unpacks the cached kernels instead of recompiling, cutting compile+warmup by more than half (202.9–208.2 s → 77.8–90.6 s). `machine_type = "GPU-H100"` is pinned to exactly one type, not a fallback list, because inductor caches are per GPU type — a cache built on an H100 doesn't help an L40 boot.

**`wan_flashpack.py`.** The four classes it defines (`FlashPackWanTransformer3DModel`, `FlashPackAutoencoderKLWan`, `FlashPackUMT5EncoderModel`, `FlashPackWanPipeline`) are the verbatim flagship pattern from FlashPack's own WAN test — [`tests/test_wan_pipeline.py`](https://github.com/fal-ai/flashpack/blob/main/tests/test_wan_pipeline.py) in [github.com/fal-ai/flashpack](https://github.com/fal-ai/flashpack) — not a fal-specific rewrite. Getting that pattern to run on fal surfaces two integration wrinkles the docs don't cover yet (no fal concept doc exists for FlashPack; SERV-1286 tracks the docs-side follow-up):

1. **The module needs a real `sys.modules` entry.** This example ships `wan_flashpack.py` to the runner via `local_python_modules` (cloudpickle by-value), which reconstructs the module *without* a `sys.modules` entry. Two consumers need one anyway: transformers reads `sys.modules[cls.__module__]` when instantiating any `PreTrainedModel` subclass, and diffusers' pipeline loader resolves component classes via `importlib.import_module("wan_flashpack")`. `_publish_classes()` installs the entry at runtime, reusing it if the module was imported normally (local runs, tests) — see the docstring and function at the top of [`wan_flashpack.py`](wan_flashpack.py).
2. **diffusers strips the "FlashPack" prefix on load.** diffusers' `get_class_obj_and_candidates()` (pipeline_loading_utils.py, diffusers 0.39) strips a `"FlashPack"` prefix from the class names recorded in the pack's `model_index.json` before resolving them — `"FlashPackUMT5EncoderModel"` gets looked up as `"UMT5EncoderModel"`. The registration loop at the bottom of [`wan_flashpack.py`](wan_flashpack.py)'s `get_classes()` registers every class under **both** names so either resolution path lands on the FlashPack-mixed-in subclass, not diffusers' plain upstream class.

**The readiness marker is an honesty mechanism, not an optimization.** `READY_MARKER` is written — atomically, via a temp file + rename — only *after* both the `/data` pack and the synced kernel cache exist. A worker that crashes mid-`setup()` leaves no marker, so the next boot finds `PACK_DIR` (or the cache) incomplete and honestly rebuilds rather than trusting a half-finished cache. `built_caches_this_boot` (surfaced on `/setup-report`) reports exactly this: `True` only when the boot itself did the building.

## Deploy

```bash
fal deploy app.py
```

This registers `cut-cold-starts` and prints its playground and API URLs.

```bash
curl -X POST https://queue.fal.run/<your-username>/cut-cold-starts \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "A calico cat surfing a small wave at golden hour, cinematic", "num_frames": 17, "num_inference_steps": 8, "seed": 42}'
```

First-boot economics: the first deploy ever made under your account for a given `PACK_VERSION` pays the full one-time cost — 520.3 s spawn→healthy in the measured run below, covered by `startup_timeout=2700`. Every deploy after that — new worker, restarted worker, redeploy — finds both caches warm and gets the steady state (~96–133 s `setup()`, 114.7–267.3 s spawn→healthy). The baseline never gets a steady state: every single boot pays ~275 s, forever, because it builds nothing to reuse.

## Measured cold starts

**Measured 2026-08-12 (UTC) · GPU-H100 · fal 1.78.2 · torch 2.12.1 · diffusers 0.39.0 · transformers 5.3.0 · flashpack 0.4.0**

| stage | baseline run 1 | baseline run 2 | optimized first boot (one-time) | optimized run 1 (steady) | optimized run 2 (steady) |
|---|---|---|---|---|---|
| imports | 26.26 | 26.26 | 54.61 | 26.09 | 11.0 |
| download + load weights (Hub, safetensors -> CPU -> GPU) | 46.09 | 41.48 | - | - | - |
| download + build FlashPack pack (first boot only) | - | - | 179.79 | - | - |
| publish pack to /data (first boot only) | - | - | 27.92 | - | - |
| load weights (FlashPack, /data -> GPU) | - | - | 6.78 | 15.45 | 7.66 |
| compile + warmup (no cache) | 202.9 | 208.19 | - | - | - |
| compile + warmup (shared inductor cache) | - | - | 198.75 | 90.56 | 77.79 |
| setup() total | 275.25 | 275.93 | 467.95 | 132.58 | 96.46 |
| spawn -> healthy | 302.1 | 303.9 | 520.3 | 267.3 | 114.7 |

Reproduce with `bench.py` (needs fal credentials configured):

```bash
HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app_baseline --runs 2
HARNESS_STARTUP_TIMEOUT=2700 python 02-scaling-cold-starts/cut-cold-starts/bench.py --variant app --runs 3
```

**Headline multiplier, computed from `spawn -> healthy` medians:** baseline median 303.0 s, optimized steady-state median 191.0 s → **≈1.6×**. That number is smaller than the `setup()`-only figure above (2.1–2.9×) because `spawn -> healthy` also bills GPU scheduling and allocation wait time, which is noisy: the two steady-state runs measured 267.3 s and 114.7 s respectively — a real spread caused by GPU-scheduling/allocation contention on a shared account (15 "waiting for allocation" retries during the slower run vs. 1 during the faster one), not by any difference in model work. Both numbers are published rather than averaged away; `setup()` total is the more honest signal for what this example's levers actually control, since it excludes platform scheduling noise.

**Other disclosures that travel with this table:**

- **Warm HF cache.** `HF_HOME=/data/.cache/huggingface` was already warm on the account these numbers were measured from (the model's weights were already resident). Every "download" figure above — the first-boot column's pack build (179.79 s) and the baseline's `download + load weights` (46.09 s / 41.48 s) — reflects fal's platform-warm, `/data`-backed HF cache, not a genuinely cold fetch over the network. A first-ever account would see these specific cells meaningfully higher; nothing else in the table is affected.
- **First boot is one-time per account per pack version.** The 520.3 s first-boot row happens once — the next time `PACK_VERSION` changes, or on a brand-new account, it happens again. Every other cold start any account ever pays is the steady-state row.
- **Stage granularity comes from in-app timers, not the platform.** Ephemeral runners are only visible over REST as scheduling states (e.g. "waiting for allocation") plus a coarse running/healthy signal — there's no platform-side breakdown of what a boot spent its time on. Every stage row above comes from timers inside `setup()` itself (the `stage()` helper, surfaced via `/setup-report`), which is also why `bench.py` calls that endpoint after every run instead of trying to parse platform logs.

## Cost & tuning

- **You pay for every second of that table.** Both apps bill from `setup()` start through teardown, same as any fal app — a cold start isn't free just because no request has landed yet. Under bursty traffic, the difference between paying ~275 s vs. ~96–133 s on every scale-from-zero event compounds fast; see [fal pricing](https://fal.ai/pricing).
- **`keep_alive=60`** (both apps) holds a warm worker for 60 seconds after the last request — enough to absorb back-to-back requests in a burst without a second cold start. Raise it if your traffic pattern has gaps longer than that but you still want to avoid cold starts between them; this example doesn't tune autoscaling further, that's 02.2.
- **`startup_timeout=2700`** on the optimized app is sized for the worst case it will ever see once — the account's first-ever boot (download + pack build + compile), measured at 520.3 s here with real headroom to spare. The baseline sets the same value even though it never builds a cache, because its steady-state boot (~275 s, every time) still needs generous headroom under contention.
- **Bump `PACK_VERSION`** whenever `MODEL_ID`, serving dtypes, or the torch/diffusers/flashpack pins change. A stale pack or kernel cache built under old pins must never be loaded under new ones — bumping the version string is what forces a fresh first boot instead of a silent mismatch.
- **When not to bother with any of this:** a model with a short or no `torch.compile` step (see 00.2's `first-image-api`, which skips compile entirely — nothing here to cache), an app that rarely scales from zero (a high `keep_alive` or `min_concurrency` already solves the problem more simply than any of these three levers), or a model small enough that a cold cold-start is already a few seconds.
- **Smoke/CI knobs:** `num_frames=17, num_inference_steps=8` (this README's `curl` example and `test_app.py`'s `SMALL_REQUEST`) generate a video fast and cheaply enough for CI's live smoke test without touching the cold-start levers at all — they only affect `generate()`, not `setup()`.

## Next steps

- **02.2 — Tune autoscaling for bursty traffic**: `keep_alive`, concurrency, and the cost math this example's fixed `keep_alive=60` only gestures at. See the [tier README](../README.md) for the folder once it lands.
- **02.3 — Serve a 20+ GB video model**: parallel loading and GPU-sizing reasoning for models too big for this example's single-GPU FlashPack pattern. See the [tier README](../README.md) for the folder once it lands.
- [FlashPack](https://github.com/fal-ai/flashpack) — the disk→GPU streaming library this example builds on.
- Concept docs: [application setup](https://fal.ai/docs/documentation/development/app-setup) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) · [machine types](https://fal.ai/docs/documentation/deployment/machine-types) · [optimize cold starts](https://fal.ai/docs/documentation/serverless/optimizations/optimize-cold-starts)
