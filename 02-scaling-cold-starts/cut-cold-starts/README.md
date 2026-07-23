# Cut cold starts by Nx

Your model API is fast once it's warm — but the first request after scale-up eats a cold start: pull the image, download weights, deserialize them onto the GPU, and (if you use `torch.compile`) recompile every kernel. This example takes one app (SD-Turbo) and pulls fal's three cold-start levers — **weights on `/data` persistent storage**, **FlashPack tensor loading**, and **shared `torch.compile` caches** — with the before/after measured on real deployments.

> Catalog slot `02.1` ([SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286)). Two apps, same endpoint: [`baseline_app.py`](baseline_app.py) pays full price every boot; [`app.py`](app.py) pays once per account, then boots from caches. [`bench.py`](bench.py) produces the table below from live runs.

## What you get

The same endpoint, before and after — measured on real ephemeral deploys (`GPU-H100`):

<!-- BENCH_TABLE_START -->
| Setup stage | Baseline (2 runs) | Optimized, steady state (3 runs) | Optimized, first boot (one-time) |
|---|---|---|---|
| imports | 10.5s / 20.3s | 10.0s / 12.3s / 75.8s | 11.0s |
| download weights | 10.8s / 9.3s | 0 (weights on `/data`) | included in pack stage ↓ |
| weights → GPU | 1.3s / 1.9s (safetensors) | 2.6s / 8.0s / 571.5s (FlashPack, see note) | 2.8s |
| download + pack + publish to `/data` | — | — | 7.0s |
| compile + warmup | 328.9s / 383.5s | **73.2s / 64.7s / 87.6s** (cache hit) | 359.4s (compiles, syncs cache) |
| **`setup()` total** | **351.6s / 414.9s** | **85.8s / 648.5s / 171.4s** | 380.2s |
| spawn → healthy | 357.9s / 435.1s | 92.6s / 655.4s / 211.7s | 388.4s |
| first request after boot | 1.55s / 1.96s | 1.54s / 1.71s / 1.87s | 1.74s |

**Typically 2–4.8× faster cold starts**, measured 2026-07-23 on real fal ephemeral deploys (`GPU-H100`, torch 2.12.1, identical environments — see [`bench.py`](bench.py)); every run is published, including the ugly one. The consistent win is the compile cache: **65–88s on every cache-hit boot vs 329–384s of recompilation on every baseline boot** — the dominant cold-start term for this model, cut ~4–5× reliably. The weight path is normally 2.6–8.0s for 2.6 GB, but one steady-state boot landed on a host with degraded `/data` read throughput (571.5s for the same read, ~4.5 MB/s) and totaled 648.5s — host storage variance is the real tail risk, and it would hit a `/data`-cached safetensors load just as hard. Notes on the first-boot column: the HF fetch is not timed separately — it is folded into the 7.0s "download + pack + publish" stage, and on our measured boot it was served by an already-warm platform-side cache; a cold cache adds the full model download to that stage. The runners' REST state timeline exposed only `PENDING`/`RUNNING` for ephemeral runners, so the per-stage split above comes from each app's own `/setup-report` endpoint.
<!-- BENCH_TABLE_END -->

…and a typed text-to-image API that answers in well under a second once warm:

```json
{
  "image": { "url": "https://v3.fal.media/files/…/output.png", "width": 512, "height": 512 },
  "seed": 42,
  "generation_ms": 87
}
```

Each app also exposes `POST /setup-report`, which returns what *that worker's* cold start spent its time on, stage by stage — the numbers above come straight from it.

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py            # optimized app; first-ever run builds the /data caches
fal run baseline_app.py   # baseline, for comparison
```

To reproduce the table with real deployments (bills real GPU minutes):

```bash
python bench.py           # deploys each app twice, prints the before/after table
```

`bench.py` labels each optimized run by what the worker actually did (the app's `built_caches_this_boot` flag): the one-time cache build only happens on an account whose `/data` caches don't exist yet — once they do, every run is a steady-state boot. To reproduce the first-boot measurement, bump the version in `app.py`'s `PACK_DIR`/`INDUCTOR_CACHE_KEY` (or measure on a fresh account).

## How it works

Both apps serve identical SD-Turbo endpoints; the whole difference is `setup()`.

**Lever 1 — weights on [`/data` persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage).** The baseline downloads ~2.6 GB into container-local disk on *every* boot — the container is gone when the worker dies, so the next cold start downloads it all again. The optimized app touches the network once per *account*: the first boot downloads into the `/data`-backed HF cache and repacks the weights next to it; every later cold start (any worker, any deploy, CI included) reads from `/data` and skips the download entirely.

**Lever 2 — [FlashPack](https://github.com/fal-ai/flashpack) weight loading.** Safetensors deserialization reads tensors on the CPU and copies them one by one onto the GPU. FlashPack stores the state dict in a layout built for streaming — the optimized app loads the UNet and VAE through small mixin subclasses (`from_pretrained_flashpack(...)`), and the text encoder through the direct function API (`assign_from_file(...)`): the module skeleton is built on the meta device, then tensors stream from `/data` straight onto the GPU (2.6 GB in ~2.6 s measured). Two hard-won details, both explained in comments in [`app.py`](app.py):

- **Pack locally, then copy to `/data`.** `pack_to_file(...)` writes chunked and aligned, a pattern that is pathologically slow directly against `/data`'s network filesystem (measured ~0.1 MB/s — a pack that never finished — vs ~1 GB/s to local disk). The first boot packs to container-local disk, bulk-copies to a temp dir on `/data`, and atomically renames it so concurrently booting workers never see a half-written pack.
- **Why the text encoder skips the mixin.** transformers 5.x inspects the source module of a model's class at `__init__` time; a subclass defined inside a fal app file — which the platform loads by path, without an importable module name — crashes that lookup. The direct function API does the same work without a subclass.

**Lever 3 — compiled caches via [`synchronized_inductor_cache`](https://fal.ai/docs/serverless/optimizations/optimize-startup-with-compiled-caches).** `torch.compile` makes SD-Turbo's UNet meaningfully faster per step, but costs minutes of kernel compilation — which the baseline pays on every single boot. The optimized app compiles inside fal's cache context:

```python
with synchronized_inductor_cache("cut-cold-starts-sd-turbo/v1"):
    self.pipe.unet = torch.compile(self.pipe.unet, dynamic=True)
    self.warmup()   # compilation happens HERE - warmup must stay inside the block
```

The first boot compiles for real and syncs the kernels to `/data` on exit; every later boot unpacks them and warmup is just cache-hit execution. Caches are stored per GPU type — which is one reason `machine_type` pins a single type (`"GPU-H100"`) instead of a fallback list; the other is that a benchmark across mixed GPUs wouldn't be comparing the levers.

**Everything is versioned.** The pack dir (`…/sd-turbo-fp16-v1`) and cache key (`…/v1`) both carry a version. Compiled kernels are torch-version-specific, and a stale pack silently loaded by a new pin is a debugging nightmare — bump the version whenever the model, dtype, or torch pin changes.

**What's deliberately identical** so the benchmark isolates the levers: both apps pin the same requirements (one shared environment), the same machine type, the same fp16 payload, and the same compile flags.

## Deploy

```bash
fal deploy app.py
```

This registers the optimized app and prints its playground and API URLs. Call it:

```python
import fal_client

result = fal_client.subscribe(
    "<your-username>/cut-cold-starts",
    arguments={"prompt": "a calico cat riding a bicycle, film photo", "seed": 42},
)
print(result["image"]["url"], f"({result['generation_ms']} ms)")
```

There's no reason to deploy the baseline except to measure it — that's `bench.py`'s job.

## Cost & tuning

- **Cold starts are a cost multiplier on autoscaling.** Every second of `setup()` is billed GPU time on *each* new worker — and slow setups mean scale-up lags your traffic. Cutting setup by Nx cuts both.
- **The one-time build is the price of admission**: the optimized app's first boot pays download + pack + compile. After that it's amortized across every worker forever. If you change GPU type, the inductor cache rebuilds once for the new type (packs on `/data` are GPU-agnostic).
- **`/data` throughput varies by host.** The same 2.6 GB FlashPack read measured 2.6s, 8.0s and 571.5s across three boots. The caches make the *typical* cold start cheap; if your p99 cold start matters more than idle cost, `min_concurrency > 0` (always-warm workers) is the only hard guarantee.
- `keep_alive=60` holds a warm worker for a minute after the last request — cold starts only matter for traffic that outruns your warm pool. Pair this example with autoscaler tuning (example 02.2) for the full picture.
- `startup_timeout=1200` exists for the first-ever boot; steady-state boots come in far under it.

## Next steps

- **02.2 — Tune autoscaling for bursty traffic**: `keep_alive`, concurrency and the cost math that decides how often you cold-start at all.
- **02.3 — Serve a 20+ GB video model**: the same levers at a scale where they're not optional.
- Concept docs: [optimize cold starts](https://fal.ai/docs/serverless/deployment-operations/optimize-cold-starts) · [compiled caches](https://fal.ai/docs/serverless/optimizations/optimize-startup-with-compiled-caches) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) · [understand runners](https://fal.ai/docs/serverless/deployment-operations/understand-runners)
