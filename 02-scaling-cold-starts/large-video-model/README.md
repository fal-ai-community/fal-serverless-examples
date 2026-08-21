# Serve a 20+ GB video model — and watch it generate

Two things go wrong when you put a frontier video model behind an API. The weights are bigger than most GPUs and take minutes to reach the machine, so your first request pays for all of it. And even once the model is warm, generation is a long silence — the caller stares at a spinner while seventy gigabytes of weights do their work. This example serves **LTX-2.3 Distilled** on a single H200 and takes both seriously: `loader.py` gets ~88 GB of weights onto the GPU in about 40 s once they are cached — with the slow tail measured and written down rather than hidden — the GPU choice is arithmetic rather than a guess, and `/stream` sends a playable video of the *in-progress* generation after every denoising step.

> Model: [`diffusers/LTX-2.3-Distilled-Diffusers`](https://huggingface.co/diffusers/LTX-2.3-Distilled-Diffusers) — a 22B-parameter DiT plus a 12B text encoder, **ungated**, so nothing here needs a Hugging Face token. It is **not** Apache-2.0; read [Licensing](#licensing) before you ship it.

## What you get

![Six tiny-VAE previews from one 1024x576, 97-frame generation, sharpening step by step, followed by the finished clip](viewer-demo.gif)

*One request to `/stream` at 1024×576 × 97 frames. Every frame above came off the wire: the six `preview` events in order — each a complete little mp4 of the whole clip, decoded from half-finished latents — then the `result` video from the full VAE. Nothing is staged; the whole request took 10.5 s, and the first preview landed 3.4 s in.*

[`viewer.html`](viewer.html), pointed at the deployed app, shows the generation as it happens: each preview is a complete little mp4 of every frame in the clip, decoded from the half-finished latents while the model is still denoising them. The last swap is the real thing — full VAE, full resolution, with audio.

The plain endpoint is an ordinary typed JSON API. Request:

```json
{
  "prompt": "a red fox trotting through fresh snow at golden hour",
  "resolution": "768x512",
  "num_frames": 57,
  "seed": 42
}
```

Response:

```json
{
  "video": {
    "url": "https://v3b.fal.media/files/b/…/output.mp4",
    "content_type": "video/mp4",
    "file_name": "output.mp4",
    "file_size": 311365
  },
  "seed": 42,
  "num_frames": 57,
  "timings": { "generate_s": 2.9, "encode_s": 1.0 }
}
```

`seed` is echoed back whether you supplied it or not, `num_frames` is the frame count actually generated (after snapping to the model's 8k+1 grid), and `timings` carries the server-side stage breakdown — `generate_s` for the denoise loop, `encode_s` for the mp4 + audio mux. The `content_type` is set explicitly: `File.from_path` defaults to `application/octet-stream`, which makes a browser download the clip rather than play it.

## Run it

```bash
cd 02-scaling-cold-starts/large-video-model   # app.py imports the sibling loader module
pip install fal==1.78.2                       # keep pinned to this example's requirements.txt
fal auth login
fal run app.py                                # ephemeral app on a real H200; prints the app URL
```

The first-ever run downloads ~88 GB into your account's `/data` cache — budget several minutes and leave it alone. Every boot after that skips the network entirely. `fal run` holds the app open until Ctrl-C.

Then watch a generation arrive, from a second terminal:

```bash
pip install fal-client==1.0.0
export FAL_KEY=...
python client.py <app-id> "a red fox trotting through fresh snow at golden hour"
```

`<app-id>` is whatever `fal run` printed (`<owner>/<ephemeral-id>`), or `<owner>/large-video-model` once you have deployed. [`client.py`](client.py) rewrites `preview_latest.mp4` as each preview lands — open it in a looping player and it updates in place — then downloads `final.mp4` when the run completes. For the browser version, open [`viewer.html`](viewer.html) and paste in the app URL and your key. (It is a local demo page: a real frontend proxies the key through your backend instead of holding it.)

## How it works

### `loader.py` — 88 GB onto the GPU, in order

Weight loading is three separate problems, and [`loader.py`](loader.py) times each one so you can see which is hurting.

**Tune the transport before importing the hub.** `tune_hf_env()` sets `HF_XET_HIGH_PERFORMANCE`, `HF_XET_NUM_CONCURRENT_RANGE_GETS=32`, a 1 TB Xet chunk cache (it lives on `/data`, which has the room), and `HF_HUB_ENABLE_HF_TRANSFER` for non-Xet repos. These are read when `huggingface_hub` initializes, so the call has to happen first — that ordering is the whole reason it is a separate function.

**Download once per account, then never again.** `fetch_model()` pins the repo by revision and pulls the snapshot into `/data` with `max_workers=32`. `/data` is [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) shared across every worker and every CI deploy, so the ~88 GB transfer happens once for your whole account. The warm path matters as much as the cold one: `fetch_model()` *first* tries `snapshot_download(..., local_files_only=True)`, because an online-but-cached call still round-trips the hub to re-validate every file before concluding nothing changed. Offline, the same call returns in about a second.

**Pre-read the files before `from_pretrained` touches them.** fal's `/data` is content-addressed in 4 MB chunks spread across storage tiers, which means one sequential reader leaves most of the available bandwidth idle. `warm_page_cache()` is a deliberately blunt `find | xargs -P 32 cat > /dev/null`: 32 readers pull from many chunk servers at once and land everything in the OS page cache, so `LTX2Pipeline.from_pretrained(...)` reads from RAM. See fal's [parallel file loading](https://fal.ai/docs/documentation/serverless/optimizations/parallel-file-loading) guide for the mechanism.

### GPU sizing: why an H200, decided on paper

This is the part worth internalizing, because it generalizes to every large model you will ever deploy. Add up the resident weights in the precision you plan to run:

| Component | bf16 |
|---|---|
| DiT (22B) | ~44 GB |
| Gemma-3-12B text encoder | ~24 GB |
| Video VAE + audio VAE + vocoder | ~2 GB |
| **Resident total, before a single activation** | **~70 GB** |

Measured after load: **65.9 GB**. Now put that against the [machine types](https://fal.ai/docs/documentation/deployment/machine-types) and ask not "does it fit" but "what is left over":

| Machine type | VRAM | Verdict |
|---|---|---|
| `GPU-A100` | 40 GB | No — the weights alone do not fit |
| `GPU-L40` | 48 GB | No |
| `GPU-H100` | 80 GB | Weights fit with ~14 GB to spare — and peak at the *small* shape is already 68.5 GB. A knife-edge, not a plan; untested here |
| `GPU-RTXPRO6000` | 96 GB | Tight: fine at these shapes, no room to grow |
| `GPU-H200` | 141 GB | **What this example uses** — 68.5 GB peak at 768×512×57f, 71.9 GB at 1024×576×97f, on a 139.8 GB device |
| `GPU-B200` | 192 GB | More headroom than this model needs, at a higher rate |

Activations are the reason the answer is not "H100". Video attention scales with frames × height × width, so the gap between resident weights and peak VRAM is not a constant you can size once — it grows with every knob your callers can turn. Size for the largest shape your API accepts, then leave room for fragmentation.

### Previews: extrapolate, decode small, encode off the hot path

A raw latent at step 3 of 8 looks like noise, so streaming it would be pointless. Three things turn intermediate state into something worth watching, all in `_PreviewSession` in [`app.py`](app.py):

**x0 extrapolation.** Flow-matching latents travel nearly straight lines toward the clean sample, so two consecutive states and their sigmas are enough to guess the destination: `x0 = latents - sigma * (latents - prev_latents) / (sigma - prev_sigma)`. That estimate — not the raw latent — is what gets decoded, which is why previews look like a soft version of the final video instead of static.

**A tiny VAE instead of the real one.** Decoding with the production VAE would cost more than the generation. [`taehv.py`](taehv.py) is vendored from [madebyollin/taehv](https://github.com/madebyollin/taehv) (MIT), a distilled video autoencoder that takes diffusion-space latents directly — no `_denormalize_latents`, which saturates its input clamp. Decoding all 57 frames takes **0.033 s**. Previews are then downsampled to half resolution, which shrinks each event by ~4× for no perceptual loss at preview size.

**The encode runs on a worker thread.** The surprise is that the tiny VAE is not the expensive part: muxing those frames to mp4 through x264 costs **0.15–0.20 s** of CPU, four to six times the decode. So `on_step_end` never blocks the denoise loop — it stores the newest `x0` under a lock, dropping any predecessor that has not been decoded yet (latest-wins), and returns. One worker thread drains that slot, renders, and pushes an event onto a queue; the async SSE generator drains the queue and writes it to the wire. When generation ends, the generator freezes the queue and drains it once more, so a preview that was still encoding at the finish line still reaches the client — and nothing ever arrives after the terminal event.

At the default `preview_every=1` you get **6 preview events**, steps 1 through 6: step 0 has no predecessor to extrapolate from, and the last step is left to the real result. Each event's `video.url` is a self-contained `data:video/mp4;base64,…` URI — **41–208 KB** at the 768×512 smoke shape, **94–243 KB** at 1024×576×97f, climbing step by step as the picture sharpens and stops compressing so kindly — so a client can assign it straight to `video.src`. The `video` object carries the same `{url, content_type, file_size}` shape as the final `result` event, so one parsing path handles previews and the finished clip alike.

The final video is not a preview: it comes out of the full VAE at full resolution, with audio from the vocoder, and is uploaded to the fal CDN.

## Deploy

```bash
fal deploy app.py       # → <your-username>/large-video-model
```

The app is `app_auth = "private"`, so every call carries your key. The streaming endpoint mirrors [`client.py`](client.py):

```python
import fal_client

for event in fal_client.stream(
    "<your-username>/large-video-model",
    {"prompt": "a red fox trotting through fresh snow at golden hour",
     "resolution": "1024x576", "num_frames": 97},
    path="/stream",
):
    print(event["type"])
```

**The event schema.** Every SSE frame is one JSON object with a `type`:

| `type` | Fields | When |
|---|---|---|
| `preview` | `step`, `total_steps`, `progress`, `video` (`{url: "data:video/mp4;base64,…", content_type, file_size}`), `width`, `height`, `frames` | after each sampled denoising step |
| `result` | `video` (`{url: "https://…", content_type, …}`), `seed`, `num_frames`, `timings` | once, after the full-VAE decode and CDN upload |
| `error` | `detail`, `status_code` (422 for the out-of-memory guard, 500 otherwise) | instead of `result`, if generation or upload fails |

Exactly one terminal event ends the stream — `result` or `error`, never a trailing preview. That guarantee exists because the response sets `x-fal-sse-last-event-as-body: 1`: a caller that POSTs to `/stream` without reading incrementally receives the *last* event as the entire response body, so the last event has to be the one worth having. Handle all three types; `client.py` and `viewer.html` both do.

**`/stream` is synchronous.** It goes through the `fal.run` gateway, which holds the connection open for the whole generation — not the queue. For fire-and-forget or batch work, call `/` through the queue API instead:

```bash
curl -X POST https://queue.fal.run/<your-username>/large-video-model \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a red fox trotting through fresh snow at golden hour"}'
# → {"request_id": "...", "status_url": "...", ...} — poll the status_url, then fetch the result
```

## Cost & tuning

**Measured 2026-08-11 on `GPU-H200`.** The generate and warm-boot rows come from the registered deploy (`<owner>/large-video-model`) and the repo's ephemeral deploy-checks, all on one afternoon; the cold-boot breakdown and the VRAM peaks come from instrumented spike runs at the same shapes, because a first-ever 88 GB download only happens once per account. One region, a handful of runs — read the ranges, not the digits.

| Stage | Measured |
|---|---|
| **Cold boot** — first ever, download → weights on GPU | **392 s** |
| — snapshot download, 88.45 GB, 32 parallel range-gets | 365 s |
| — page-cache pre-read | 5.5 s |
| — `LTX2Pipeline.from_pretrained(...).to("cuda")` | 21.3 s |
| — tiny-VAE load | 0.4 s |
| **Warm boot** — `/data` already populated, through to serving traffic | **42 / 42 / 48 / 62 / 99 / 299 s** over six boots |
| — `fetch_model()`, now a `local_files_only` lookup | 0.0–1.2 s |
| — page-cache pre-read | 3.9–221.5 s |
| — `from_pretrained(...).to("cuda")` + tiny VAE | 20.7–83.7 s |
| — `setup()` warmup generation | 8.3–42.4 s |
| Warm generate — 768×512, 57 frames, 8 steps, via `/` | 2.6–3.2 s |
| Warm generate — same shape via `/stream`, previews on | 3.1–4.2 s |
| Warm generate — 1024×576, 97 frames, via `/stream` | 7.0–7.7 s (10.5–12.4 s to the `result` event) |
| Warm generate — 1024×576, 121 frames (the largest shape the API accepts) | 7.7 s + 2.8 s mux |
| HTTP round trip — 768×512, 57 frames, `/` (generate + mux + CDN upload) | 5.9–6.6 s |
| First request after scale-to-zero — machine allocation + warm boot + generate | 281 s, once (175 s waiting for an H200, 99 s boot, 7 s work) |
| VRAM — weights resident / peak at 768×512×57f / peak at 1024×576×97f | 65.9 / 68.5 / 71.9 GB of 139.8 GB |

**The warm boot is the number that moves.** The fetch collapses to nothing once `/data` holds the snapshot, but the two stages after it depend on how hot that region's `/data` chunk cache is for the node you happen to land on. Our slowest boot spent 221 s in the page-cache pre-read; the next-slowest sailed through the pre-read in 6 s and then spent 84 s inside `from_pretrained` instead. It is the same 88 GB arriving over the same pipe either way — the pre-read does not *add* the time so much as move it somewhere you can see it. Size `startup_timeout` for the bad node, not the good one.

`setup()` then runs one warmup generation at the smoke shape before the worker serves traffic, so the first real request pays no autotune tax — that warmup is the last row of the boot breakdown, already included in the totals above.

- **You pay for every second the H200 is alive**, warm or idle; see [fal pricing](https://fal.ai/pricing). That is the lens for every knob below.
- **`keep_alive=300`.** Five minutes. Video work arrives in bursts — a person iterating on a prompt sends ten requests in ten minutes — and re-warming means paying the boot row again. Shorten it for genuinely sporadic traffic; the autoscaling math (and when `min_concurrency` beats a long `keep_alive`) is [tune-autoscaling](../tune-autoscaling/)'s subject.
- **`startup_timeout=1800`** covers exactly one boot in the app's lifetime. Measured cold boot is 392 s and the slowest *warm* boot we saw was 299 s — the margin is for the unlucky node, not for the normal case.
- **`preview_every`.** The distilled schedule is a fixed 8 steps, so this knob is coarser than it looks: `1` → 6 previews, `2` → 3, `3` → 2, `4` → 1. Anything above `2` undersells the feature. Previews cost the denoise loop almost nothing (the decode and encode run on the worker thread), so raising it saves bandwidth, not GPU time.
- **Shape.** `1024x576` at 97 frames is the shape worth demoing — ~11 s from request to final video, with six previews landing between 3.4 s and 6.8 s, where the 768×512 smoke shape is over almost before it starts. 121 frames at the same width costs the same 7.7 s of denoising and one more second of muxing, so the top of the accepted envelope is not a cliff. `num_frames` is snapped down to the model's 8k+1 grid by `snap_frames()`, so callers never get a 422 for asking for 60 frames — the response's `num_frames` reports the count actually generated.
- **Oversized requests fail safely.** A CUDA OOM becomes a typed 422 (`fal.exceptions.AppException`), deliberately *not* a 503 — a 503 tells the platform to restart the runner, throwing away 88 GB of warm weights over one bad request.
- **Untested direction:** an fp8 or otherwise quantized DiT would cut the largest line in the sizing table and could open up smaller, cheaper machine types. We did not measure it, so this example makes no claim about it.

## Licensing

The weights are **not** Apache-2.0 — unlike the FLUX.2 [klein] 4B in [first-image-api](../../00-getting-started/first-image-api/), you cannot treat this model as unconditionally yours to commercialize.

- **Model weights:** [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE). Royalty-free for most users, commercial use included, but two clauses decide whether that applies to you: organizations with **annual revenue of $10M or more** must obtain a commercial license from Lightricks, and distributed outputs must be **disclosed as machine-generated** where context does not already make that obvious. Read the license itself before shipping; the summary here is not legal advice.
- **No token required.** Both the model repo and the tiny-VAE weights are ungated, which is why CI can redeploy this example with no Hugging Face secret.
- **`taehv.py`** is vendored (no PyPI package exists, and this repo pins everything) from [madebyollin/taehv](https://github.com/madebyollin/taehv) — MIT, © Ollin Boer Bohan. The file header records the exact commit.

## Next steps

- [**tune-autoscaling**](../tune-autoscaling/): the `keep_alive` / `min_concurrency` / concurrency math behind the one-line rationale above, on a model that is cheap to be wrong about.
- [**multi-gpu-video**](../../04-genmedia-patterns/multi-gpu-video/): what to do when the sizing table above runs out of single-GPU options.
- Concept docs: [parallel file loading](https://fal.ai/docs/documentation/serverless/optimizations/parallel-file-loading) · [machine types](https://fal.ai/docs/documentation/deployment/machine-types) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage)
