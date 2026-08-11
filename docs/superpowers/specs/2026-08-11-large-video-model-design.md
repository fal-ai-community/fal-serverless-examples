# Example 02.3 — Serve a 20+ GB video model, and watch it generate (design)

- **Date:** 2026-08-11
- **Ticket:** [SERV-1303](https://linear.app/features-and-labels/issue/SERV-1303) (scope addendum approved 2026-08-10: intermediate-step preview decoding added; big-model serving stays the headline)
- **Status:** approved design, **spike-gated** — implementation starts only after the spike below passes
- **Folder:** `02-scaling-cold-starts/large-video-model/`

## Problem

Serving a modern 20+ GB video model on fal raises three questions this example answers in one place:

1. **How do I get ~70 GB of weights onto a GPU fast?** (parallel download, `/data` persistent storage, page-cache pre-read)
2. **Which GPU do I actually need?** (real VRAM math against fal's machine-types table, not vibes)
3. **How do I make a minutes-long generation feel alive?** (decode intermediate diffusion steps with a tiny video VAE and stream previews over SSE, so users watch the video sharpen from noise — in motion)

The preview mechanism is novel: production systems stream previews from causal/block models that decode *finished* segments. Decoding *partially-denoised* full-sequence video latents mid-schedule for client preview has no public precedent we know of — which is exactly why it makes a good example.

## Model & assets (all public)

| Asset | Source | Notes |
|---|---|---|
| DiT, 22B distilled | Lightricks LTX-2.3 open-weights family (`Lightricks/LTX-2.3` originals; the app loads the official diffusers-layout conversion — exact repo id + revision pinned at spike) | bf16 ≈ 44 GB; 8-step distilled variant |
| Text encoder | Gemma-3-12B (as packaged by the diffusers-layout repo) | ≈ 24 GB bf16; **gated-public** — HF token with accepted license may be required |
| Video+audio VAE, vocoder | same checkpoint family | LTX-2.3 generates audio and video jointly |
| Tiny preview VAE | `taeltx2_3` (+ `_wide` variant) from [madebyollin/taehv](https://github.com/madebyollin/taehv), MIT | ~24 MB; decoder code vendored as `taehv.py` (no pip package exists; repo CI forbids unpinned/`git+` deps) |
| Pipeline code | `diffusers` `LTX2Pipeline` (native since v0.37) | Required path: Lightricks' official package needs Python ≥ 3.12; this repo targets 3.11 |

**License note (goes in README):** LTX-2.3 ships under the **LTX-2 Community License** — royalty-free including commercial use, but organizations with ≥ $10M annual revenue need a commercial license, and outputs carry an AI-disclosure requirement. It is *not* Apache 2.0. Gemma adds its own gated-license acceptance.

**Catalog note:** 04.2 (SERV-1300) is scoped as "LTX-class text-to-video". With 02.3 taking LTX-2.3, 04.2 should pick a different model or angle when it's built — flagged for the owner, not resolved here.

## File layout

```
02-scaling-cold-starts/large-video-model/
├── app.py          # one fal.App: '/' invoke (CI-smoked) + '/stream' SSE previews
├── loader.py       # ~150-line parallel weight loader (the ticket's promised artifact)
├── taehv.py        # vendored tiny video VAE decoder (MIT, attribution header)
├── client.py       # terminal SSE consumer (fal_client.stream)
├── viewer.html     # single-file browser viewer, POST+fetch SSE parsing
├── README.md       # §4-A order; GPU-sizing reasoning is the teaching centerpiece
├── requirements.txt  # exact mirror of App.requirements, all pinned
└── test_app.py     # '/' contract + raw-httpx SSE assertions
```

`loader.py` and `taehv.py` ship with the app via `local_python_modules`.

## Architecture

### App configuration

- `machine_type = ["GPU-H200"]` (possible `"GPU-B200"` capacity fallback — 00.2 precedent); `num_gpus` default 1.
- `app_auth = "private"`, `keep_alive = 300` (an idle H200 bills; revisit at plan time with a cost note), `startup_timeout = 1800` (cold boot downloads ~70 GB).
- `requirements`: `fal`, `diffusers` (version with LTX-2.3 support), `transformers`, `accelerate`, `imageio` + `imageio-ffmpeg`, `huggingface_hub` — final list and every `==` pin fixed at spike from the working deploy, mirrored in `requirements.txt`.
- **Generation envelope:** resolution presets up to ~768×512, frames 9–121 on the `8k+1` grid (default 121 @ 24 fps), steps fixed at 8 by the distilled schedule. Exact preset list and defaults set from spike measurements.

### `setup()` (every stage timed and logged — cold-start numbers are part of the story)

1. `loader.py`: set `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_CHUNK_CACHE_SIZE_BYTES=1000000000000`, `HF_XET_NUM_CONCURRENT_RANGE_GETS=32`; `snapshot_download(..., local_dir=/data/..., max_workers=32)` with a `local_files_only=True` warm-boot fast path (sub-second when cached); then `find … | xargs -P 32 cat > /dev/null` page-cache pre-read before any `from_pretrained` touches disk. Clean-room implementation of the recipe in fal's public docs (parallel file loading + download tuning pages).
2. Load `LTX2Pipeline` (distilled, bf16) → CUDA. Load `taehv.py` + `taeltx2_3` weights → CUDA, eval.
3. Warmup generation at the smoke-test shape (absorbs compile/autotune before the first request).

### Endpoints

- **`/` (invoke, CI-smoked):** plain generation, `callback_on_step_end=None`, returns `Output` JSON. The harness prefers `/`, so CI smoke needs no streaming support.
- **`/stream` (SSE):** `StreamingResponse(media_type="text/event-stream")` with header `x-fal-sse-last-event-as-body: 1` (same convention as fal's public streaming docs). Async generator drains a preview queue while generation runs in a worker thread.

### Preview pipeline (the novel part)

```
callback_on_step_end (denoise thread)          decode worker (thread pool)
  cache prev latents ─ x0-extrapolate ──▶  denormalize → split AV latent →
  push to maxlen=1 deque, return             taeltx decode → uint8 → half-res
  (never blocks denoising)                   mp4 (x264 ultrafast) → base64 URI
                                             → event queue → SSE generator
```

- **x0 extrapolation:** flow-matching latents are linear interpolations, so two consecutive callback states give a clean-sample estimate — early previews look like images, not noise. Public math; the spike compares raw vs extrapolated.
- **Latest-wins backpressure:** a `maxlen=1` deque between callback and decoder; if a preview is still encoding when the next step lands, the stale one drops. Previews can never stall denoising. 8 steps ⇒ typically 6–8 preview events.
- **Final event = complete `Output`:** full-VAE decode + audio mux → CDN upload → `{"type": "result", ...}` as the last SSE event.

### SSE event schema (every event JSON — `fal_client.stream()` contract)

```json
{"type": "preview", "step": 5, "total_steps": 8, "progress": 0.62,
 "video": "data:video/mp4;base64,...", "width": 384, "height": 256, "frames": 121}
{"type": "result", "video": {"url": "https://..."}, "seed": 42,
 "timings": {"generation_s": 31.2}}
```

Previews target ~0.3–0.8 MB base64 per event (half resolution, no audio).

### Clients

- `client.py`: `fal_client.stream()`, per-step progress line, writes `preview_latest.mp4` per event, saves the final video.
- `viewer.html`: single file; `fetch()` POST with manual SSE parsing (`EventSource` can't POST); hot-swaps a looping `<video>` per preview event. Demo-only, with a visible warning about not shipping FAL keys in real frontends. Not CI-tested.

## GPU sizing (README teaching section)

Walk the machine-types VRAM table against real numbers: ~44 GB DiT + ~24 GB Gemma + ~2 GB VAEs/vocoder ≈ **~70 GB of weights before activations**. A100-40G/L40-48G: no. H100-80G: weights alone nearly fill it — viable only with text-encoder offload or fp8, documented as the cost-reduction path in Cost & tuning. H200-141G: everything resident with activation headroom — the target. B200-192G: more headroom, higher cost. Actual peak VRAM measured in the spike; no unmeasured performance claims anywhere (repo rule).

## Error handling

- **Typed 422s:** prompt length caps; resolution presets only (LTX dims must be divisible by 32); `num_frames` clamped and coerced to the model's `8k+1` grid; total-pixel cap protecting VRAM headroom.
- **Previews never break generation:** entire callback body wrapped; on failure log, drop the preview, continue. Zero-preview stream with a good final video = degraded success.
- **Client disconnect:** generation completes in its worker thread; runner stays healthy.
- **OOM guard:** friendly error naming the knobs (frames, resolution) + `torch.cuda.empty_cache()` recovery.

## Testing & CI

- `test_app.py`:
  1. `/`: `video.url` starts `https://`, seed echoes.
  2. `/stream` via raw `httpx`: ≥ 2 `preview` events with valid `data:video/mp4;base64,` payloads (magic-byte check), strictly increasing `step`, terminal `result` with CDN URL.
- Smoke shape = `Input` `examples=[...]` = README sample; spike picks settings (~480p / ~57 frames / 8 steps expected) that finish well inside the 120 s request timeout warm.
- `HARNESS_STARTUP_TIMEOUT` raised for this example in the workflow if the spike's cold-boot measurement requires it (04.5 precedent).
- **CI prerequisite:** if the text encoder is license-gated in the pinned repo, the CI fal account needs `HF_TOKEN` set as a fal secret — surfaced in README + PR description up front.

## Spike (gate; one throwaway deploy, ~half a day)

| # | Question | Pass looks like |
|---|---|---|
| 1 | Pinned diffusers loads LTX-2.3 distilled via `LTX2Pipeline` on H200; `callback_on_step_end` exposes latents; 2.3 kwargs exist in a released version | model loads, callback fires with latent tensors |
| 2 | taeltx previews legible mid-schedule (raw vs x0-extrapolated; standard vs `_wide`) | a step-by-step contact sheet a human calls "watchable" from ~step 2–3 |
| 3 | Peak VRAM + wall time at candidate smoke/README shapes; cold download + load timings | fits H200 with headroom; smoke shape < 120 s warm |
| 4 | Preview decode+encode latency vs per-step budget; event sizes | decode+encode comfortably under the step time; events ≤ ~1 MB |
| 5 | Full `/stream` session through the real `fal.run` gateway from a local client | multi-minute SSE with MB events arrives incrementally, no buffering surprises |

**Pre-agreed fallbacks (no re-litigating):** previews → latent2rgb projection; GPU → B200; legibility → previews every 2 steps from step 2 with `_wide` decoder; diffusers LTX-2.3 support fundamentally broken → stop, re-spec on WAN 2.2 A14B.

## Rollout

1. Spike (above) → record numbers in the plan doc.
2. Implementation plan via writing-plans; TDD; example PR into this repo (spec + plan committed on-branch; drop-or-keep at merge is the owner's call).
3. CI deploy-check green (ephemeral deploy + smoke + SSE test + teardown confirmation).
4. Registered `fal deploy` validation; SERV-1303 updated.
5. fal-docs page: separate later phase, branched from current fal-docs `origin/main`, placement per SERV-1266 IA. Cross-link with 03.1 (image SSE example) both ways when both exist.

## Evidence policy

Production patterns informing this design (step-callback tiny-VAE previews, SSE data-URI payloads, latest-wins decode threading, parallel loading) are reimplemented clean-room from public sources: fal's public docs, the public diffusers/Lightricks/madebyollin repos, and fal's public streaming conventions. No internal file paths, internal wheels, or private repos are referenced by the shipped example or README.
