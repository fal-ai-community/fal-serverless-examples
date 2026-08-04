# Design: 04.4 — Train a LoRA → serve it

**Catalog slot** `04.4` · [SERV-1312](https://linear.app/features-and-labels/issue/SERV-1312) · Folder: `04-genmedia-patterns/train-serve-lora/`
**Status**: approved design, pre-implementation. Working doc for the branch; drop or keep at PR time (owner's call).
**Scope retarget (owner decision, 2026-08-03)**: the ticket predates 04.3's model choice and says "closes the WAN training loop." Retargeted to **FLUX.2-klein-4B** so the serve side genuinely reuses 04.3's pattern, the trained adapter feeds 04.3's `/register`, and the tier tells one coherent story. The old WAN training docs page gets superseded in the fal-docs phase; a scope-note comment goes on SERV-1312.

## Problem & story

You shipped an image feature and now users want *their* style. This example closes the training loop on fal serverless: POST a ZIP of ~16 style images to your own trainer app, get a LoRA fine-tuned on FLUX.2-klein-**base**-4B, and have your serving app generate with it on **distilled** klein-4B at 4 steps — train where gradients behave, serve where inference is fast (the same split BFL recommends and fal's hosted trainer uses). The handoff is `/data` persistent storage: account-wide, cross-app, no download step. The serving app is a deliberately simplified single-adapter cousin of 04.3's fleet server; 04.4's output is drop-in for 04.3's `/register`.

Platform features taught, per the ticket: Persistent Storage · Import Code · Application Setup · Deploy to Production · forward link to 04.3.

## Load-bearing facts (verified 2026-08-03)

1. **Long requests**: an app's `request_timeout` caps execution time (not queue wait); standard accounts can raise it to **7200 s**. On breach the runner is killed and the queue **retries the whole job** unless `skip_retry_conditions=["timeout"]` is set. Long jobs run through the queue (`submit`/`subscribe`), never raw sync HTTP.
2. **`/data`**: mounted on every runner **across all the account's apps**; renames are atomic (documented pattern: write temp file beside destination, `os.rename`). Same-server visibility is immediate on close; cross-datacenter visibility lands after a background object-store upload — a reader that sees the path but not the chunks blocks briefly rather than erroring. Responding only after the rename makes the training response itself the readiness signal. `HF_HOME` is auto-set to `/data/.cache/huggingface`.
3. **Model**: `black-forest-labs/FLUX.2-klein-base-4B` and `FLUX.2-klein-4B` are both **Apache-2.0, ungated** (9B variants are gated/non-commercial — do not swap in). Both have `guidance_embeds: false` (no guidance plumbing in training). Text encoder is Qwen3-4B; total weights ~16 GB bf16 per variant. LoRA training at rank 16 / 1024 px / batch 1 with gradient checkpointing fits an L40 comfortably; H100 is a speed upgrade.
4. **diffusers 0.39.0** (the repo's serving pin) ships the official klein DreamBooth LoRA script at its tag (`examples/dreambooth/train_dreambooth_lora_flux2_klein.py`) and its `Flux2LoraLoaderMixin` loads all four key-format families in the wild (diffusers-native, ai-toolkit, kohya, raw-PEFT). Saving via `Flux2KleinPipeline.save_lora_weights(transformer_lora_layers=get_peft_model_state_dict(transformer))` emits diffusers-native keys — loadable by this example's server, 04.3's `/register`, and fal's hosted klein LoRA endpoints.
5. **BFL's published recipe** (HF blog + BFL docs): train on klein-base, serve the adapter on distilled ("faster and usually even better results"); style LoRAs peak ~750–1500 steps; lr 8e-5–1e-4 (fal's hosted trainer defaults to 5e-5 / 1000 steps).
6. **Harness**: multiple `app*.py` files in one folder are each ephemeral-deployed sequentially and run against the same `test_app.py`, with `HARNESS_VARIANT=<file stem>` exposed (precedent: `same-model-two-ways`). Exactly one `fal.App` subclass per app file. The smoke request is built from `Field(examples=[...])` and hits `/` with a 120 s client timeout — so the trainer's `/` must be cheap.

## Files

```
04-genmedia-patterns/train-serve-lora/
├── app.py            # KleinLoraServer — serve side (deployed & smoked first)
├── app_trainer.py    # KleinLoraTrainer — training side
├── shared.py         # dataset validation, /data path conventions, adapter sanity check, atomic writes
├── make_dataset.py   # generates the ~16-image style set via hosted klein (or your 00.2 app)
├── test_app.py       # branches on HARNESS_VARIANT; plus free no-GPU unit tests for shared.py
├── requirements.txt  # mirrors the (identical) App.requirements of both apps
└── README.md
```

Both apps ship `shared.py` via `app_files=["shared.py"]` — the documented Import Code mechanism this entry teaches. Both declare the **same** pinned `requirements` list so one `requirements.txt` mirrors both: `fal==1.78.2`, `diffusers==0.39.0`, `transformers==5.3.0`, `torch==2.12.1`, `accelerate==1.14.0`, `peft` (exact pin at build), `safetensors`, `hf-transfer==0.1.9`. No bitsandbytes: only LoRA params train, so optimizer state is tiny and plain AdamW suffices.

## Trainer app (`app_trainer.py`)

`KleinLoraTrainer(fal.App, keep_alive=0)` · `machine_type = "GPU-H100"` · `startup_timeout = 1200` · `request_timeout = 7200` · `skip_retry_conditions = ["timeout"]` · `max_concurrency = 1` · `app_auth = "private"`. The settings block is teaching material: keep_alive=0 (each job pays for exactly what it uses; no adapter-state leak), one training per runner, never re-run a timed-out job.

### `POST /` — dataset preflight (CI-smoked)

Runs the exact validation head of `/train`, no GPU work, and returns `PreflightReport{num_images, num_captioned, warnings: list[str], ready: bool}`. Validation: HTTPS-only download with the size cap enforced on **streamed bytes** (not just Content-Length), reject `text/html` responses (expired presigned URLs), zip-slip-safe unpack (reject path traversal and symlinks; skip `__MACOSX`/`._*`), full PIL decode of every image (catches truncation), EXIF orientation normalize, min dimensions, caption pairing (`img.txt` sidecars; `default_caption` fallback; missing-caption errors list the exact files). Input: `image_data_url` (with `examples=[<the published demo ZIP>]` for the harness), `default_caption`, `trigger_phrase`. An honest README note: hosted fal trainers validate inline at the head of the training request — this split exists so CI can exercise the input path without paying for a train.

### `POST /train` — the real thing (manually validated, not CI-run)

Input (echoes the hosted klein trainer's public surface; same names and bounds):

| Field | Type / default | Notes |
|---|---|---|
| `image_data_url` | `str`, required | ZIP of images, optional per-image `.txt` captions (matches the hosted trainer's field name) |
| `steps` | `int = 1000` | ge=100, le=10000, multiple_of=100 |
| `learning_rate` | `float = 5e-5` | ge=1e-6, le=1e-3 (explicit bounds — an improvement over the product) |
| `trigger_phrase` | `str \| None` | plain caption-prepend at prep time (no token surgery) |
| `default_caption` | `str \| None` | used where a sidecar caption is missing |
| `lora_name` | `str`, required | `^[a-z0-9][a-z0-9_-]{0,39}$` — becomes the `/data` path; same slug discipline as 04.3's `user_id` |

Fixed internals as commented constants: `RANK=16`, `ALPHA=16`, `DROPOUT=0.05`, AdamW + constant-after-warmup schedule (100 warmup steps), `BATCH_SIZE=1`, `RESOLUTION=1024`, bf16 weights with fp32 LoRA master weights, gradient checkpointing, seed 42. LoRA targets use the official README's fal-trainer-equivalent list (attention + feed-forward + context projections, plus single-block attention outputs) — the script's attention-only default trains adapters too weak to express a style at `lora_scale` 1.0 (validated empirically during the 2026-08-04 run; cosine decay had the same weakening effect, hence constant-with-warmup). Params mean what they say — no silent rescaling of steps or lr.

Output: `lora_file: File` (CDN upload — portable; drop-in for 04.3 `/register`), `data_path: str` (`/data/loras/<lora_name>/pytorch_lora_weights.safetensors`), `config: dict` (echo of the public inputs incl. trigger phrase), `steps: int`. Progress via `print()` every 25 steps (queue logs); README warns that first-step silence is compilation, not a hang, and points to 03.1 for real streaming.

### Training core (~120-line loop, in-app)

Anchored verbatim to the official diffusers klein script's step semantics at the v0.39.0 tag (source-URL cited in a comment: "the official script's core, inlined so you can read it"):

1. `setup()`: load `Flux2KleinPipeline` for **klein-base-4B** (HF cache on `/data`), hard-fail, no lazy loading.
2. Per request: preflight head → **precompute once** — VAE-encode all images, text-encode all captions, then move VAE + text encoder to CPU and `empty_cache()`. A 16-image set caches in seconds; the loop then touches only the transformer.
3. Fresh `LoraConfig` on the transformer per request; attention + fused-projection target modules exactly as the official script's defaults.
4. Loop over cached latents: timestep sampling, loss weighting, and the flow-matching target copied exactly as the official script implements them (the script is the source of truth — no scheme substitutions), guidance unset, gradient checkpointing on.
5. Finalize: `save_lora_weights(transformer_lora_layers=get_peft_model_state_dict(transformer))` → **artifact sanity gate** (safetensors header parses; keys match expected LoRA modules; tensors finite and not all-zeros) → atomic temp+rename into `/data/loras/<lora_name>/` (temp file on `/data`, same filesystem) + `config.json` sidecar → CDN upload → respond.

Error surface: `FieldException` 422s naming the exact offending file for everything user-fixable (bad ZIP, no images, oversized, missing captions, bad slug); CUDA/OOM faults re-raised as 5xx so the platform retries on healthy hardware — validation all happens before GPU time, so failures cost nothing.

## Server app (`app.py`)

`KleinLoraServer(fal.App, keep_alive=300)` · `machine_type = ["GPU-L40", "GPU-H100"]` · `startup_timeout = 1200` · `app_auth = "private"`. Warm **distilled** klein-4B pipe; one resident adapter slot (a request naming a different `lora_name` swaps it) — deliberately the 20-line version of 04.3's LRU fleet.

Input: `prompt`, `lora_name: str = DEMO_LORA` (a `shared.py` constant — its string value is fixed at implementation together with the style choice; same pattern as trainer), `lora_scale: float = 1.0` (ge=0, le=4 — the canonical single-adapter field), `image_size: ImageSizeInput`, `num_inference_steps` (default 4), `seed: int | None` (echoed), `sync_mode`. Output: `image: Image`, `seed: int`, `timings: {lora: s, inference: s}`, `serving: {adapter_source: "cached" | "disk" | "seeded", lora_name}`.

Adapter load path: resolve `/data/loras/<lora_name>/pytorch_lora_weights.safetensors` → missing → 422 `lora_not_found` (message tells the reader to run the trainer or check the slug) → header-only `safetensors.safe_open` validation (corrupt file → 422 with cleanup) → `load_lora_weights` → log "N/M keys unused" (the diagnostic that answers most bad-LoRA support tickets) → size-mismatch `RuntimeError` rewritten to a curated "likely trained for a different model architecture" 422. GPU OOM during load/apply re-raises (retryable 5xx, never a permanent 422).

`setup()` seeding: if the demo adapter is absent from `/data`, download the published reference adapter through the **same** atomic-write helper in `shared.py` (idempotent bootstrap, 04.3's pattern), then run one warmup generation **through the adapter path** — CI is green on a fresh account and the whole serving path is validated at boot.

## CI & tests

- **Free unit tests (no GPU)**: zip-slip rejection, caption pairing, slug pattern, atomic-rename helper, sanity-gate on a crafted bad safetensors.
- **Trainer variant (harness)**: `/` preflight smoke against the published demo ZIP; 422 paths (non-HTTPS URL, HTML response, empty ZIP, uncaptioned files without `default_caption`). **`/train` is never called in CI.**
- **Server variant (harness)**: generate with the seeded demo adapter asserting `serving.adapter_source` and timings present; `lora_scale=0` still succeeds; unknown `lora_name` → 422; fixed seeds throughout.
- **One-time manual validation protocol** (documented in README's deploy section, closes the DoD): `fal deploy` both apps → queue-submit a real 1000-step train of the demo dataset → confirm the server generates with the fresh `/data` artifact (adapter_source transitions) → confirm `lora_file` loads via `load_lora_weights` on a distilled pipe → publish that adapter + dataset ZIP to the fal-org HF repo → wire the URLs into `shared.py` constants and the README.

## Demo assets

`make_dataset.py` generates ~16 images in one distinctive style (fixed prompts + seeds; exact style chosen at implementation by looking at outputs — candidate: paper-cutout diorama) via the hosted klein endpoint or the reader's own 00.2 deployment, and zips them. Canonical ZIP + reference adapter published under a **fal-org HF repo** (Apache-2.0, license page visible, stable anonymous `resolve/main` URLs). The README's before/after grid renders base vs `lora_scale` 0.7 vs 1.0 with fixed seeds.

## README (§4-A order)

Problem → output-first before/after grid → run it (queue `submit` + status polling via `fal_client`; "a few minutes" phrasing, **no wall-clock or latency claims** per the measurement descope) → how it works (loop diagram trainer → `/data` → server; two-app folder anatomy; Import Code; the settings block explained) → deploy + manual validation protocol → cost & tuning (parametric steps × GPU-rate math, `RESOLUTION=512` for cheap iteration, L40 fallback, when to raise `request_timeout`, why keep_alive=0) → next steps: 04.3 forward link ("serve thousands of these — `lora_file` is drop-in for its `/register`"), fal's hosted `flux-2-klein-4b-base-trainer` as the managed alternative, 02.1 cold starts, 03.1 streaming.

## Non-goals

Multi-GPU / distributed training (04.5) · SSE progress streaming (03.1) · LoRA fleets, routing hints, KV registry (04.3) · resumable checkpointing (no checkpoint knobs shipped at all — knobs that imply resumability without implementing it mislead) · video/WAN (retargeted; superseded docs page handled in the fal-docs phase) · marketplace billing headers (08.3).

## Key decisions log

| Decision | Choice | Why |
|---|---|---|
| Model pair | train klein-**base**-4B, serve distilled klein-4B | BFL's published recommendation; matches fal's hosted trainer; both Apache-2.0/ungated; adapter loads on distilled |
| Training engine | handwritten in-app PEFT loop (~120 lines), math copied verbatim from the official diffusers 0.39.0 klein script | matches the direction of fal's production image trainers (in-app loops; runtime-fetched training code avoided); best teaching value; diffusers-native output keys; the subprocess-a-script alternative gives no in-band progress and stdout-scraped errors |
| Handoff | `/data` atomic temp+rename + CDN `File` in the response | `/data` is the entry's headline feature and the simpler in-account path; CDN URL keeps the artifact portable (04.3 `/register`, sharing) |
| Trainer `/` endpoint | dataset preflight (validation head of `/train`) | CI must smoke the trainer cheaply; validation-before-GPU is the production principle worth teaching |
| Public surface | 5 user fields + `lora_name`, product-matching names/bounds/defaults | readers graduate to the hosted trainer with the same vocabulary; fixed internals as commented constants |
| Trainer machine | `GPU-H100` (single string) | training wants speed; L40 documented as the budget fallback in Cost & tuning |
| Server seeding | idempotent `/data` bootstrap from the published reference adapter + adapter-path warmup | CI green on any account; same pattern as 04.3's featured users |
| No bitsandbytes | plain AdamW | only LoRA params train; optimizer state is tiny; one fewer heavyweight pin |

## Landing checklist

Tier README row update (`needs-deploy` → link folder) · `llms.txt` entry · commit `feat(04-genmedia-patterns): train a LoRA and serve it (SERV-1312)` · SERV-1312 comment documenting the WAN→klein retarget + link to this spec (ticket currently assigned to Jim — owner to reassign or coordinate) · publish HF artifacts under the fal org (needs org perms) · fal-docs page (incl. superseding the old WAN training page and its nav/card fixes from the ticket) is a separate later phase branched from current fal-docs main.
