# Train a LoRA → serve it

You shipped an image feature; now users want *their* style. This example
closes the loop on fal serverless: POST a ZIP of ~16 style images to your own
trainer app, get back a LoRA fine-tuned on FLUX.2-klein-**base**-4B, and serve
it from `/data` on the **distilled** klein-4B at 4 steps — train where
gradients behave, serve where inference is fast.

## What you get

| base model (`lora_scale: 0`) | `lora_scale: 0.7` | `lora_scale: 1.0` |
|---|---|---|
| ![base](https://huggingface.co/rehan-fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-base.png) | ![0.7](https://huggingface.co/rehan-fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-scale-07.png) | ![1.0](https://huggingface.co/rehan-fal/flux-2-klein-4b-paper-diorama-lora/resolve/main/images/grid-scale-10.png) |

Same prompt, same seed — `pprdrma style, a tram crossing a city bridge at
sunset`. At 0.7 the style is a light miniature-model look; at 1.0 it's the
full paper-cutout diorama. The adapter was trained by this folder's own
trainer on 16 klein-generated images ([dataset](https://huggingface.co/rehan-fal/flux-2-klein-4b-paper-diorama-lora/tree/main/dataset), [how it was made](#how-the-demo-was-made)).

## Run it

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py            # the server, seeded with the demo adapter
```

Then train your own (deploy first — training runs go through the queue):

```bash
fal deploy app_trainer.py
```

```python
import fal_client

# 1. preflight - free sanity check on your dataset
print(fal_client.run("<username>/klein-lora-trainer", arguments={
    "image_data_url": "https://.../your-style.zip",
    "trigger_phrase": "mystyle style,",
}))

# 2. train - submit to the queue and poll; a full run takes a few minutes
handle = fal_client.submit("<username>/klein-lora-trainer", path="/train",
    arguments={
        "image_data_url": "https://.../your-style.zip",
        "trigger_phrase": "mystyle style,",
        "lora_name": "my-style",
        "steps": 1800,  # BFL's style band is 750-1500; the demo used 1800 @ 1e-4
    })
for event in handle.iter_events(with_logs=True):
    print(event)
print(handle.get()["data_path"])   # -> /data/loras/my-style/...

# 3. serve it - the server loads your adapter from /data by name
print(fal_client.run("<username>/klein-lora-server", arguments={
    "prompt": "mystyle style, a lighthouse at dusk",
    "lora_name": "my-style",
}))
```

## How it works

```
ZIP of images ──▶ app_trainer.py /train ──▶ /data/loras/<name>/ ──▶ app.py /
                  (klein-base-4B,            (atomic temp+rename)    (distilled
                   LoRA gradients)                                    klein-4B,
                                                                      4 steps)
```

- **Two apps, one folder, one shared module.** `shared.py` (dataset gates,
  `/data` conventions, artifact checks) ships to both runners via
  `local_python_modules` — [Import Code](https://docs.fal.ai/serverless/development/import-code).
  (`app_files` also ships code, but it fails to resolve at deserialization on
  ephemeral `fal run` — verified live; `local_python_modules` pickles the
  module by value instead.) Unit tests exercise the exact code the runners
  execute.
- **The trainer's settings block is the long-job pattern**: `keep_alive=0`,
  `request_timeout=7200`, `skip_retry_conditions=["timeout"]` (a timed-out
  training must not be silently re-run and re-billed), `max_concurrency=1`.
- **`/data` is the handoff** — mounted account-wide across all your apps
  ([persistent storage](https://docs.fal.ai/serverless/development/use-persistent-storage)).
  The trainer writes to a temp file *on /data* and `os.rename`s it into
  place: readers see the old file or the new one, never a partial write.
  The training response returns only after the rename — job completion IS
  the readiness signal.
- **The loop is the official recipe, inlined.** `/train` runs the core of
  HF's [klein DreamBooth script](https://github.com/huggingface/diffusers/blob/v0.39.0/examples/dreambooth/train_dreambooth_lora_flux2_klein.py)
  (pinned at the same diffusers 0.39.0 the server pins): precompute text
  embeddings + VAE latents once, offload both encoders to CPU, train
  transformer-only with flow matching. ~120 lines you can actually read.
- **Train on base, serve on distilled.** Gradients want the undistilled
  `klein-base-4B`; the saved adapter (diffusers-native keys) loads straight
  onto distilled `klein-4B` for 4-step inference — BFL's recommended split.
- **Validation before GPU time.** `POST /` is a dataset preflight (the exact
  head of `/train`): zip-slip-safe unpack, decode-check on every image,
  caption pairing — every user-fixable problem is a typed 422 naming the
  offending file, before a single training step is paid for. (Hosted fal
  trainers validate inline instead; the split here is for cheap CI.)

## Deploy

```bash
fal deploy app_trainer.py && fal deploy app.py
```

Both apps are `app_auth="private"`; the playground and the snippets above
authenticate with your key. CI re-verifies this example on every push: both
apps deploy ephemerally, the trainer's preflight and the server's generate
are smoked, and this folder's test suite runs against each live app (the
training run itself was validated once against a real deployment — the
demo adapter *is* that run's output).

## Cost & tuning

- **Training cost is `steps × seconds-per-step` on your `machine_type`** — see
  [pricing](https://fal.ai/pricing). Two cheap levers while you iterate:
  `RESOLUTION = 512` (a constant in `app_trainer.py`) roughly quarters
  per-step work, and `machine_type = "GPU-L40"` trades speed for a lower
  rate (klein-4B LoRA training fits comfortably in 48 GB).
- **Steps**: BFL's guidance for style LoRAs is 750–1500; the demo adapter
  used 1800 to push the style fully to `lora_scale` 1.0 (see "How the demo
  was made"). Below ~500 expect drift, not style.
- **`keep_alive=0` on the trainer** means every job pays boot time but idle
  costs nothing — right for occasional training. If you're queueing many
  jobs back-to-back, raise it so the 16 GB weights load once.
- **The server** follows 00.2's economics (`keep_alive=300`, L40-first) —
  adapter swaps add one `/data` read + `load_lora_weights` call, visible in
  `timings.lora` and `serving.adapter_source`.

## How the demo was made

`make_dataset.py` generated the 16 training images with klein itself (fixed
prompts + seeds, one consistent paper-diorama style suffix, captions describe
content only) — no external dataset, no licensing questions. The published
[reference adapter](https://huggingface.co/rehan-fal/flux-2-klein-4b-paper-diorama-lora)
is the output of this folder's own `/train` on that ZIP, trained for 1800
steps at `lr 1e-4` with a constant (not cosine) schedule and a widened LoRA
target-module list — the attention + feed-forward + context-projection set
the official script's README documents as its fal-trainer-equivalent
`--lora_layers`, rather than the script's bare attention-only default. That
default trains adapters roughly 3x too weak to express a style at
`lora_scale` 1.0; the wider targets plus the constant schedule are what get
the paper-diorama look to fully saturate. The server seeds this adapter into
`/data/loras/paper-diorama/` at boot if missing, so the example works before
your first training.

## Next steps

- **Serve thousands of these** — one adapter per user, LRU-cached on a warm
  fleet with routing hints: example 04.3. Your `lora_file` URL is drop-in
  for its `/register`.
- **Managed alternative**: [fal's hosted klein trainer](https://fal.ai/models/fal-ai/flux-2-klein-4b-base-trainer)
  is this pattern productized (same field vocabulary, no app to run).
- **Cold starts**: the 16 GB boot download is example 02.1's problem —
  FlashPack, `/data` caches, and compiled artifacts.
- **Progress streaming**: training logs stream through the queue; for
  first-class SSE progress see example 03.1.
