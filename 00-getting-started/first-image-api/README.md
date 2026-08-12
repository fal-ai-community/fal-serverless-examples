# Your first image API

You want a text-to-image endpoint you own: your model choice, your API schema, your infrastructure — not a black-box third-party API. This example deploys **FLUX.2 [klein] 4B** behind a typed HTTP endpoint on a fal GPU: weights load once per worker, every request hits a warm model, and callers get back a CDN-hosted image.

> Catalog slot `00.2` ([SERV-1293](https://linear.app/features-and-labels/issue/SERV-1293)). Model: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — **Apache-2.0 and ungated**, so you can ship what this tutorial teaches commercially, with no Hugging Face token anywhere. (The klein **9B** variants are gated and non-commercial — don't swap them in.)

## What you get

```json
{
  "image": {
    "url": "https://v3.fal.media/files/b/…/output.png",
    "content_type": "image/png",
    "width": 1024,
    "height": 768
  },
  "seed": 42
}
```

…and a generated playground UI: prompt textarea, image-size preset dropdown, steps slider, seed with a randomize button — all derived from the `Input` schema in [`app.py`](app.py).

One shape note: this tutorial returns a single `image`; fal's hosted FLUX endpoints return `images` (a list), so expect the plural when you move between the two.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # ephemeral app on a real GPU; prints playground + API URLs
```

The first-ever run downloads ~16 GB of weights into your account's `/data` cache (a few minutes); every run after that boots straight from the cache. `fal run` keeps the app alive until Ctrl-C.

## How it works

**`setup()` — load → compile → warmup.** Everything slow happens once per worker, before the first request is served:

- **Load**: `Flux2KleinPipeline.from_pretrained(...)` with no `cache_dir` — fal points `HF_HOME` at [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) (`/data`), so weights download once per *account*, then every future worker (and every CI deploy) loads from cache. `HF_HUB_ENABLE_HF_TRANSFER=1` accelerates that first download.
- **Compile**: this example deliberately skips `torch.compile` — it's cold-start engineering with a measurable trade-off, covered where it belongs (example 02.1, compiled caches). The comment in `setup()` marks where it goes.
- **Warmup**: one single-step generation pays CUDA init and kernel autotuning during [SETUP](https://fal.ai/docs/documentation/development/app-setup), so the first user request runs at steady-state speed.

**Machine-type fallback list.** `machine_type = ["GPU-L40", "GPU-H100"]` is tried in order until capacity is found: klein-4B in bf16 (~16–18 GB peak) fits comfortably on the cheaper L40, and H100 is the fallback under load — see [machine types](https://fal.ai/docs/documentation/deployment/machine-types).

**Typed schema = docs + playground + tests.** Every `Field` carries a description, and required fields carry `examples` — that single convention powers the [playground](https://fal.ai/docs/documentation/model-apis/playground) form, the OpenAPI docs ([document your model](https://fal.ai/docs/documentation/development/document-your-model)), and the CI smoke request ([inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs)).

**The seed contract.** `seed` is optional; when omitted the server picks one — and the response always echoes the seed it used. Same seed + same inputs → the same image, which is what makes the API debuggable and the smoke test deterministic.

**`sync_mode`.** By default the image uploads to fal's CDN and you get a URL. With `sync_mode: true` the endpoint returns the image as a `data:` URI instead — no upload round-trip, for callers who want the bytes as fast as possible and don't need a hosted file.

**Fixed by design:** `guidance_scale` stays at 1.0 — klein is guidance-distilled, so classifier-free guidance is baked into the weights; exposing the knob would only let callers break output quality.

## Deploy

```bash
fal deploy app.py
```

This registers `first-image-api` and prints its playground and API URLs. The app deploys with `app_auth = "private"`, so only your fal key can call it; production image endpoints also add content safety (prompt moderation, NSFW checks on outputs) that this tutorial omits. Call it from anywhere ([calling your endpoints](https://fal.ai/docs/documentation/development/calling-your-endpoints)) — the queue API is the production default for multi-second generations:

**Python**

```bash
pip install fal-client
```

```python
import fal_client

result = fal_client.subscribe(
    "<your-username>/first-image-api",
    arguments={"prompt": "a calico cat riding a bicycle", "seed": 42},
)
print(result["image"]["url"], "seed:", result["seed"])
```

**JavaScript**

```javascript
import { fal } from "@fal-ai/client";

const result = await fal.subscribe("<your-username>/first-image-api", {
  input: { prompt: "a calico cat riding a bicycle" },
});
console.log(result.data.image.url);
```

**curl (queue API)**

```bash
curl -X POST https://queue.fal.run/<your-username>/first-image-api \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a calico cat riding a bicycle"}'
# → {"request_id": "...", "status_url": "...", ...} — poll the status_url, then fetch the result
```

## Cost & tuning

- **You pay per second the worker is alive**, on whichever machine type the platform picked from the fallback list — L40 is the cheap seat for this model; see [fal pricing](https://fal.ai/pricing).
- `keep_alive=300` holds a warm worker for 5 minutes after the last request: bursts of traffic skip the cold start entirely, at the price of idle seconds. Drop it toward `0` for batch-style workloads; raise `min_concurrency` above 0 to never cold-start at all (always-on billing).
- `startup_timeout=1200` exists for exactly one boot in the app's life — the first-ever weight download. After `/data` is warm, startup is load + warmup only.
- A 4-step distilled model is the frugal choice by construction: ~7–12× less compute per image than a 28–50-step base model at the same resolution.

## Next steps

- **00.3 — Media in, media out**: file inputs, CDN behavior, and persistent storage patterns this example only touched.
- **02.1 — Cut cold starts**: `torch.compile`, shared inductor caches, and weight-loading tricks — the "compile" step this example skipped, with the trade-offs measured.
- Concept docs: [application setup](https://fal.ai/docs/documentation/development/app-setup) · [machine types](https://fal.ai/docs/documentation/deployment/machine-types) · [calling your endpoints](https://fal.ai/docs/documentation/development/calling-your-endpoints)
