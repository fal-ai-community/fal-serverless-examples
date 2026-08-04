# Serve thousands of per-user LoRAs with routing hints

Your product trains one LoRA per user — an avatar style, a brand look, a fine-tuned identity for every paying customer. Serving them naively means either a deployment per user (thousands of cold fleets) or reloading adapters on every request (one slow fleet). This example runs **one warm FLUX.2 [klein] 4B fleet for the whole user base**: adapters live on `/data`, a KV store maps user → adapter, each runner keeps a small GPU LRU of loaded adapters — and each runner advertises its currently-hot users via [fal routing hints](https://fal.ai/docs/documentation/serverless/optimize-routing-behavior) so the router can prefer a runner that already has the adapter loaded. A hint is pure, best-effort affinity: any runner can serve any user; a hinted miss just pays the adapter load instead of failing. How much lift affinity adds on top of the scheduler's own runner reuse depends on your fleet size, cache pressure, and traffic shape — this example ships the measurement tool (`fleet_demo.py`) so you can answer that for your own workload instead of trusting anyone's benchmark.

> Catalog slot `04.3` ([SERV-1306](https://linear.app/features-and-labels/issue/SERV-1306)). Base model: [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — Apache-2.0 and ungated. The eight featured demo users below are real community/fal LoRAs, each license-verified (Apache-2.0/MIT); the "thousands" come from synthetic adapters the app generates on demand.

## What you get

Every response tells you *how* it was served — the routing story is observable per request:

```json
{
  "image": {
    "url": "https://v3b.fal.media/files/b/…/output.png",
    "content_type": "image/png",
    "width": 1024,
    "height": 1024
  },
  "seed": 42,
  "timings": { "lora": 0.0049, "inference": 1.4024 },
  "serving": {
    "worker_id": "ca5704b4",
    "adapter_source": "gpu-cache",
    "hint": "lora:numzoo",
    "resident_adapters": 1
  }
}
```

One warm worker cycling through all eight featured users (real run, one request each — warm hit is free, cold loads cost roughly their file size):

```
numzoo       source=gpu-cache  lora=0.00s inf=1.40s   # warmup had it loaded
yarn-art     source=disk       lora=0.35s inf=1.39s
baroque      source=disk       lora=0.35s inf=1.41s
koni-anime   source=disk       lora=2.93s inf=1.40s   # the 185 MB rank-64 one
wild-africa  source=disk       lora=0.25s inf=1.41s
mikkoani     source=disk       lora=0.33s inf=1.41s
spritesheet  source=disk       lora=0.30s inf=1.39s
liqglass     source=disk       lora=0.20s inf=1.35s   # 12 MB rank-8
```

Eight featured users, eight visibly different styles from one warm fleet — and they deliberately span all four LoRA key-format families (ai-toolkit, kohya, fal-trainer raw-PEFT, diffusers-native), all loaded by the same code path:

| `user_id` | Adapter | License | Size | Format family |
|---|---|---|---|---|
| `numzoo` | [goumsss/numzoo-flux2-klein-lora](https://huggingface.co/goumsss/numzoo-flux2-klein-lora) | Apache-2.0 | 92 MB | ai-toolkit |
| `yarn-art` | [Barath/flux2-klein-yarn-art-lora](https://huggingface.co/Barath/flux2-klein-yarn-art-lora) | Apache-2.0 | 92 MB | ai-toolkit |
| `baroque` | [noema-art/dutchbaroqueflux-klein](https://huggingface.co/noema-art/dutchbaroqueflux-klein) | Apache-2.0 | 92 MB | ai-toolkit |
| `koni-anime` | [Sawata97/flux2_4b_koni_animestyle](https://huggingface.co/Sawata97/flux2_4b_koni_animestyle) | Apache-2.0 | 185 MB | ai-toolkit |
| `wild-africa` | [zfrsgtcu/flux-wild-africa](https://huggingface.co/zfrsgtcu/flux-wild-africa) | MIT | 46 MB | ai-toolkit |
| `mikkoani` | [mikkoph/mikkoani-klein4b](https://huggingface.co/mikkoph/mikkoani-klein4b) | Apache-2.0 | 92 MB | kohya |
| `spritesheet` | [fal/flux-2-klein-4b-spritesheet-lora](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora) | Apache-2.0 | 76 MB | raw-PEFT (fal trainer) |
| `liqglass` | [Andy-ML-And-AI/LIQGLASS](https://huggingface.co/Andy-ML-And-AI/LIQGLASS) | Apache-2.0 | 12 MB | diffusers-native |

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # ephemeral app on a real GPU; prints playground + API URLs
```

The first-ever run downloads the base model (~16 GB) plus the eight featured adapters (~700 MB) into your account's `/data` cache; every run after that boots from cache. Try `user_id: numzoo` in the playground, then switch to `yarn-art` with the same prompt.

## How it works

**Three tiers of adapter state.** `/data` holds every adapter file (durable, shared account-wide). The KV store ([use the KV store](https://fal.ai/docs/documentation/serverless/use-kv-store)) maps `user:<id>` → `{path, url, scale, kind}` — the registry any runner can consult. Each runner keeps a bounded GPU LRU (`MAX_RESIDENT_ADAPTERS = 8`) of adapters actually loaded into the pipeline as diffusers named adapters: a hit is `set_adapters` and go; a miss loads from `/data` (`adapter_source: "disk"`) or re-fetches (`"downloaded"`), then evicts the least-recently-used adapter (`delete_adapters` + CUDA cache flush).

**Hints advertise the LRU — nothing more.** `provide_hints()` returns `lora:<user_id>` for exactly the residents, and fal's SDK re-evaluates it on **every response**, so the router always sees each runner's current cache. Two rules keep this safe at any fleet size: hints ride an HTTP response header with a 16 KB budget, so the list must stay small (the LRU bound *is* the hint-list bound — never advertise your full user table); and matching is exact and **best-effort** — if no advertising runner is free, the request goes to any free runner immediately. That's why correctness never depends on a hint. The strict `user_id` pattern (`^[a-z0-9][a-z0-9_-]{0,39}$`) makes every hint short, latin-1, and comma-free by construction; for URL-keyed fleets, hash instead: `lora:<sha256(url)[:16]>`.

**Clients never compute hints.** Every response carries the header `x-fal-runner-hint: lora:<user_id>` — capture it and pass it back as the `hint=` option on your next request for that user. One line of client code turns a stateless queue into cache-affine routing.

**`/register` is the product loop.** POST a `user_id` plus a LoRA URL or Hugging Face repo id (this is what a training job emits — see 04.4), and the app downloads it to `/data` (size-capped, safetensors-validated, temp-file + atomic rename so concurrent runners never see partial files) and writes the KV index. Omit the URL and the app generates a small deterministic synthetic adapter instead — that's how `fleet_demo.py` creates a thousand users without training a thousand LoRAs. Registration failures are typed 422s (`lora_too_large`, `lora_invalid_format`, `lora_download_failed`, `lora_incompatible`); GPU OOM during a load is deliberately **not** a 422 — it propagates as a retryable failure.

**Warmup goes through the LoRA path.** `setup()` ends by activating a featured adapter and generating once — a runner that can't serve adapters never enters the pool.

## Deploy

```bash
fal deploy app.py
```

**Python** — the `hint=` option sets the `X-Fal-Runner-Hint` header for you:

```bash
pip install fal-client
```

```python
import fal_client

# First request for this user: no hint yet - lands anywhere.
result = fal_client.subscribe(
    "<your-username>/per-user-loras",
    arguments={"user_id": "numzoo", "prompt": "NUMZOO a fox portrait, pastel colors"},
)
hint = result["serving"]["hint"]          # "lora:numzoo" (also in the response header)

# Repeat traffic: send the hint - the router now prefers runners
# already advertising this user's adapter.
result = fal_client.subscribe(
    "<your-username>/per-user-loras",
    arguments={"user_id": "numzoo", "prompt": "NUMZOO a lighthouse at dusk"},
    hint=hint,
)
print(result["serving"]["adapter_source"])  # "gpu-cache" on a hint hit
```

**JavaScript** — note: `hint` is a `queue.submit` option; `fal.subscribe` does not take it:

```javascript
import { fal } from "@fal-ai/client";

const { request_id } = await fal.queue.submit("<your-username>/per-user-loras", {
  input: { user_id: "numzoo", prompt: "NUMZOO a fox portrait, pastel colors" },
  hint: "lora:numzoo",
});
```

**curl (queue API)** — the raw header:

```bash
curl -X POST https://queue.fal.run/<your-username>/per-user-loras \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -H "X-Fal-Runner-Hint: lora:numzoo" \
  -d '{"user_id": "numzoo", "prompt": "NUMZOO a fox portrait, pastel colors"}'
```

**Measure affinity on your own traffic.** `fleet_demo.py` registers a synthetic fleet (a thousand users in a few minutes — each `/register` without a URL generates a small deterministic adapter server-side), drives Zipf-skewed traffic through the queue, and prints a per-worker warm/cold table — run it with and without hints and compare:

```bash
fal apps scale per-user-loras --min-concurrency 2
python fleet_demo.py --app <your-username>/per-user-loras --users 1000 --requests 300 --queue
python fleet_demo.py --app <your-username>/per-user-loras --users 1000 --requests 300 --queue --no-hints --skip-register
fal apps scale per-user-loras --min-concurrency 0   # stop the always-on billing
```

Two things to know before reading your numbers. First, the scheduler already reuses recently-released runners on its own, so small fleets have strong *implicit* stickiness even without hints — the delta hints buy grows with fleet size and cache pressure, and can be near zero on a 2–3 runner demo. Second, the regime where per-runner caches matter most is when your hot working set is **bigger than one runner's `MAX_RESIDENT_ADAPTERS` but fits the fleet's aggregate** — below that everything is warm everywhere; far above it, every cache thrashes no matter how requests route.

## Cost & tuning

- **You pay per second a worker is alive** on whichever machine the fallback list lands (`["GPU-L40", "GPU-H100"]`; klein-4B bf16 plus eight resident adapters fits the L40 comfortably) — see [fal pricing](https://fal.ai/pricing).
- **Affinity needs ≥ 2 warm runners to be visible** — with one runner every request lands on it anyway. `min_concurrency=2` while you demo; back to `0` after (scale-to-zero).
- `MAX_RESIDENT_ADAPTERS` trades VRAM for hit rate: sizing it to your hot-user working set matters more than raising `keep_alive`. Each resident adapter costs roughly its file size in VRAM.
- `keep_alive=300` holds a warm worker 5 minutes past the last request — long enough that a user's edit-generate-edit loop never cold-starts.
- **`/data` growth is real for real fleets**: this demo's synthetic adapters are ~10 MB × N users, and nothing on `/data` is auto-pruned. A production fleet should periodically delete adapters for churned users (the KV record is the index to sweep).

## Next steps

- **04.4 — Train a LoRA → serve it**: the training side of this loop; its output URL is exactly what `/register` accepts.
- **02.1 — Cut cold starts**: `torch.compile` and compiled caches — deliberately skipped here so heterogeneous adapters can't trigger recompiles; the advanced path is diffusers' [LoRA hotswapping](https://huggingface.co/docs/diffusers/main/tutorials/using_peft_for_inference).
- **06.2 — Async queue + webhooks**: wire this endpoint into a product backend.
- Concept docs: [optimize routing behavior](https://fal.ai/docs/documentation/serverless/optimize-routing-behavior) · [use the KV store](https://fal.ai/docs/documentation/serverless/use-kv-store) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) · [scale your application](https://fal.ai/docs/documentation/deployment/scale-your-application)
