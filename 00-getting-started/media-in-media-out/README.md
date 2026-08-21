# Media in, media out

Every real fal app moves files: users send media in, models send media out, and in between you need downloads that can't be abused, caching that doesn't re-fetch the world, and outputs a browser can actually render. This example is that plumbing, isolated — no model, just a fast image processor on a **CPU machine** that demonstrates how media flows through fal serverless.

> Catalog slot `00.3`.

## What you get

```json
{
  "image":     { "url": "https://v3.fal.media/files/…/….webp", "content_type": "image/webp",
                 "width": 768, "height": 1024 },
  "thumbnail": { "url": "https://v3.fal.media/files/…/….webp", "content_type": "image/webp",
                 "width": 192, "height": 256 },
  "report":    { "url": "https://v3.fal.media/files/…/report.json",
                 "content_type": "application/json" }
}
```

(Those are the real dimensions for the sample input, which is 768×1024 — resizing *bounds* the longest edge, preserving aspect ratio and never upscaling.) The playground renders `image` and `thumbnail` as previews and `report` as a download link — media detection is driven by the output schema (`fal.toolkit` types tag their fields for the playground) plus `url` + `content_type` pairs.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # prints playground + API URLs; Ctrl-C to stop
```

## How it works

**No `setup()`.** Media plumbing is I/O-bound, not model-bound — there is nothing to preload, so the app defines no `setup()` at all.

**Media in — the `_url` convention.** Inputs arrive as URL strings; the `image_url` field name is what gives the playground its drag-and-drop upload widget (uploads become URLs before your code ever runs). See [handle inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs).

**Bounded download, cached on `/data`.** `download_file` refuses anything over 20 MB (a `DownloadError` becomes a typed 422 pointing at `image_url`) and every fetch goes through the SDK's SSRF-guarded HTTP path — two protections you get for free. What the SDK does *not* do is skip repeat transfers, so the example adds the cache itself: downloads land in a URL-hashed directory on [`/data` persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage), and a repeat request for the same URL reuses the file — across workers, deploys, and CI runs. The `report.source.cache_hit` and `download_ms` fields make it visible: hundreds of ms on first fetch, ~zero after. (Hardening beyond this — decompression-bomb policy, typed error taxonomies — is 00.4's subject; this example does gate absurd pixel counts before decode. HEIC/AVIF inputs are out of scope here.)

**Media out — `output_format` → content type.** The format you ask for is the content type you get: `Image.from_pil(processed, format="webp")` uploads to fal's CDN and stamps `content_type: image/webp`. Outputs are typed `fal.toolkit.Image`/`File` — the `File` report shows that anything can ride the same pipeline, not just images.

**Per-file retention.** `expires_in_seconds` forwards an object lifecycle preference with the upload — per-file control over how long the CDN keeps it. (Callers of a deployed app can also set this per request via the `x-fal-object-lifecycle-preference` header, no app code needed; a value set in the request body wins over the header. Full file access controls are covered in the fal docs.)

**`sync_mode`.** Set it to skip the CDN entirely: outputs come back as inline `data:` URIs — no upload round-trip, for callers who want bytes over hosted URLs.

## Deploy

```bash
fal deploy app.py
```

```bash
curl -X POST https://queue.fal.run/<your-username>/media-in-media-out \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"image_url": "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png", "output_format": "jpeg"}'
```

## Cost & tuning

- **How uploads are billed — the honest version.** On fal Serverless there is no per-request metering (with one documented exception: requests failing with a 5xx aren't charged) and no separate fee for CDN uploads: you pay one per-second rate for every second the runner is alive, from `setup()` through teardown — setup, busy, and idle all bill the same. `Image.from_pil()` uploads synchronously inside your handler, so upload time is part of request wall-clock — this app *measures* it for you in `report.upload_ms` rather than promising a number. What's worth optimizing is *throughput*: a handler that returns sooner serves more requests per billed second. (fal's hosted Model APIs bill differently — per output, excluding upload — which is where "uploads aren't billed" claims come from; don't expect that on self-hosted apps.)
- A telling detail from the SDK itself: PNG outputs are saved at `compress_level=1` — fal deliberately trades a slightly larger file for less handler time, which is exactly the right trade under per-second billing.
- This app runs on `M` (2 vCPUs / 2 GB) — a small CPU rate — see [fal pricing](https://fal.ai/pricing). File plumbing needs no GPU; picking the smallest machine that fits is the whole cost story for CPU apps.

## Next steps

- **00.4 — Bulletproof inputs**: the full hardening this example only gestures at (typed error taxonomies, archive gates, validation policy — note the SDK's `download_file` is already SSRF-guarded).
- **00.2 — Your first image API**: put a model behind this same file plumbing.
- Concept docs: [inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) · [fal.toolkit](https://fal.ai/docs/api-reference/python-sdk/fal-toolkit)
