# Media in, media out

Every real fal app moves files: users send media in, models send media out, and in between you need downloads that can't be abused, caching that doesn't re-fetch the world, and outputs a browser can actually render. This example is that plumbing, isolated — no model, just a fast image processor on a **CPU machine** that demonstrates how media flows through fal serverless.

> Catalog slot `00.3` ([SERV-1302](https://linear.app/features-and-labels/issue/SERV-1302)).

## What you get

```json
{
  "image":     { "url": "https://v3.fal.media/files/…/….webp", "content_type": "image/webp",
                 "width": 1024, "height": 1024 },
  "thumbnail": { "url": "https://v3.fal.media/files/…/….webp", "content_type": "image/webp",
                 "width": 256, "height": 256 },
  "report":    { "url": "https://v3.fal.media/files/…/report.json",
                 "content_type": "application/json" }
}
```

The playground renders `image` and `thumbnail` as previews and `report` as a download link — media detection is driven purely by field names and `url` + `content_type` pairs.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # prints playground + API URLs; Ctrl-C to stop
```

## How it works

**Media in — the `_url` convention.** Inputs arrive as URL strings; the `image_url` field name is what gives the playground its drag-and-drop upload widget (uploads become URLs before your code ever runs). See [handle inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs).

**Bounded download, cached on `/data`.** `download_file(url, "cache/media-in", filesize_limit=20)` does three jobs in one line: it downloads the input, refuses anything over 20 MB (a `DownloadError` becomes a typed 422 pointing at `image_url` — input hardening gets the full treatment in 00.4), and because a *relative* target dir lands under [`/data` persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage), a repeat request for the same URL skips the download entirely — across workers, deploys, and CI runs. The `report.download_ms` field makes the cache visible: hundreds of ms on first fetch, ~zero after.

**Media out — `output_format` → content type.** The format you ask for is the content type you get: `Image.from_pil(processed, format="webp")` uploads to fal's CDN and stamps `content_type: image/webp`. Outputs are typed `fal.toolkit.Image`/`File` — the `File` report shows that anything can ride the same pipeline, not just images.

**Per-file retention.** `expires_in_seconds` forwards an object lifecycle preference with the upload — per-file control over how long the CDN keeps it. (Callers of a deployed app can also set this per request via the `x-fal-object-lifecycle-preference` header, no app code needed. Full file access controls are covered in the fal docs.)

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

- **How uploads are billed — the honest version.** On fal Serverless there is no per-request metering and no separate fee for CDN uploads: you pay one per-second rate for every second the runner is alive (setup, busy, and idle bill the same). `Image.from_pil()` uploads synchronously inside your handler, so upload time is part of request wall-clock — milliseconds for images this size. What's worth optimizing is *throughput*: a handler that returns sooner serves more requests per billed second. (fal's hosted Model APIs bill differently — per output, excluding upload — which is where "uploads aren't billed" claims come from; don't expect that on self-hosted apps.)
- A telling detail from the SDK itself: PNG outputs are saved at `compress_level=1` — fal deliberately trades a slightly larger file for less handler time, which is exactly the right trade under per-second billing.
- This app runs on `M` (2 vCPUs / 2 GB) — cents per hour. File plumbing needs no GPU; picking the smallest machine that fits is the whole cost story for CPU apps.

## Next steps

- **00.4 — Bulletproof inputs**: the full hardening this example only gestures at (typed error codes, decompression-bomb gates, SSRF-safe downloads).
- **00.2 — Your first image API**: put a model behind this same file plumbing.
- Concept docs: [inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) · [fal.toolkit](https://fal.ai/docs/api-reference/python-sdk/fal-toolkit)
