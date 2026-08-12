# Bulletproof inputs

The moment your endpoint is public, it stops receiving only the inputs you imagined. It receives a URL pointing at `169.254.169.254`, a "thumbnail" that turns out to be 400 MB, a 200 KB PNG that decodes into 430 MB of pixels and OOM-kills the worker mid-request, and a zip that unpacks to 200 GB. This example is the gauntlet those inputs hit before your model sees them — and, just as importantly, the **typed errors** callers get back instead of an opaque 500.

> Catalog slot `00.4` ([SERV-1309](https://linear.app/features-and-labels/issue/SERV-1309)).

## What you get

A clean request returns the audit trail of gates it passed:

```json
{
  "accepted": true,
  "format": "PNG", "width": 768, "height": 1024, "file_size": 1035058,
  "gates": ["url_policy", "download_cap", "format_allowlist", "pixel_budget", "decode"],
  "archive": null
}
```

A hostile one returns a 422 a client can branch on — never a 500:

```json
{
  "detail": [{
    "loc": ["body", "image_url"],
    "msg": "image is 12000x12000 (144,000,000 pixels); maximum is 89,478,485 pixels",
    "type": "image_too_large"
  }]
}
```

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py
```

## How it works

### What fal already protects you from

Verified against fal 1.78.2 — don't rebuild these:

- **SSRF.** `download_file()` fetches http(s) through an SSRF-safe client that refuses private, loopback, link-local (including the `169.254.169.254` metadata endpoint), CGNAT, and IPv6-transition addresses — and it **pins the resolved IP**, so a host can't pass validation then rebind to something internal before the connection. Every redirect hop is re-validated, and `Authorization`/`Cookie` headers are stripped on cross-origin hops.
- **Typed 422s for free.** Any pydantic `Field` constraint (`ge`, `le`, `Literal`, …) produces a documented-shape 422 with a machine-readable `type` — no code required. Try `output_format: "tiff"` on this app.
- **Failed requests stay cheap.** 4xx responses are never retried by the queue and leave the runner healthy; validation failures are billed zero units. (Never return 503 for bad input — that kills the runner.)

### What is yours to add — the five gates

1. **URL policy** (`check_url_policy`) — the SDK restricts *addresses*, not **ports**, and `data:` URIs bypass the SSRF client entirely. So the app states its own scheme and port allowlist first, before a byte moves.
2. **Download cap** (`download_guarded`) — `download_file`'s `filesize_limit` **defaults to `None`**, i.e. unbounded. Pass one. The SDK enforces it both from `Content-Length` and by counting streamed bytes, so a lying header doesn't help an attacker. Uncaught, a `DownloadError` is a plain exception → a 500 that looks like *your* bug; catching it and re-raising typed is the whole job.
3. **Format allowlist** — read from the header, before decode. Inline `data:` URIs get their own path (`read_data_uri`): base64 inflates by 4/3, so the *encoded* length gives you the decoded size before you spend memory on it — and it keeps `download_file` from logging the entire payload into your runner logs, which is what happens if you pass a data URI straight through.
4. **Pixel budget** (`check_pixel_budget`) — the decompression-bomb gate, and the one people skip. Pillow only *warns* at its default threshold and only raises at 2× it, so a 144-megapixel PNG sails through a byte cap and allocates ~430 MB in `load()`. Checking `width * height` from the header costs nothing and turns a worker crash into a 422. The default here (89,478,485) is Pillow's own threshold, which is also what fal's production ingestion enforces.
5. **Archive gates** (`inspect_zip`) — `fal.toolkit`'s `CompressedFile` calls `extractall()` with **no limits at all**. Inspect the central directory first: entry count (metadata bombs exhaust inodes even under a byte cap), declared uncompressed size, and member paths. Only extract what survives.

### Typed errors are an API, not a log line

Every rejection carries a `type` string drawn from [fal's documented error taxonomy](https://fal.ai/docs/documentation/model-apis/errors) — `image_too_large`, `file_too_large`, `file_download_error`, `unsupported_image_format`, `invalid_archive`, `archive_file_count_exceeds_maximum` — so clients that already handle fal model errors handle yours unchanged. `FieldException` gives you `loc`/`msg`/`type`; fal's first-party models additionally emit `ctx` and a docs `url`, which app code has to construct itself if it wants full parity.

The gates are pure functions, so [`test_app.py`](test_app.py) unit-tests them with zero infrastructure, then fires the real payloads — metadata endpoint, `file://`, oversized download, pixel bomb, zip bomb — at a live deployment and asserts the exact 422 that comes back.

## Deploy

```bash
fal deploy app.py
```

```bash
curl -X POST https://queue.fal.run/<your-username>/bulletproof-inputs \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"image_url": "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"}'
```

## Cost & tuning

- Validation is CPU work: this runs on `M` (2 vCPUs / 2 GB) — see [fal pricing](https://fal.ai/pricing). In a real app these gates run *in front of* an expensive GPU endpoint, which is the point: rejecting a bomb costs a few CPU-milliseconds instead of a GPU worker.
- **Tune the caps to your model, not to this file.** The defaults mirror fal's production ingestion (100 MB download, ~89 MP decode); a video pipeline reasonably allows far more, a thumbnail service far less. Every cap here is a request field so you can probe behavior before hard-coding it.
- The pixel gate protects *co-tenants* too: with multiplexing, one OOM takes down every request on that worker — which is why it's a 422 and not a retry.

## Next steps

- **00.3 — Media in, media out**: the file plumbing these gates protect.
- **08.2 — Resilience**: OOM-safe serving, GPU health, retries and cancellation, once inputs are trustworthy.
- Concept docs: [errors](https://fal.ai/docs/documentation/model-apis/errors) · [inputs & outputs](https://fal.ai/docs/documentation/development/handle-inputs-and-outputs) · [retries](https://fal.ai/docs/documentation/serverless/reliability/retries)
