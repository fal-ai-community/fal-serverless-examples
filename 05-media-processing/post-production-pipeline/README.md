# Post-production pipeline

A generated video isn't a shippable video. Every genmedia product ends up building the same layer by hand: rendition ladders so playback doesn't buffer, poster thumbnails for the feed, captions muxed in for accessibility — then a queue and a webhook so the backend hears about it. This example is that layer as one deployable fal app: ffmpeg on a cheap CPU machine, with fal's queue as the queue.

> Catalog slot `05.1`.

## What you get

```json
{
  "renditions": [
    { "label": "1080p", "video": { "url": "https://v3b.fal.media/files/b/0aa50f61/f56bxDiIBd_mQREZl9C4b_….mp4", "content_type": "video/mp4" },
      "width": 1920, "height": 1080, "size_bytes": 2008643, "encode_ms": 1092 },
    { "label": "720p",  "video": { "url": "https://v3b.fal.media/files/b/0aa50f61/TZWp4jGh6-TIAqCJo0Y1e_….mp4" }, "width": 1280, "height": 720, "size_bytes": 967744, "encode_ms": 781 },
    { "label": "480p",  "video": { "url": "https://v3b.fal.media/files/b/0aa50f72/hJil0DJ-NvF0HWeMOJzVA_….mp4" }, "width": 854,  "height": 480, "size_bytes": 525707, "encode_ms": 579 }
  ],
  "thumbnails": [ { "url": "https://v3b.fal.media/files/b/0aa50f72/NIpPIey7MPWjb6uo7ma2Z_….jpg", "content_type": "image/jpeg" } ],
  "captions_vtt": { "url": "https://v3b.fal.media/files/b/0aa50f61/HTF8mp_nSNKWU4sId7nwt_captions.vtt", "content_type": "text/vtt" },
  "timings": { "download": …, "encode_wall": …, "thumbnails": …, "upload": … },
  "report": { "url": "https://v3b.fal.media/files/b/0aa50f61/Efmd1dZZ7zKC_V1mfeoZY_report.json", "content_type": "application/json" }
}
```

*Real output from a live deploy of this example (`rehan/post-production-pipeline`) — see the full transcript in Deploy below.*

Every rendition is an H.264 MP4 (AAC audio when the source has one) with the moov atom up front (`+faststart`), so the CDN URL streams immediately in a `<video>` tag; captions ride along as a selectable `mov_text` track plus a WebVTT sidecar for web players. The `report` is the pipeline's own measurement: probe summary, per-rendition encode times, download/upload timings, skipped rungs, and the exact ffmpeg version that did the work; the top-level `timings` field mirrors the stage timings in seconds.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # prints playground + API URLs; Ctrl-C to stop
```

The playground gives `video_url` a drag-and-drop upload widget; the defaults encode a 1080p/720p/480p ladder with two demo caption cues.

## How it works

**One endpoint, five stages.** Download → probe gate → parallel ladder → captions → thumbnails + uploads. The app is a stateless CPU processor; all the async machinery (queueing, status, webhooks) comes from the platform, not from app code — see Deploy.

**The environment is a Dockerfile, and that's the point.** ffmpeg is a system package, so this app pins it the honest way: a digest-pinned `python:3.11-slim` base plus apt's ffmpeg, built server-side by fal (no local Docker, no registry). Same pattern, same base digest as [01.1 same-model-two-ways](../../01-environments-containers/same-model-two-ways/) — read that example for why digests beat tags. The report echoes `ffmpeg -version` from inside the container, so provenance is verifiable per request.

**Bounded download, cached on `/data`.** `download_file` is SSRF-guarded and size-capped (300 MB here); sources land in a URL-hashed cache directory on [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage), so re-processing the same source (new ladder, new captions) skips the transfer — `report.timings_ms.download` goes to ~0.

**Probe first, encode later.** One `ffprobe` gates the request: still images and audio-only files are a typed 422 (`invalid_video`), durations over `MAX_DURATION_S` (600 s) are refused before they can burn minutes of encode time, and the probe's height feeds the **never-upscale rule** — requested rungs above the source height are skipped and listed in `report.ladder.skipped_no_upscale`, not silently stretched.

**The ladder runs in parallel.** Each rung is its own ffmpeg subprocess (`ThreadPoolExecutor` just babysits pipes — no GIL story), so wall-clock is roughly the slowest rung, not the sum: `report.timings_ms.encode_wall` vs the per-rendition `encode_ms` makes the difference visible. Encoding is H.264 `-crf 23 -preset veryfast` with `yuv420p` normalization — genmedia models love emitting 4:4:4 or 10-bit sources that browsers then refuse to play.

**Captions are validated before you pay for encoding.** SRT comes inline (`captions_srt`) or by URL (`captions_url`, at most one); the app converts it to WebVTT first — a malformed file fails as a fast typed 422 (`invalid_media`) before any rendition is encoded — then soft-muxes it into every MP4 as a selectable track.

**Outputs are typed toolkit objects on the fal CDN.** `File`/`Image` with explicit content types (`from_path` deliberately doesn't guess — an `application/octet-stream` MP4 won't stream inline). URLs are served from fal's CDN (`v3.fal.media` / `v3b.fal.media` shards); retention follows your account's media lifecycle. Per-request retention via the `x-fal-object-lifecycle-preference` header (see [00.3 media-in-media-out](../../00-getting-started/media-in-media-out/)) would additionally need the incoming FastAPI request forwarded to each upload call (`from_path(..., request=...)`) — this app skips that to keep the endpoint minimal.

## Deploy

```bash
fal deploy app.py
```

Now the part every product backend actually needs — submit through **fal's queue** and get the result **pushed** to you. The queue API for your private app is the same one the public Model APIs use, with your app id in the URL:

```bash
# submit (returns request handles immediately)
curl -X POST "https://queue.fal.run/<your-username>/post-production-pipeline" \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"video_url": "https://storage.googleapis.com/falserverless/example_outputs/pika/pika_i2v_v22_output.mp4"}'

# poll
curl "https://queue.fal.run/<your-username>/post-production-pipeline/requests/<request_id>/status?logs=1" \
  -H "Authorization: Key $FAL_KEY"
```

Or use the two reference scripts, which is the full product story:

```bash
# terminal 1 - a receiver that VERIFIES fal's Ed25519 webhook signatures
pip install httpx==0.28.1 cryptography==50.0.0
python webhook_receiver.py --port 8321
cloudflared tunnel --url http://localhost:8321   # or any public HTTPS route

# terminal 2 - submit with a webhook; polling stays on as the fallback
export FAL_KEY=<key_id>:<key_secret>
python client.py --app <your-username>/post-production-pipeline \
  --srt-file captions.srt --webhook https://<tunnel-host>/fal-webhook
```

Webhook deliveries are signed (Ed25519 over the request id, user id, timestamp, and the SHA-256 of the raw body — keys at `rest.fal.ai/.well-known/jwks.json`); `webhook_receiver.py` rejects anything that doesn't verify, checks timestamp freshness, and dedupes retries by `request_id`. Statuses are `IN_QUEUE → IN_PROGRESS → COMPLETED` — there is no `FAILED` status; failures arrive as `COMPLETED` with `error`/`error_type`, and the result URL returns the error body with a non-2xx code.

A real end-to-end run against a live deploy (`rehan/post-production-pipeline`) — client output on top, the receiver's independently-verified signature check below it:

```text
$ export FAL_KEY=<key_id>:<key_secret>
$ python client.py --app rehan/post-production-pipeline \
    --srt-file demo-captions.srt \
    --webhook https://<tunnel-host>/fal-webhook
queued: request_id=019fcfa8-8987-7370-8180-30514acb2d36
fal will POST the result to https://<tunnel-host>/fal-webhook (signed; see webhook_receiver.py)
polling too, as the fallback a real backend keeps:
  status: IN_QUEUE (queue position 0)
  status: IN_PROGRESS
  [app] INFO:     172.17.0.1:55870 - "POST / HTTP/1.1" 200 OK
  status: COMPLETED
{
  "renditions": [
    { "label": "1080p", "video": { "url": "https://v3b.fal.media/files/b/0aa50f61/f56bxDiIBd_mQREZl9C4b_….mp4" }, "width": 1920, "height": 1080, "size_bytes": 2008643, "encode_ms": 1092 },
    { "label": "720p",  "video": { "url": "https://v3b.fal.media/files/b/0aa50f61/TZWp4jGh6-TIAqCJo0Y1e_….mp4" }, "width": 1280, "height": 720,  "size_bytes": 967744,  "encode_ms": 781 },
    { "label": "480p",  "video": { "url": "https://v3b.fal.media/files/b/0aa50f72/hJil0DJ-NvF0HWeMOJzVA_….mp4" }, "width": 854,  "height": 480,  "size_bytes": 525707,  "encode_ms": 579 }
  ],
  "thumbnails": [ … 3 entries … ],
  "captions_vtt": { "url": "https://v3b.fal.media/files/b/0aa50f61/HTF8mp_nSNKWU4sId7nwt_captions.vtt" },
  "report": { "url": "https://v3b.fal.media/files/b/0aa50f61/Efmd1dZZ7zKC_V1mfeoZY_report.json" }
}
```

```text
$ python webhook_receiver.py --port 8321
listening on http://localhost:8321 - POST path can be anything
expose it, e.g.: cloudflared tunnel --url http://localhost:8321
VERIFIED status=OK request_id=019fcfa8-8987-7370-8180-30514acb2d36
  1080p: https://v3b.fal.media/files/b/0aa50f61/f56bxDiIBd_mQREZl9C4b_….mp4
  720p: https://v3b.fal.media/files/b/0aa50f61/TZWp4jGh6-TIAqCJo0Y1e_….mp4
  480p: https://v3b.fal.media/files/b/0aa50f72/hJil0DJ-NvF0HWeMOJzVA_….mp4
```

Also verified against the same deploy: a second identical submit got a **new** `request_id` (`019fcfa9-ef6b-…`) and the receiver logged it as a fresh `VERIFIED` line, not a duplicate (dedupe is per-`request_id`, and this was a different request); its `report.json` showed `"cache_hit": true` with `timings_ms.download` of 48ms, since the source was already on `/data` from the first run. A hand-forged delivery (`curl -X POST localhost:8321 -d '{}'`) was rejected: `REJECTED: missing header 'x-fal-webhook-request-id'` and HTTP 401.

## Cost & tuning

- **CPU machines bill per second, per machine.** This app runs on `L`; the whole cost story is throughput — how many seconds of runner time a job needs. The report measures exactly that (`encode_wall`, per-rung `encode_ms`, `download`, `upload`) on your actual workload; tune from those numbers, not from folklore. See [machine types & pricing](https://fal.ai/pricing).
- **Ladder width ↔ vCPUs.** Rungs encode in parallel, so wall-clock ≈ the slowest rung — until the rungs starve each other for cores. `report.toolchain.vcpus` tells you what the machine actually has; if `encode_wall` approaches the *sum* of `encode_ms` values, the ladder is core-bound: use a bigger machine or fewer rungs.
- **`keep_alive=60`** keeps the worker warm for a minute after each job — post-production traffic is bursty (a batch of generations finishes, a batch of transcodes follows), and a warm worker skips the container cold start for the rest of the batch. Raise it if your queue is steadily busy; it bills like any other runner second.
- **The queue is free capacity management.** Submitting via `queue.fal.run` instead of a synchronous call means bursts wait in line instead of timing out, and webhook delivery means no client holds a connection open for a minute-long job.
- **The `/data` cache is append-only.** Every distinct source URL keeps up to 300 MB on [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage) forever, and a hard-killed download can strand a `.fal_download.tmp.*` partial next to it (the app never treats those as cache hits, but nothing deletes them either). When self-hosting, add a cleanup policy — a cron that prunes by mtime is plenty.

## Next steps

- **Wire this into a product backend**: `client.py` and `webhook_receiver.py` are the skeleton — your app server submits through the queue, and the verified webhook handler updates your database.
- **The same archetype works for images**: an AVIF/WebP/resize ladder is this pipeline with different ffmpeg invocations — the queue, webhook, and container story carry over unchanged.
- **HLS is the natural extension**: per-rung segment playlists + a master playlist. The ladder here is the hard part; HLS is packaging (`-f hls`) plus playlist URL rewriting for the CDN.
- Concept docs: [queue](https://docs.fal.ai/model-endpoints/queue) · [webhooks](https://docs.fal.ai/model-endpoints/webhooks) · [container apps](https://fal.ai/docs/documentation/development/use-custom-container-image) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage)
