# 05.1 Post-production pipeline — design

**Ticket:** [SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282) · catalog slot `05.1` · tier 05 Media processing · `net-new`
**Scope decision (this session):** repo example only — the fal-docs page (mdx + snippet + nav + gallery card) is a follow-up.
**Approved:** 2026-08-04 by Rehan (architecture A; MP4 ladder; soft-mux captions + VTT sidecar; client.py + verified webhook receiver).

## Problem

Every genmedia product that generates video re-builds the same post-production layer by hand: rendition ladders for delivery, poster thumbnails, captions muxed in. This example is the reference for the **CPU proxy archetype** — ffmpeg on fal CPU machines, with fal's platform queue and webhooks providing the async layer a product backend needs. It is also the verified reference for `queue.fal.run` + webhook usage against a **private serverless app** (the existing queue/webhook docs only cover public Model API endpoints — the ticket's named risk).

## Architecture (approved: option A)

One `fal.App` with a single `/` endpoint that performs the whole job per request:

```
bounded download → ffprobe gate → parallel ffmpeg ladder → captions → thumbnails → CDN uploads
```

The app stays a stateless CPU processor; **fal's queue is the queue**. Async submission, status polling, and webhook completion all come from `queue.fal.run` against the registered deploy. CI smokes the synchronous path on an ephemeral deploy (queue URLs only exist for registered apps).

Rejected: stage endpoints composed by the client (fragments the story, overlaps 06.6); self-orchestrated per-rendition fan-out (overlaps 05.6, blows the S/M budget).

## Folder

`05-media-processing/post-production-pipeline/`

| File | Purpose |
|---|---|
| `app.py` | The fal.App — single `/` endpoint, five-stage pipeline |
| `Dockerfile` | Digest-pinned `python:3.11-slim` + apt `ffmpeg` (01.1 pattern, same base digest) |
| `client.py` | Queue submit / status / result against `queue.fal.run`, `--webhook` flag |
| `webhook_receiver.py` | Stdlib HTTP server verifying fal's ED25519 webhook signatures |
| `requirements.txt` | Exact mirror of the Dockerfile's pip installs (`fal==1.78.2`) |
| `test_app.py` | Live smoke tests run by `harness/check.py` |
| `README.md` | §4-A template order; inlines fal-CDN and cost guidance (concept pages don't exist yet) |

Tier README table gets the slot ↔ folder mapping.

## Input schema

- `video_url: str` — source video; `_url` suffix gives the playground an upload widget. `Field(examples=[...])` points at the CI sample clip (see Open items).
- `ladder: list[Literal[2160, 1440, 1080, 720, 480, 360]] = [1080, 720, 480]` — rung heights; deduped and sorted descending in the endpoint. **Never upscales:** rungs above source height are skipped and recorded in the report, not errored.
- `captions_srt: str | None = None` — inline SRT content. Its `examples` carries a tiny 2-cue SRT so the harness-generated smoke payload exercises muxing hermetically (no external SRT dependency in CI).
- `captions_url: str | None = None` — hosted SRT alternative. Exactly one of the two caption fields may be set; both → typed 422.
- `thumbnail_count: int = 3` (ge=1, le=10) — frames at `duration · (i+1) / (count+1)`, dodging the black first frame.

## Output schema

- `renditions: list[Rendition]` — `Rendition = {label: "720p", video: fal.toolkit.Video, width: int, height: int, size_bytes: int, encode_ms: int}` (Video confirmed present in fal 1.78.2: `fal.toolkit.video.video.Video(File)`, so `File.from_path` semantics apply).
- `thumbnails: list[Image]` — JPEG posters.
- `captions_vtt: File | None` — normalized WebVTT sidecar for web players.
- `report: File` — JSON provenance: ffprobe summary (duration, codec, dimensions, fps), ffmpeg version string, per-stage timings, per-rendition stats, skipped rungs, cache hit.

## Pipeline stages

1. **Bounded download** — `fal.toolkit.download_file` (SSRF-guarded, `filesize_limit=300` MB) into a `/data` cache keyed by URL hash (00.3 pattern). `DownloadError` → `FieldException(field="video_url")` (typed 422).
2. **Probe & gate** — `ffprobe -print_format json`. No video stream → 422. Duration above `MAX_DURATION_S = 600` → 422, with a "raise when self-hosting" comment. Probe feeds never-upscale logic and the report.
3. **Transcode ladder** — parallel ffmpeg subprocesses via `ThreadPoolExecutor` (external processes; threads are free). Per rung: H.264, CRF 23, preset `veryfast`, `scale=-2:<height>`, AAC 128k **only if the source has audio**, `-movflags +faststart` (moov atom first — what makes MP4s stream from a CDN).
4. **Captions** — validate by converting SRT → WebVTT with ffmpeg (failure → 422 on the captions field); soft-mux as `mov_text` track into every rendition; upload the VTT sidecar.
5. **Thumbnails + uploads** — JPEG frames at the computed timestamps, scaled to `scale='min(1280,iw)':-2` (≤1280 px wide, aspect preserved, never upscaled); all outputs to fal CDN via toolkit types.

**Error taxonomy:** ffmpeg failures caused by user media/captions → 422 with stderr tail in the message; genuine encoder/infra failures → 500 so the queue marks the job failed.

## Environment & cost posture

- `machine_type = "L"` (catalog's call — multi-rendition x264 wants the vCPUs; the parallel-ladder ↔ vCPU relationship is the Cost & tuning centerpiece).
- `keep_alive = 60`, consistent with existing examples; README argues the queue-worker burst case for raising it.
- `app_auth = "private"`.
- Dockerfile: same digest-pinned `python:3.11-slim` base as 01.1, apt `curl ca-certificates ffmpeg` (Debian bookworm ffmpeg 5.1.x; version echoed into the report), `pip install fal==1.78.2` last. `App.requirements = []` — the image carries everything; `requirements.txt` mirrors the Dockerfile's pip installs per repo convention.

## Client & webhook verification

- `client.py` — raw `httpx` on purpose (the exact URLs/headers for private apps are the doc gap being verified): `POST https://queue.fal.run/{owner}/{app}/?fal_webhook=<url>` with `Authorization: Key $FAL_KEY`; poll `.../requests/{id}/status`; fetch result; print a readable transcript. Flags: `--video-url`, `--srt-file` (local file → inline `captions_srt`), `--ladder`, `--webhook`. README notes the `fal_client` equivalence.
- `webhook_receiver.py` — stdlib `http.server` + `cryptography`: verify fal's ED25519 webhook signature against the published JWKS, check timestamp freshness, reject tampered payloads, print verified completions. README includes a `cloudflared` tunnel one-liner for a laptop demo.
- Client-script deps (`httpx`, `cryptography`) are documented in the scripts' headers/README, not in `requirements.txt` (that file mirrors the app image only).

## CI tests (`test_app.py`, sync against the ephemeral deploy)

1. **Full pipeline** — sample clip + inline SRT: assert rendition labels/heights, non-empty CDN files, thumbnail count, VTT present, report JSON shape.
2. **Never-upscale** — ladder `[2160, 480]` on the 1080p source → exactly one rendition + reported skip.
3. **Typed 422s** — non-video URL; malformed SRT; both caption fields set — direct `httpx` POSTs asserting status 422 and the field name (00.4 pattern).

The ~10 s sample keeps every run inside the 120 s smoke window; `/data` caching makes repeat downloads near-free across tests.

## Registered-deploy validation (DoD's "validated deployment")

`fal deploy` the app (e.g. `rehan/post-production-pipeline`), run `client.py` with a tunneled `webhook_receiver.py`, and capture the verified queue → webhook transcript. That evidence closes the ticket's "queue/webhook docs don't cover private serverless apps" risk and feeds the README Deploy section. Per the 2026-07-21 descope, the DoD's "measured numbers" line reads as "passes the CI deploy-check" — no performance claims in the README.

## README (§4-A order)

Problem ("every genmedia product builds this by hand") → output first (response JSON + thumbnail) → run it → how it works (five stages, never-upscale, faststart, container rationale) → deploy (queue + webhook transcript) → cost & tuning (CPU machine sizing XS/S/M/L, billable-unit metering, keep_alive math, ladder-width ↔ vCPU tradeoff — inlined here because the Optimizing Costs concept page doesn't exist yet) → next steps (05.3 image variants, 06.2 queue-into-backend, HLS as the natural extension).

## Open items (resolve during implementation)

1. **Sample clip hosting (CI-critical).** CI downloads the sample every run. Preferred: a ~10 s, 1080p, CC-licensed clip (e.g. Big Buck Bunny cut) on the fal-controlled `storage.googleapis.com/falserverless` bucket — needs Rehan/fal to upload, or scout the bucket for an existing suitable asset. Interim fallback: test-videos.co.uk URL (flagged as a CI-stability risk, swap before merge if possible).
2. **Webhook signature details.** Confirm exact signed-message construction + JWKS URL from fal docs/source before writing `webhook_receiver.py`; the verified receiver is the deliverable, so this must come from source, not memory.
3. ~~`fal.toolkit.Video` availability~~ — resolved: present in fal 1.78.2.

## DoD mapping (SERV-1282)

| DoD line | How this design meets it |
|---|---|
| §4-A page structure | README follows the template order (docs-page mdx is follow-up scope) |
| Snippet + nav + gallery card | fal-docs follow-up session |
| Runnable code validated on real deployment | Registered deploy + queue/webhook transcript + CI ephemeral deploy-check |
| Measured numbers | Descoped 2026-07-21 → CI deploy-check green |
| Source linked | This repo, `05-media-processing/post-production-pipeline/` |
