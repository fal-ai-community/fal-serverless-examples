# 04.7 · 3D progressive rendering (SAM-3D) — rework design

**Ticket:** [SERV-1319](https://linear.app/features-and-labels/issue/SERV-1319) · Tier 04 — Genmedia patterns · Keep/refresh
**Date:** 2026-08-10 · **Status:** approved by Rehan (scope: full rebase; viewer: single-file; weights: official gated repo)

## Problem

The current example (docs page `examples/video-generation/deploy-3d-progressive-rendering.mdx`, source in `rehan-remade/Manifold`) streams SAM-3D Objects reconstruction progress over SSE — voxels appearing live during two diffusion stages, then a final GLB. The pattern is strong and is the docs streaming page's own showcase, but a four-lens audit (code, docs, ecosystem, repo-fit; 2026-08-10) found:

- **Fragile personal dependency chain**: unpinned personal model fork (`rehan-remade/sam-3d-objects` at HEAD behind a cache-bust ARG), personal HF weights mirror (`jetjodh/sam-3d-objects`, mutable `main`, SAM License optics risk), personal endpoint `rehan/sam-3d-stream`, personal Vercel demo. Violates the ticket DoD (source in the examples repo) and repo pinning conventions.
- **Real SSE plumbing bugs**: client disconnect wedges the GPU worker permanently (blocking `put()` on a bounded queue, no cancellation, no try/finally); backpressure couples GPU throughput to the slowest client; heartbeat fires at 180s — past typical 60–100s proxy idle timeouts — and as a data event, not an SSE comment; full tracebacks streamed to clients; appearance-voxel colors cast to uint8 without 0–1→0–255 scaling (streams black); per-frame bounding-box renormalization makes the cloud jitter; entire GLB base64'd into one SSE event; dead input fields (`prompt`/`point_prompts`/`box_prompts`) promise auto-segmentation that never runs; Pydantic v1 idioms.
- **Ecosystem moved on**: Meta's official release is live — code at `facebookresearch/sam-3d-objects`, gated checkpoints at `facebook/sam-3d-objects` (SAM License, Nov 2025). The official generator base class exposes a public `generate_iter(...)` yielding after every solver step for **both** diffusion stages, so streaming likely no longer needs a model fork. fal now hosts an official **non-streaming** `fal-ai/sam-3/3d-objects` endpoint — progressive streaming is this example's differentiator.
- **Docs page vs §4-A**: no Cost & tuning (shows `keep_alive=600` on H100, never priced), no Next steps, zero concept cross-links (ticket requires Streaming Endpoints + file-handling), how-it-works stranded after Deploy, frontend-first run path with two API keys of removable friction (Groq, z-image hop), and two broken frontend snippets (fake `instancedMesh` instancing; SSE parse without chunk buffering).

## Decisions taken (with owner sign-off)

1. **Full rebase** on the official SAM-3D release, spike-gated (see below). Fork retired if the spike passes.
2. **Single-file `viewer.html`** ships in the example folder as the visual client; the full Next.js/R3F app moves to a linked companion repo.
3. **Weights from the official gated `facebook/sam-3d-objects`**, pinned revision, authed via an `HF_TOKEN` fal secret; the one-time HF access request is documented in the README. No fal-owned mirror.

## Spike gate (first implementation step)

On a real H100 deploy, verify:

- `generate_iter(...)` on the official pipeline's `ss_generator` and `slat_generator` yields per-step latents usable for progressive output (drive `sample_sparse_structure` / `sample_slat` via direct `generate_iter` loops in a thin subclass/driver — no fork).
- Intermediate stage-1 latents can be decoded to voxel coords at acceptable per-step cost (decode throttled to every N steps).
- End-to-end latency of a full generation (informs the `/` endpoint vs the harness's 120s smoke window; distillation flags / `stage1_only` / step counts are the levers, `HARNESS_REQUEST_TIMEOUT` bump in `check.yml` is the fallback).
- The gated `HF_TOKEN` download path works end-to-end under a fal secret, weights land in `FAL_PERSISTENT_DIR`, and worst-case `setup()` stays under the harness's 600s health window on a warm `/data`.
- Whether the xformers/triton monkey-patch is still needed on current pins (if yes: pin triton, comment the upstream issue, fail loudly at setup when the patch can't apply; if no: delete).

**Fallback if the spike fails:** the existing fork, pinned to a commit SHA and moved to an org account, with the callback API documented as the delta vs upstream. Everything else in this design is unchanged.

## What ships

`04-genmedia-patterns/3d-progressive-rendering/` (short kebab-case, no slot prefix):

| File | Role |
|---|---|
| `app.py` | One `fal.App` subclass, `kind="container"`, explicit `app_auth`, H100 |
| `Dockerfile` | Separate file (not inline string) — content-keyed build cache, reviewable diffs |
| `viewer.html` | Single-file Three.js progressive voxel viewer, no build step |
| `test_app.py` | Static-safe unit tests + harness SSE integration test |
| `requirements.txt` | Hand-mirror of the Dockerfile's pip pins (container app ⇒ `App.requirements` empty) |
| `README.md` | §4-A order |

## Endpoints

**`/` (harness smoke + sync users)** — plain JSON request/response: run the pipeline, return `{glb_url, gaussian_splat_url, seed}`. Every required field carries `Field(examples=[...])` (harness builds its smoke payload from `examples[0]`). Budgeted by the spike to fit the 120s smoke window.

**`/stream` (the lesson)** — SSE at exactly that path (`fal_client.stream()` appends it). Documented fal pattern: `StreamingResponse`, `data:` frames, queue/thread bridge. `x-fal-sse-last-event-as-body: 1` set as a one-line optional hardening so plain sync calls get the terminal event — framed as optional, not the pattern.

**Input schema (both endpoints):** `image_url` (required, `examples=[...]`), `mask_url` (optional, single), `seed`, `stream_geometry_every` / `stream_colors_every` (stream only). Deleted: `prompt`, `point_prompts`, `box_prompts`, the backward-compat validator, the unused `response` param. Pydantic v2 idioms (`model_validator`, `default_factory`); every field description matches actual behavior. Groq prompt-enhancement and the z-image text→image hop are cut from the core flow; the README notes text→3D as an optional pre-step linking the z-image model page (no latency claims).

## Streaming core — design rules the code teaches

1. **Progress frames are disposable; terminal events are guaranteed.** Bounded queue; callbacks use `put_nowait` with drop-or-coalesce-to-latest on overflow; only `complete`/`error` get blocking delivery. Called out in README as *the* design decision of GPU progress streaming.
2. **Every stream ends in exactly one terminal event.** `complete` (asset URLs) or `error` (sanitized code + message, traceback logged server-side only). The drain loop checks the worker future on exit paths so a died worker still yields a terminal `error`.
3. **Cancellation is real.** try/finally around the SSE generator; on client disconnect set a `threading.Event` the step loop checks (aborts the pipeline via exception); executor joined in the finally.
4. **Heartbeats are SSE comments** (`: keepalive`) every ~15s of queue silence — under proxy idle timeouts, invisible to event parsers.
5. **Validate before streaming.** Input image fetched/validated before returning the `StreamingResponse`, so field errors are honest 422s (`FieldException`), never error events on a 200.
6. **Small frames inline, big assets by URL.** Voxel/mesh-preview frames stream inline; GLB + Gaussian splat upload via `fal.toolkit File` (multipart) and arrive as URLs in `complete`. No inline GLB event.

**Event types:** `loading` → `geometry` (stage 1, throttled) → `appearance` (stage 2, throttled) → `mesh_preview` (vertex-colored, decimated/size-capped so it stays a frame, not an asset) → `complete` | `error` (+ comment heartbeats).

## Binary encoding — kept as centerpiece, fixed

- 6 bytes/voxel (`uint8 x,y,z,r,g,b`), base64 in the JSON event, self-describing `encoding` field. README shows the byte math vs naive JSON.
- **Raw grid indices, not per-frame renormalization**: stage-1 coords are indices in a fixed ≤256³ grid, so uint8 is *lossless* and the cloud is frame-stable. The grid size is part of the event contract.
- **One shared encoder, explicit color contract**: float 0–1 in, scaled to 0–255 once. No range-sniffing heuristics. Fixes the black-appearance-voxel bug and the voxel/mesh encoder asymmetry.

## Container & reproducibility

Per `same-model-two-ways` conventions: digest-pinned CUDA base (`FROM ...@sha256:`), Python 3.11 on PATH (cloudpickle match), `curl` + `ca-certificates` present, no ENTRYPOINT/CMD, `fal==1.78.2` installed last. Every pip dep `==`-pinned; official `sam-3d-objects` cloned at a pinned SHA; prebuilt wheels (pytorch3d, flash-attn) pinned by exact URL with sha256 recorded and a README note on why prebuilt (30+ min source builds) and the trusted-source options. No cache-bust ARGs (SHA pinning obviates them), no double torch install, no `--force-reinstall`. `requirements.txt` hand-mirrors the Dockerfile pins with a header comment naming the Dockerfile authoritative.

`setup()`: `snapshot_download("facebook/sam-3d-objects", revision=...)` pinned to a specific commit hash (chosen during the spike) into `FAL_PERSISTENT_DIR` (no deprecated kwargs), token from the fal secret. Single documented seed mechanism (whichever `pipeline.run`/driver actually honors — spike determines).

App config: `machine_type` H100 (capacity-fallback list only if the spike shows VRAM headroom — open question), `keep_alive=60` (repo norm; README's Cost & tuning explains raising it), `max_concurrency=1`, explicit `request_timeout` sized from measured generation time.

## Testing (sets the repo's SSE precedent — tier 03 is still empty)

- **Static-safe unit tests** (run free on every PR, CPU-only, no secrets): encoder round-trip (encode → decode → compare coords/colors), color-contract scaling, event-framing helpers. Module-level imports in `app.py`/`test_app.py` limited to fal/pydantic/httpx/pytest/stdlib; heavy imports live inside `setup()`/endpoints.
- **Harness integration** (under `HARNESS_APP_URL`): `/` smoked by the harness itself; `test_app.py` consumes `/stream` with raw `httpx.stream(...)` + `iter_lines()` (auth headers rebuilt via `fal.sdk.get_credentials().to_headers()`), asserting: `content-type: text/event-stream`; monotonic `progress`; all `geometry` events precede all `appearance` events; ≥1 voxel frame decodes to a non-empty cloud; exactly one terminal event; `complete` carries a GLB URL whose fetched bytes start `b"glTF"` and a splat URL. Total pytest wall-clock within the 600s harness cap.

## CI budget

Pre-warm fal's Dockerfile-content-keyed build cache with an out-of-band deploy before opening the PR, so CI's ephemeral deploy reuses the image (the 900s URL timeout cannot survive a cold 30-min container build). If cold-build coverage is ever wanted, bump `HARNESS_URL_TIMEOUT` and the job `timeout-minutes` in `check.yml` explicitly in the same PR. Billing note: PENDING/DOCKER_PULL states are unbilled; `setup()` time is billed.

## viewer.html

One file, no build step: reader pastes their deployed endpoint URL + key, hits generate, watches voxels fill in. Correct THREE `InstancedMesh` usage (`setMatrixAt`/`setColorAt` + `instanceMatrix.needsUpdate`) and a correct buffered SSE parser (text buffer split on `\n\n`; ignores comment lines) — deliberately fixing the two broken snippets from the old docs page, and serving as the reference client for the event contract. Warns that pasting a key into a page is for local development only.

## README (§4-A) & docs-repo follow-up

README order: problem → output first (GIF/video of progressive voxels) → run it (`fal run` + curl + viewer.html) → how it works (event contract → queue semantics → encoding → container notes, linking Streaming Endpoints and file-handling docs instead of re-explaining) → deploy → **cost & tuning** (H100 $/s worked example: cold start + one generation + keep_alive tail; `stream_*_every` as payload/overhead knobs; step-count/distillation as latency knobs; machine-type rationale) → next steps (03.1 SSE example, 05.2 LOD processor, Streaming Endpoints + file-handling docs, positioning vs hosted `fal-ai/sam-3/3d-objects` — "hosted for batch, deploy this for progressive" — and TRELLIS.2-4B / Hunyuan3D 3.x as quality-first alternatives). SAM License attribution included. No unpinned installs anywhere; no unbacked performance claims (measurement layer descoped 2026-07-21).

**Companion fal-docs PR (after this repo lands, branched from current origin/main):** refresh the existing page backend-first in the same §4-A order; replace `rehan/sam-3d-stream` promotion with deploy-your-own; cut Groq; fix or remove the broken frontend snippets; link this repo's folder as the source; move the full frontend app to the companion repo link. Page placement / nav / gallery card / snippet follow SERV-1266's IA conventions (read it before creating pages). DoD's "measured numbers" reads as CI deploy-check per the 2026-07-21 descope.

## Open questions (non-blocking)

1. Does an org-owned hosted endpoint replace `rehan/sam-3d-stream` on the docs page, or does the page go deploy-your-own only?
2. H100-only vs `machine_type` capacity-fallback list — spike informs VRAM headroom.
3. Where the companion frontend repo lives (Manifold refreshed in place vs a fal-ai-community home).

## DoD mapping (SERV-1319)

- §4-A page structure → README + docs-page refresh above.
- Snippet + nav + gallery card with `dateAdded` → companion fal-docs PR (per SERV-1266).
- Runnable code validated on a real deployment → harness deploy-check + pre-warmed container build.
- "Measured numbers via the harness" → CI deploy-check (2026-07-21 descope).
- Source linked in fal-examples → this folder.
