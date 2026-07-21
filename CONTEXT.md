# Project context — read me first

This file onboards anyone (human or AI assistant) into the **fal Serverless Examples & Onboarding Revamp**. It is the single entry point; follow the links for depth. Keep it updated as decisions close.

## Mission

Close fal's examples/onboarding gap vs Modal, Replicate, Baseten, and RunPod by rebuilding the serverless examples as a **numbered curriculum (tiers 00–08, ~40 entries)** on current models — every example ending in a production deploy that **CI re-verifies with real ephemeral deploys** (deploy + smoke + tests). This repo holds the runnable source; the docs pages that walk through each example live in [fal-docs](https://github.com/fal-ai/fal-docs).

## Current repo state (read this before assuming anything)

Live at **[github.com/rehan-remade/fal-serverless-examples](https://github.com/rehan-remade/fal-serverless-examples)** (private, git user `rehan-remade`) since 2026-07-20. PR #1 (merged 2026-07-21) landed the deploy-check harness (`harness/check.py` + `.github/workflows/check.yml`) and the first example (`00-getting-started/hello-fal/`, also live as a registered app `rehan/hello-fal`). The **`FAL_KEY` Actions secret is not set yet** — CI's guard step skips deploy-checks until it exists; the weekly cron stays commented until the first green run.

Later the repo transfers to [fal-ai-community](https://github.com/fal-ai-community) per [SERV-1279](https://linear.app/features-and-labels/issue/SERV-1279) — after the transfer, never create a same-named repo on the personal account (it kills GitHub's redirect), and re-add `FAL_KEY` + re-check org Actions policies.

Project owner: Rehan Sheikh (rehan@fal.ai).

## Where everything lives

| What | Where |
|---|---|
| Design doc (catalog §4, audit §2, template §4-A) | [Notion](https://www.notion.so/featuresandlabels/Serverless-Examples-Onboarding-Revamp-397230f2245e8189a945e3a5e6b1b22e) |
| Execution board (58 tickets, milestones = Foundations → Tier 00…08) | [Linear project](https://linear.app/features-and-labels/project/serverless-examples-and-onboarding-revamp-ab1f7491f6e7) |
| Execution-plan matrix (40-entry gap map, defects, build order) | [Artifact](https://claude.ai/code/artifact/8a6d27f8-c63e-4d41-b2b3-115a994a28d9) — WebFetch works from Rehan's Claude sessions; humans need a share link. The load-bearing part (§4-A page anatomy) is mirrored in `templates/example/README.md`, so the artifact/Notion aren't required for page structure |
| Docs repo (pages live here, Mintlify) | [fal-ai/fal-docs](https://github.com/fal-ai/fal-docs) — private, needs fal-ai org access |
| Predecessor demo repo (being superseded) | [fal-ai-community/fal-demos](https://github.com/fal-ai-community/fal-demos) |
| Live docs | [fal.ai/docs](https://fal.ai/docs) |

Linear + Notion MCP tools are typically connected — load them via ToolSearch to read tickets/design doc directly.

## How the Linear board is organized

- **Milestones = catalog order**: `Foundations` → `Tier 00 · Getting started` → … → `Tier 08 · Production ops`. Foundations holds pre-catalog work: link hygiene ([SERV-1264](https://linear.app/features-and-labels/issue/SERV-1264)), page template + card data ([SERV-1265](https://linear.app/features-and-labels/issue/SERV-1265)), IA scaffolding ([SERV-1266](https://linear.app/features-and-labels/issue/SERV-1266)), and the **numbers harness ([SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310) — build this first; every entry's numbers depend on it; same ticket as catalog entry 08.4, because the harness IS that example: build it in Foundations, document it as 08.4)**. Note: the re-branch ([SERV-1262](https://linear.app/features-and-labels/issue/SERV-1262)) and descoped lint-fix ([SERV-1263](https://linear.app/features-and-labels/issue/SERV-1263)) tickets were **archived in Linear on Jul 20** — treat them as retired unless Rehan says otherwise (the lint ticket's defect list remains valid reference).
- **Labels**: `tier-00`…`tier-08` (catalog tier) · `ready-now` (8: writable from existing docs) / `needs-deploy` (20: needs validation deployment) / `net-new` (12: new code/pattern) · `decision` (gating tickets).
- **Blocked-by relations** wire 12 gated entries to their decision tickets — "blocked" is filterable and auto-clears.
- Every entry ticket carries: work needed, risks, existing docs assets, missing reference docs, and a **Definition of Done** (§4-A page structure, snippet + nav + gallery card with dateAdded, validated deployment, measured numbers via the harness, source linked here). **Note:** per the 2026-07-21 descope (fact #2 below), read "measured numbers via the harness" in DoDs as "passes the CI deploy-check".

## The full catalog (slot → ticket)

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `00.1` | Hello, fal: one app, run three ways (fal run → hot-reload → deploy) | `needs-deploy` · blocked: [SERV-1267](https://linear.app/features-and-labels/issue/SERV-1267) (confirm hot-reload in CLI) | [SERV-1284](https://linear.app/features-and-labels/issue/SERV-1284) |
| `00.2` | Your first image API (FLUX-class, warmup, playground, clients); replaces SDXL tutorial | `needs-deploy` · blocked: [SERV-1268](https://linear.app/features-and-labels/issue/SERV-1268) (FLUX weight licensing decision) | [SERV-1293](https://linear.app/features-and-labels/issue/SERV-1293) |
| `00.3` | Media in, media out (files, CDN, persistent storage) | `needs-deploy` · blocked: [SERV-1269](https://linear.app/features-and-labels/issue/SERV-1269) (Files UI / ACL confirmation) | [SERV-1302](https://linear.app/features-and-labels/issue/SERV-1302) |
| `00.4` | Bulletproof inputs (typed 422s, decompression-bomb gate, SSRF-safe downloads) | `net-new` | [SERV-1309](https://linear.app/features-and-labels/issue/SERV-1309) |
| `01.1` | Same model, two ways: fal runtime vs your own Dockerfile | `needs-deploy` · blocked: [SERV-1270](https://linear.app/features-and-labels/issue/SERV-1270) (mutable-tag re-pull semantics) | [SERV-1285](https://linear.app/features-and-labels/issue/SERV-1285) |
| `01.2` | Migrations (external Docker, Modal, Replicate, RunPod) | `ready-now` | [SERV-1291](https://linear.app/features-and-labels/issue/SERV-1291) |
| `01.3` | Dev → staging → prod (environments, secrets, promotion, rollback) | `ready-now` | [SERV-1299](https://linear.app/features-and-labels/issue/SERV-1299) |
| `02.1` | Cut cold starts by Nx (FlashPack + /data + compiled caches, measured before/after) | `needs-deploy` · blocked: [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271) (FlashPack public-wheel parity) | [SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286) |
| `02.2` | Tune autoscaling for bursty traffic (keep_alive, concurrency, cost math) | `ready-now` | [SERV-1294](https://linear.app/features-and-labels/issue/SERV-1294) |
| `02.3` | Serve a 20+ GB video model (parallel loading, GPU sizing reasoning) | `needs-deploy` | [SERV-1303](https://linear.app/features-and-labels/issue/SERV-1303) |
| `02.4` | Free-lunch optimizations, with numbers (prompt cache, FA3 one-liner, GPU branching) | `net-new` · blocked: [SERV-1272](https://linear.app/features-and-labels/issue/SERV-1272) (IP disclosure decision) | [SERV-1308](https://linear.app/features-and-labels/issue/SERV-1308) |
| `03.1` | Stream progressive results with SSE (preview frames during long generations) | `ready-now` | [SERV-1280](https://linear.app/features-and-labels/issue/SERV-1280) |
| `03.2` | Realtime WebSocket endpoint (interactive-latency editing) | `needs-deploy` | [SERV-1283](https://linear.app/features-and-labels/issue/SERV-1283) |
| `03.3` | Real-time video-to-video (re-theme from YOLO to genmedia style transfer) | `net-new` | [SERV-1289](https://linear.app/features-and-labels/issue/SERV-1289) |
| `03.4` | Real-time world model over WebRTC (Matrix-Game / WMA) | `ready-now` | [SERV-1295](https://linear.app/features-and-labels/issue/SERV-1295) |
| `04.1` | ComfyUI workflow → production API (real workflow, custom nodes, current models) | `needs-deploy` · blocked: [SERV-1274](https://linear.app/features-and-labels/issue/SERV-1274) (coordinate with Alex, PR #404) | [SERV-1292](https://linear.app/features-and-labels/issue/SERV-1292) |
| `04.2` | Text-to-video on a current model (LTX-class, 720p+, endpoint + client + numbers) | `needs-deploy` | [SERV-1300](https://linear.app/features-and-labels/issue/SERV-1300) |
| `04.3` | Serve thousands of per-user LoRAs with routing hints (avatar/personalization) | `needs-deploy` · blocked: [SERV-1273](https://linear.app/features-and-labels/issue/SERV-1273) (provide_hints at scale) | [SERV-1306](https://linear.app/features-and-labels/issue/SERV-1306) |
| `04.4` | Train a LoRA → serve it (closes the WAN training loop) | `needs-deploy` | [SERV-1312](https://linear.app/features-and-labels/issue/SERV-1312) |
| `04.5` | Multi-GPU inference on a frontier video model (fal.distributed) | `needs-deploy` | [SERV-1316](https://linear.app/features-and-labels/issue/SERV-1316) |
| `04.6` | TTS (Kokoro) + text-to-music (DiffRhythm); refresh models | `needs-deploy` | [SERV-1318](https://linear.app/features-and-labels/issue/SERV-1318) |
| `04.7` | 3D progressive rendering (SAM-3D) | `needs-deploy` | [SERV-1319](https://linear.app/features-and-labels/issue/SERV-1319) |
| `05.1` | Post-production pipeline (transcode ladder, thumbnails, captions; ffmpeg on CPU machines + queue + webhooks) | `net-new` | [SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282) |
| `05.2` | LOD processor for 3D assets (GPU-generate → CPU-refine, packaged glTF) | `net-new` | [SERV-1288](https://linear.app/features-and-labels/issue/SERV-1288) |
| `05.3` | Image variant factory (AVIF/WebP/resize ladders to CDN) | `net-new` · blocked: [SERV-1275](https://linear.app/features-and-labels/issue/SERV-1275) (CDN retention confirmation) | [SERV-1296](https://linear.app/features-and-labels/issue/SERV-1296) |
| `05.4` | Audio finishing (loudness normalization, waveform peaks) | `net-new` | [SERV-1301](https://linear.app/features-and-labels/issue/SERV-1301) |
| `05.5` | Provenance & safety stamping (watermark / C2PA on every output) | `net-new` · blocked: [SERV-1276](https://linear.app/features-and-labels/issue/SERV-1276) (C2PA certificate story) | [SERV-1307](https://linear.app/features-and-labels/issue/SERV-1307) |
| `05.6` | Batch fan-out over a bucket (CPU stages on CPU machines) | `net-new` | [SERV-1313](https://linear.app/features-and-labels/issue/SERV-1313) |
| `06.1` | Chain models with Workflows (expand → generate → upscale → safety, one endpoint) | `needs-deploy` | [SERV-1290](https://linear.app/features-and-labels/issue/SERV-1290) |
| `06.2` | Async queue + webhooks into a product backend (Next.js submits, webhook updates DB) | `needs-deploy` | [SERV-1298](https://linear.app/features-and-labels/issue/SERV-1298) |
| `06.3` | Multi-app routing (one proxy endpoint, many apps) | `ready-now` | [SERV-1305](https://linear.app/features-and-labels/issue/SERV-1305) |
| `06.4` | Built on Serverless live demo app (Together-style: hosted app + repo + page) | `net-new` | [SERV-1311](https://linear.app/features-and-labels/issue/SERV-1311) |
| `06.5` | Wrap a partner API with a self-hosted fallback (reroute + re-bill on failure) | `net-new` | [SERV-1315](https://linear.app/features-and-labels/issue/SERV-1315) |
| `06.6` | Compose fal endpoints inside your app (fal_client chaining: caption, upscale, fan out) | `ready-now` | [SERV-1317](https://linear.app/features-and-labels/issue/SERV-1317) |
| `07.1` | Next.js/Vercel, n8n (keep) + Deploy via MCP (Claude/Cursor: zero → deployed endpoint) | `net-new` · blocked: [SERV-1278](https://linear.app/features-and-labels/issue/SERV-1278) (no deploy-capable MCP yet) | [SERV-1281](https://linear.app/features-and-labels/issue/SERV-1281) |
| `08.1` | Observability in production (OTel traces, metrics export, Slack alerts) | `needs-deploy` | [SERV-1287](https://linear.app/features-and-labels/issue/SERV-1287) |
| `08.2` | Resilience: OOM-safe serving, GPU health, retries, cancellations | `needs-deploy` | [SERV-1297](https://linear.app/features-and-labels/issue/SERV-1297) |
| `08.3` | Publish to the Marketplace (private app → public monetized endpoint) | `needs-deploy` · blocked: [SERV-1277](https://linear.app/features-and-labels/issue/SERV-1277) (marketplace not self-serve) | [SERV-1304](https://linear.app/features-and-labels/issue/SERV-1304) |
| `08.4` | Test endpoints in CI with AppClient (ephemeral deploys as tests) | `needs-deploy` | [SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310) |
| `08.5` | Track usage & cost per app and API key | `ready-now` | [SERV-1314](https://linear.app/features-and-labels/issue/SERV-1314) |

## Open decisions (label: `decision`)

| Ticket | Decision | Gates |
|---|---|---|
| [SERV-1267](https://linear.app/features-and-labels/issue/SERV-1267) | Confirm hot-reload in the fal CLI | 00.1 |
| [SERV-1268](https://linear.app/features-and-labels/issue/SERV-1268) | FLUX-class model choice (weight licensing) | 00.2 |
| [SERV-1269](https://linear.app/features-and-labels/issue/SERV-1269) | Files dashboard UI + ACL semantics | 00.3 |
| [SERV-1270](https://linear.app/features-and-labels/issue/SERV-1270) | Container mutable-tag re-pull semantics | 01.1 |
| [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271) | FlashPack public-wheel parity | 02.1 |
| [SERV-1272](https://linear.app/features-and-labels/issue/SERV-1272) | Which optimizations/numbers fal will publish | 02.4 |
| [SERV-1273](https://linear.app/features-and-labels/issue/SERV-1273) | provide_hints at thousands of hints | 04.3 |
| [SERV-1274](https://linear.app/features-and-labels/issue/SERV-1274) | ComfyUI coordination with Alex (fal-docs PR #404) | 04.1 |
| [SERV-1275](https://linear.app/features-and-labels/issue/SERV-1275) | CDN persistence vs 7-day retention FAQ | 05.3 |
| [SERV-1276](https://linear.app/features-and-labels/issue/SERV-1276) | C2PA signing certificate story | 05.5 |
| [SERV-1277](https://linear.app/features-and-labels/issue/SERV-1277) | Marketplace publishing coordination | 08.3 |
| [SERV-1278](https://linear.app/features-and-labels/issue/SERV-1278) | Deploy-capable MCP server (dependency) | 07.1 |
| [SERV-1279](https://linear.app/features-and-labels/issue/SERV-1279) | This repo: personal → fal-ai-community transfer plan | repo |


## Hard-won facts — do not relearn these

1. **Always branch fal-docs work from current `origin/main`.** An earlier working branch was 135 commits stale and half its plan was fiction. Main's layout: concept docs in `documentation/`, references in `api-reference/` + `model-api-reference/`, examples categorized in `examples/{image-generation,video-generation,audio-speech,integrations,serverless}/`, shared code in `snippets/`. Existing example pages live under `examples/<category>/`; where NEW curriculum pages and nav go is decided by the IA-scaffolding ticket ([SERV-1266](https://linear.app/features-and-labels/issue/SERV-1266)) — read it before creating pages.
2. **Measured numbers were descoped on 2026-07-21 (Rehan's call, after a modal-examples CI audit).** The original plan made CI-measured cold start/latency/cost the differentiator; the owner rescoped CI to correctness only — deploy + health + smoke + tests via `harness/check.py`, matching modal-examples' exit-code-correctness model. The full measurement harness (`harness/measure.py`, per-example `numbers.json`) was built, adversarially reviewed, and E2E-validated first (hello-fal: 7.3s cold start, p50 0.41s, $0.000012/run on S) — recover it from git history (PR #1) if numbers come back. Until then: never publish performance claims that can't be backed by a reproducible run, and ticket DoDs mentioning "measured numbers via the harness" should be read as "validated deployment via the deploy-check".
3. **Known defects in the old examples** are catalogued in [SERV-1263](https://linear.app/features-and-labels/issue/SERV-1263) (descoped to surviving content + copy-and-it-breaks bugs; ticket archived Jul 20 but the defect list remains valid reference). Don't copy patterns from `snippets/wan-code.mdx` et al. without checking that ticket.
4. **This repo's transfer plan** ([SERV-1279](https://linear.app/features-and-labels/issue/SERV-1279)): built on a personal account first, then ownership transfers to [fal-ai-community](https://github.com/fal-ai-community). GitHub auto-redirects old URLs after transfer — but creating a same-named repo on the personal account afterwards kills the redirect. Post-transfer: re-check Actions secrets, org Actions policies, apps, branch protection, team permissions.
5. **Differentiation is substance, not cosmetics** (numbered example repos are a genre): journey tiers, CI-regenerated numbers, playground links on every example, llms.txt/MCP-readable index. Don't mirror modal-examples' README voice.
6. **ComfyUI (04.1)**: do not parallel-build — Alex's [fal-docs PR #404](https://github.com/fal-ai/fal-docs/pull/404) and internal serverless-comfyui-v2 are the path.

## Build order (recommended)

1. **Harness first** ([SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310)): ✅ done — `harness/check.py` + `.github/workflows/check.yml` (deploy-check scope per fact #2).
2. **Template end-to-end**: ✅ done — `00-getting-started/hello-fal/` proven through run → ephemeral deploy-check → registered `fal deploy` (live as `rehan/hello-fal`).
3. **Flagships with tickets unblocked**: 00.2 (first image API), 04.2 (LTX text-to-video), 02.1 (cold starts — FlashPack gate pending).
4. The 8 `ready-now` entries are docs-first (their pages can be written before code lands here).

## Repo conventions

See [CLAUDE.md](CLAUDE.md). Every example: own folder under its tier, copied from `templates/example/`, pinned requirements, smoke test run by CI against a real ephemeral deploy, README in §4-A-derived order (problem → output first → code → deploy → cost & tuning → next steps; the "numbers" section was dropped with the measurement descope).
