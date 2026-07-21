# Tier 05 · Media processing

The CPU/GPU pipelines around the models: transcode ladders, 3D LODs, image variants, audio finishing, provenance stamping, batch fan-out.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `05.1` | Post-production pipeline (transcode ladder, thumbnails, captions; ffmpeg on CPU machines + queue + webhooks) | `net-new` | [SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282) |
| `05.2` | LOD processor for 3D assets (GPU-generate → CPU-refine, packaged glTF) | `net-new` | [SERV-1288](https://linear.app/features-and-labels/issue/SERV-1288) |
| `05.3` | Image variant factory (AVIF/WebP/resize ladders to CDN) | `net-new` · blocked: [SERV-1275](https://linear.app/features-and-labels/issue/SERV-1275) (CDN retention confirmation) | [SERV-1296](https://linear.app/features-and-labels/issue/SERV-1296) |
| `05.4` | Audio finishing (loudness normalization, waveform peaks) | `net-new` | [SERV-1301](https://linear.app/features-and-labels/issue/SERV-1301) |
| `05.5` | Provenance & safety stamping (watermark / C2PA on every output) | `net-new` · blocked: [SERV-1276](https://linear.app/features-and-labels/issue/SERV-1276) (C2PA certificate story) | [SERV-1307](https://linear.app/features-and-labels/issue/SERV-1307) |
| `05.6` | Batch fan-out over a bucket (CPU stages on CPU machines) | `net-new` | [SERV-1313](https://linear.app/features-and-labels/issue/SERV-1313) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment + measured numbers · `net-new` = new code/pattern to design first.
