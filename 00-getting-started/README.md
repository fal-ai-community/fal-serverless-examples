# Tier 00 · Getting started

First app to first production API: hello-fal three ways, your first image API, media in/out, bulletproof inputs.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `00.1` | Hello, fal: one app, run three ways (fal run → hot-reload → deploy) | `needs-deploy` · blocked: [SERV-1267](https://linear.app/features-and-labels/issue/SERV-1267) (confirm hot-reload in CLI) | [SERV-1284](https://linear.app/features-and-labels/issue/SERV-1284) |
| `00.2` | Your first image API (FLUX-class, warmup, playground, clients); replaces SDXL tutorial | `needs-deploy` · blocked: [SERV-1268](https://linear.app/features-and-labels/issue/SERV-1268) (FLUX weight licensing decision) | [SERV-1293](https://linear.app/features-and-labels/issue/SERV-1293) |
| `00.3` | Media in, media out (files, CDN, persistent storage) | `needs-deploy` · blocked: [SERV-1269](https://linear.app/features-and-labels/issue/SERV-1269) (Files UI / ACL confirmation) | [SERV-1302](https://linear.app/features-and-labels/issue/SERV-1302) |
| `00.4` | Bulletproof inputs (typed 422s, decompression-bomb gate, SSRF-safe downloads) | `net-new` | [SERV-1309](https://linear.app/features-and-labels/issue/SERV-1309) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment + measured numbers · `net-new` = new code/pattern to design first.
