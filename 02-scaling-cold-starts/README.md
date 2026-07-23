# Tier 02 · Scaling & cold starts

Cold-start engineering with measured before/afters, autoscaling for bursty traffic, serving very large models.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `02.1` | Cut cold starts by Nx (FlashPack + /data + compiled caches, measured before/after) — [`cut-cold-starts/`](cut-cold-starts/) (public `flashpack` wheel; parity decision [SERV-1271](https://linear.app/features-and-labels/issue/SERV-1271) still open) | `needs-deploy` | [SERV-1286](https://linear.app/features-and-labels/issue/SERV-1286) |
| `02.2` | Tune autoscaling for bursty traffic (keep_alive, concurrency, cost math) | `ready-now` | [SERV-1294](https://linear.app/features-and-labels/issue/SERV-1294) |
| `02.3` | Serve a 20+ GB video model (parallel loading, GPU sizing reasoning) | `needs-deploy` | [SERV-1303](https://linear.app/features-and-labels/issue/SERV-1303) |
| `02.4` | Free-lunch optimizations, with numbers (prompt cache, FA3 one-liner, GPU branching) | `net-new` · blocked: [SERV-1272](https://linear.app/features-and-labels/issue/SERV-1272) (IP disclosure decision) | [SERV-1308](https://linear.app/features-and-labels/issue/SERV-1308) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment · `net-new` = new code/pattern to design first.
