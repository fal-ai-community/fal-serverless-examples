# Tier 06 · Pipelines & apps

Composing endpoints into products: workflows, queue + webhooks backends, multi-app routing, failover, client-side chaining.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `06.1` | Chain models with Workflows (expand → generate → upscale → safety, one endpoint) | `needs-deploy` | [SERV-1290](https://linear.app/features-and-labels/issue/SERV-1290) |
| `06.2` | Async queue + webhooks into a product backend (Next.js submits, webhook updates DB) | `needs-deploy` | [SERV-1298](https://linear.app/features-and-labels/issue/SERV-1298) |
| `06.3` | Multi-app routing (one proxy endpoint, many apps) | `ready-now` | [SERV-1305](https://linear.app/features-and-labels/issue/SERV-1305) |
| `06.4` | Built on Serverless live demo app (Together-style: hosted app + repo + page) | `net-new` | [SERV-1311](https://linear.app/features-and-labels/issue/SERV-1311) |
| `06.5` | Wrap a partner API with a self-hosted fallback (reroute + re-bill on failure) | `net-new` | [SERV-1315](https://linear.app/features-and-labels/issue/SERV-1315) |
| `06.6` | Compose fal endpoints inside your app (fal_client chaining: caption, upscale, fan out) | `ready-now` | [SERV-1317](https://linear.app/features-and-labels/issue/SERV-1317) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment + measured numbers · `net-new` = new code/pattern to design first.
