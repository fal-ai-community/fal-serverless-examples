# Tier 01 · Environments & containers

Runtimes and Dockerfiles side by side, migrations from other platforms, dev-to-prod environment promotion.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `01.1` | Same model, two ways: fal runtime vs your own Dockerfile | `needs-deploy` · blocked: [SERV-1270](https://linear.app/features-and-labels/issue/SERV-1270) (mutable-tag re-pull semantics) | [SERV-1285](https://linear.app/features-and-labels/issue/SERV-1285) |
| `01.2` | Migrations (external Docker, Modal, Replicate, RunPod) | `ready-now` | [SERV-1291](https://linear.app/features-and-labels/issue/SERV-1291) |
| `01.3` | Dev → staging → prod (environments, secrets, promotion, rollback) | `ready-now` | [SERV-1299](https://linear.app/features-and-labels/issue/SERV-1299) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment · `net-new` = new code/pattern to design first.
