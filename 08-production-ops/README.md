# Tier 08 · Production ops

Observability, resilience, CI testing with ephemeral deploys, marketplace publishing, cost tracking.

Example folders land here as their tickets complete. Every example follows the layout in [`templates/example/`](../templates/example/) and the quality bar in [`CONTEXT.md`](../CONTEXT.md).

| Slot | Example | Readiness | Ticket |
|---|---|---|---|
| `08.1` | Observability in production (OTel traces, metrics export, Slack alerts) | `needs-deploy` | [SERV-1287](https://linear.app/features-and-labels/issue/SERV-1287) |
| `08.2` | Resilience: OOM-safe serving, GPU health, retries, cancellations | `needs-deploy` | [SERV-1297](https://linear.app/features-and-labels/issue/SERV-1297) |
| `08.3` | Publish to the Marketplace (private app → public monetized endpoint) | `needs-deploy` · blocked: [SERV-1277](https://linear.app/features-and-labels/issue/SERV-1277) (marketplace not self-serve) | [SERV-1304](https://linear.app/features-and-labels/issue/SERV-1304) |
| `08.4` | Test endpoints in CI with AppClient (ephemeral deploys as tests) | `needs-deploy` | [SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310) |
| `08.5` | Track usage & cost per app and API key | `ready-now` | [SERV-1314](https://linear.app/features-and-labels/issue/SERV-1314) |

Readiness: `ready-now` = writable from existing docs today · `needs-deploy` = needs a validation deployment + measured numbers · `net-new` = new code/pattern to design first.
