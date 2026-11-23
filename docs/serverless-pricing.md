# fal Serverless pricing

> Mirrored from [fal.ai/docs/documentation/serverless/pricing](https://fal.ai/docs/documentation/serverless/pricing). The docs page is the source of truth.

When you deploy your own applications on fal Serverless, you are billed for the **total time your runners are alive**, measured per-second by machine type.

## Billing by runner state

Every runner transitions through the states below during its lifecycle. You are billed for the states marked **Yes** at the per-second rate for your machine type.

| State | Billed | Description |
|---|---|---|
| **PENDING** | No | Waiting to be scheduled on available hardware |
| **DOCKER_PULL** | No | Pulling your container image from the registry |
| **SETUP** | Yes | Running your `setup()` method — loading models, initializing resources |
| **IDLE** | Yes | Runner is ready but waiting for requests (includes `keep_alive` time) |
| **RUNNING** | Yes | Actively processing one or more requests |
| **DRAINING** | Yes | Finishing in-flight requests before shutdown |
| **TERMINATING** | Yes | Running your `teardown()` method |
| **TERMINATED** | No | Runner has stopped and resources are released |

5xx errors (HTTP 500+) are also not charged.

See [Runners](https://fal.ai/docs/documentation/deployment/runners) for full details on each state and transitions.

## GPU count multiplier

Multi-GPU instances are billed as `gpu_count x duration`. For example, a runner using 2x A100 GPUs for 60 seconds is billed as 120 GPU-seconds.

## Monitoring your usage

- [Dashboard Billing](https://fal.ai/dashboard/billing): View your overall spend, invoices, and payment methods.
- [App Analytics](https://fal.ai/docs/documentation/serverless/observability/app-analytics): See per-app cost breakdown, request counts, and runner utilization.

## GPU prices

Prices as of 2026-10-19, see [fal.ai/pricing](https://fal.ai/pricing). Per-second prices are the hourly price divided by 3,600. "As low as" prices are the lowest published rates; [contact sales](https://fal.ai/enterprise#contact-sales) for terms.

| GPU | VRAM | List price (per hour) | List price (per second) | As low as (per hour) | As low as (per second) |
|---|---|---|---|---|---|
| B300 | 288 GB | $12.99 | $0.003608 | $5.99 | $0.001664 |
| GB200 | 192 GB | $9.99 | $0.002775 | $5.89 | $0.001636 |
| B200 | 192 GB | $7.99 | $0.002219 | $5.49 | $0.001525 |
| H200 | 141 GB | $6.00 | $0.001667 | $2.99 | $0.000831 |
| H100 | 80 GB | $4.50 | $0.00125 | $2.49 | $0.000692 |
| RTX PRO 6000 | 96 GB | $4.00 | $0.001111 | $1.99 | $0.000553 |
