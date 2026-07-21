# Measurement harness

Implements the repo's core promise: **every example ships CI-measured numbers** — cold start, latency, and cost per machine type — written to each example's `numbers.json`. Never hand-edit those files.

Tracked in [SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310) (Foundations milestone — build this before example content).

## Contract

`python harness/measure.py --example <dir>` — omit `--example` (or pass an empty value, as CI does) to measure all examples:

1. Ephemeral-deploys the example's app (AppClient pattern — see the fal docs page on testing models and endpoints).
2. Measures: cold start (runner state transitions), p50/p95 latency over N warm requests with the README's sample input, and cost derived from machine-type pricing x measured runtime.
3. Writes `<dir>/numbers.json` (schema in `templates/example/numbers.json`) with `measured_at` and `harness_version`.
4. Exits non-zero if the app fails to deploy or the smoke test fails — CI doubles as example rot detection.

## Notes

- Runs cost real GPU time: CI triggers are manual dispatch + weekly cron only (`.github/workflows/numbers.yml`).
- Requires `FAL_KEY` in the environment.
