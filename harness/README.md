# Measurement harness

Implements the repo's core promise: **every example ships CI-measured numbers** — cold start, latency, and cost per machine type — written to each example's `numbers.json`. Never hand-edit those files.

Tracked in [SERV-1310](https://linear.app/features-and-labels/issue/SERV-1310) (Foundations milestone — built before example content).

## Contract

`python harness/measure.py --example <dir>` — omit `--example` (or pass an empty value, as CI does) to measure all examples:

1. Ephemeral-deploys the example's app (AppClient spawn pattern — see the fal docs page on testing models and endpoints) under a unique run-scoped alias.
2. Measures: cold start (runner state transitions), p50/p95 latency over N warm requests (default 10, `--runs`), and cost derived from machine-type pricing × measured runtime.
3. Runs the example's `test_app.py` (if present) against the live app via the `app_client` fixture from `harness/pytest_plugin.py`.
4. Writes `<dir>/numbers.json` (schema in `templates/example/numbers.json`) with `measured_at` and `harness_version`.
5. Tears the deployment down (stops runners immediately so keep-alive seconds aren't billed) and exits non-zero if the app fails to deploy or the smoke test fails — CI doubles as example rot detection.

## How each number is produced

| Field | Method |
|---|---|
| `cold_start_s` | Runner boot→app-ready duration from the platform's **server-side log timestamps**: `Runner scheduled for allocation` (PENDING begins) → `Application startup complete` (uvicorn serving). One clock domain, no polling error; matches the fal docs' definition ("the time from PENDING to IDLE is your cold start latency"). Falls back to client-clock marker arrival → first healthy `/health` if a timestamp is missing. Note: ephemeral runners are invisible to `fal runners list` / `fal apps runners` — runner state/machine-type come from REST `GET /runners/{id}` with the id parsed from the `Starting runner <uuid>` log line. Caveat: ephemeral runners restore the Python env on the runner itself, so cold start includes env restore (small when cached); a registered `fal deploy` can boot faster. |
| `p50_latency_s`, `p95_latency_s` | Client-measured wall time of `runs` warm POSTs to the example's `/` endpoint, after one untimed smoke request. |
| `cost_per_run_usd` | Effective `$ / second` for the runner's **actual** machine type — fetched at measure time from fal's billing rates endpoint (`/billing/v2/machine_prices`, account-effective rates) — × p50 latency. If the rate is unavailable or the server returns its `0.0` "unpriced" sentinel, the value is `null`; the harness never invents a rate. Note this is marginal compute per request; total billing also counts keep-alive idle seconds (see fal's serverless pricing docs). |
| `machine_type` | Taken from the live runner (not the requested value — apps may list fallback machine types). |
| `runs` | Number of timed warm requests behind the p50/p95. |

A `CRASH_BACKOFF` runner state aborts the measurement as a failed smoke test.

## Sample-input convention

The harness builds the request payload from the endpoint's Input model: every **required** field must carry `examples=[...]` on its `Field(...)` — the same sample the README shows. Optional fields use their defaults. A required field without examples fails the measurement with a clear error.

## Smoke tests (`test_app.py`)

Examples keep tests tiny; the harness owns deploy/teardown:

```python
def test_smoke(app_client):
    response = app_client.run("/", {"prompt": "a calico cat riding a bicycle"})
    assert response["result"]
```

`app_client` comes from `harness/pytest_plugin.py`; the harness invokes pytest with `HARNESS_APP_URL` pointing at the live ephemeral app. Outside the harness the fixture skips, so plain `pytest` never needs GPU time.

## Auth

One code path for both environments (`fal.sdk.get_credentials()`):

- **CI**: `FAL_KEY` env var (repository secret). An API-scoped key suffices — the billing rates endpoint serves default rates without admin scope.
- **Local**: a stored `fal auth login` session works; no `FAL_KEY` needed.

## Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `HARNESS_URL_TIMEOUT` | 900 | seconds to wait for the deploy to produce a URL (covers env build) |
| `HARNESS_STARTUP_TIMEOUT` | 600 | seconds to wait for `/health` after the URL exists (covers `setup()`) |
| `HARNESS_REQUEST_TIMEOUT` | 120 | per-request timeout for smoke/warm calls |
| `HARNESS_WARM_RUNS` | 10 | default number of timed warm requests |

## Notes

- Runs cost real GPU time: the only CI trigger is manual dispatch (`.github/workflows/numbers.yml`); the weekly cron gets uncommented after the first successful CI run.
- Deps are pinned in `harness/requirements.txt`. `fal` is pinned deliberately: measure.py guards against a v1.78.2 behavior where a deploy that fails before emitting a URL would hang the SDK's own `AppClient.connect()` forever (the spawn worker's exception is discarded upstream).
- Python 3.11 required (matches CI and the runner env — fal ships cloudpickled code, so client and runner Python versions must agree).
