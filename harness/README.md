# Deploy-check harness

Keeps the catalog honest: **every example must actually deploy and serve** — CI re-verifies it so nothing here can silently rot (the known failure mode of the old examples was "copy it and it breaks").

CI checks correctness only — deploy + health + smoke + tests; it does not measure performance.

## Contract

`python harness/check.py --example <dir>` — omit `--example` (or pass an empty value, as CI does) to check all examples:

1. Ephemeral-deploys the example's app (AppClient spawn pattern) with **private** auth.
2. Waits for `/health`, fires **one smoke request** at the example's `/` endpoint (payload built from the Input model's `Field(examples=[...])`), and validates a 2xx response.
3. Runs the example's `test_app.py` (if present) against the live app via the `app_client` fixture from `harness/pytest_plugin.py`.
4. Tears the deployment down and **confirms via REST that the runner stopped** (runners bill until terminated); an unconfirmed runner fails the run.
5. Exits non-zero if any example fails to deploy, respond, or pass its tests.

A `CRASH_BACKOFF` runner state aborts the check as a failure.

## Smoke tests (`test_app.py`)

Examples keep tests tiny; the harness owns deploy/teardown:

```python
def test_smoke(app_client):
    response = app_client.run("/", {"prompt": "a calico cat riding a bicycle"})
    assert response["result"]
```

`app_client` comes from `harness/pytest_plugin.py` (registered repo-wide by the root `conftest.py`); the harness invokes pytest with `HARNESS_APP_URL` pointing at the live ephemeral app. Outside the harness the fixture skips, so plain `pytest` never costs anything.

## Sample-input convention

The harness builds the smoke payload from the endpoint's Input model: every **required** field must carry `examples=[...]` on its `Field(...)` — the same sample the README shows. Optional fields use their defaults.

## Auth

One code path for both environments (`fal.sdk.get_credentials()`):

- **CI**: `FAL_KEY` env var (repository secret; API scope suffices).
- **Local**: a stored `fal auth login` session works; no `FAL_KEY` needed.

A preflight call fails the whole run in seconds (with a real error) on bad credentials.

`FAL_KEY` alone is not enough: the harness checks every example and has no skip list, so two examples need extra setup on the account that runs the deploy checks.

- **Required** — `04-genmedia-patterns/3d-progressive-rendering` downloads gated weights. Request access to `facebook/sam-3d-objects` at huggingface.co/facebook/sam-3d-objects, then run `fal secrets set HF_TOKEN "hf_…"` on that account. Without the approval **and** the secret, that example fails its check.
- **Optional** — `03-realtime-streaming/realtime-video-to-video` uses the `METERED_TURN_SECRET_KEY` and `METERED_TURN_LABEL` secrets for TURN relay. If they are absent, the app falls back to public STUN and the check still passes.

## Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `HARNESS_URL_TIMEOUT` | 900 | seconds to wait for the deploy to produce a URL (covers env build). An image built from source needs much more: CI sets 3600 for the flash-attn/pytorch3d compile in `04-genmedia-patterns/3d-progressive-rendering` |
| `HARNESS_STARTUP_TIMEOUT` | 600 | seconds to wait for `/health` after the URL exists (covers `setup()`) |
| `HARNESS_REQUEST_TIMEOUT` | 120 | timeout for the smoke request |
| `HARNESS_TEST_TIMEOUT` | 600 | timeout for the example's pytest run |

## Notes

- Runs cost real compute time: CI triggers are PRs touching example/harness code, manual dispatch, and a weekly cron (enable after the first green run) — `.github/workflows/check.yml`.
- Deps are pinned in `harness/requirements.txt`. `fal` is pinned deliberately: check.py guards against a v1.78.2 behavior where a deploy that fails before emitting a URL would hang the SDK's own `AppClient.connect()` forever (the spawn worker's exception is discarded upstream), and relies on REST `/runners/{id}` because ephemeral runners are invisible to the gRPC runner lists.
- Python 3.11 required (matches CI and the runner env — fal ships cloudpickled code, so client and runner Python versions must agree).
