# Hello, fal

You have Python code that needs to run on serverless infrastructure — with a real HTTP API, typed inputs and outputs, and a path to production — without writing a Dockerfile or managing a server. This is the smallest complete fal app: one endpoint, deployed the same way the biggest ones are.

> Catalog slot `00.1` ([SERV-1284](https://linear.app/features-and-labels/issue/SERV-1284)). This is the minimal version that proves the example layout end-to-end; the full "one app, run three ways" walkthrough lands with its ticket.

## What you get

A live HTTPS endpoint that answers:

```json
{
  "message": "Hello, fal!"
}
```

…plus an auto-generated playground and OpenAPI schema for the typed input (`{"name": "fal"}`), for free.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # ephemeral app on fal infra; prints playground + API URLs
```

`fal run` keeps the app alive until you Ctrl-C — nothing to clean up.

## How it works

[`app.py`](app.py) has three parts:

- **`Input` / `Output`** — pydantic models that become the endpoint's request/response schema. The `Field(examples=[...])` values feed the playground, the OpenAPI docs, and the [numbers harness](../../harness/README.md)'s sample request.
- **`HelloFal(fal.App)`** — the app. `machine_type = "S"` picks the small 1 GB CPU tier (no GPU needed to say hello); `keep_alive=60` keeps a warm worker for 60s after the last request. `requirements` is what fal installs at deploy time — [`requirements.txt`](requirements.txt) mirrors it exactly for local tooling.
- **`@fal.endpoint("/")`** — one POST route. `setup()` is where bigger apps load model weights once per worker; here it's deliberately empty.

## Deploy

```bash
fal deploy app.py
```

This registers the app as `hello-fal` and prints its playground and API URLs. Call it:

```bash
curl -X POST https://fal.run/<your-username>/hello-fal \
  -H "Authorization: Key $FAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "fal"}'
```

## Numbers

Measured by [`harness/measure.py`](../../harness/measure.py) — the machine-readable source of truth is [`numbers.json`](numbers.json) (CI-regenerated; the table below is copied from it, never invented):

| Machine type | Cold start | p50 latency | p95 latency | Cost per run | Runs |
|---|---|---|---|---|---|
| `S` | 7.3 s | 0.411 s | 0.485 s | ~$0.000012 | 10 |

## Cost & tuning

- `S` is a CPU tier billed per second; a greeting endpoint costs fractions of a cent per run. GPU tiers cost dollars per hour — see the machine-types docs before reaching for one.
- `keep_alive=60` trades ~60s of idle billing after each burst for zero cold starts within it. Set it to `0` if you only care about batch-style calls, or higher if your traffic is steady.
- Cold start on a CPU tier is dominated by worker boot and environment restore, not model loading — that's the ~7 s measured above.

## Next steps

- **00.2 — Your first image API**: same shape, but `setup()` loads a real model onto a GPU.
- Concept docs: machine types, `fal run` vs `fal deploy`, keep-alive & autoscaling.
