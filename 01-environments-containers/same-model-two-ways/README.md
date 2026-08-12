# Same model, two ways: fal runtime vs your own Dockerfile

Every fal app has to answer one question before it can run: **where does the environment come from?** You can list your dependencies and let fal build it, or you can ship a Dockerfile. Most guides show you *how* to write the Dockerfile. This one deploys the *same* model both ways and tells you **which to pick, and why** — because for a lot of apps, the answer is "you don't need the Dockerfile".

> Catalog slot `01.1` ([SERV-1285](https://linear.app/features-and-labels/issue/SERV-1285)).

## What you get

Two deployments of `openai/whisper-tiny.en` that are indistinguishable to callers except for one honest field:

```json
{ "text": "So what are we having for dinner tonight?", "built_with": "fal-runtime" }
{ "text": "So what are we having for dinner tonight?", "built_with": "container" }
```

Same schema, same pinned versions, same transcript — and **one test suite that CI runs against both**. If the two ever diverge, the comparison below stops being true and the build goes red.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py             # way 1: fal builds the environment
fal run app_container.py   # way 2: fal builds your Dockerfile, server-side
```

You do **not** need Docker installed, and you never push to a registry — fal builds the image for you.

## How it works

Both files declare the same `Input`/`Output` and the same `setup()`. The only difference:

**Way 1 — [`app.py`](app.py):** `requirements = ["fal==1.78.2", "torch==2.12.1", "transformers==5.3.0"]`. fal resolves and builds; [`requirements.txt`](requirements.txt) mirrors it exactly for local installs.

**Way 2 — [`app_container.py`](app_container.py):** `image = ContainerImage.from_dockerfile("Dockerfile")` — no `requirements` list, because [`Dockerfile`](Dockerfile) bakes the same versions in.

### Which to pick

| | fal runtime (`requirements`) | Your Dockerfile |
|---|---|---|
| **Setup cost** | none — a list of pins | a Dockerfile you own and maintain |
| **Iteration speed** | fastest: change a pin, redeploy | slower: any edit reruns the build |
| **System packages** (ffmpeg, CUDA libs, apt) | ✗ Python deps only | ✓ the reason most people switch |
| **Reproducibility** | pins + fal's resolver | exact, if you pin `FROM` by digest |
| **Build-step control** (compile, prefetch, `COPY`) | ✗ | ✓ |
| **Private base images** | ✗ | ✓ registry credentials supported |
| **Python version** | fal matches your client | **yours to get right** — see below |
| **Cold start** | image is fal's, warmed across the fleet | your image; bigger images pull slower |
| **Cost** | identical — image pull time isn't billed¹ | identical |

**Pick the runtime unless you need something in the ✓ column.** "Same model, two ways" means the container is a *superset* here, not a requirement — this example would work fine with the runtime alone, and saying otherwise would be dishonest framing.

¹ Runners bill for `SETUP`, `IDLE`, `RUNNING`, `DRAINING`, `TERMINATING`; `PENDING` and `DOCKER_PULL` are not billed. So a fat image costs you cold-start *wall clock*, not dollars — a real difference from most platforms.

### The two things that break first custom containers

1. **Python minor version must match.** fal ships your app as cloudpickled bytecode, which is CPython-minor-specific. This repo targets 3.11, so the image installs 3.11. A mismatch surfaces as an opaque connection error at runtime, not a clear message.
2. **`curl` must exist** in the image — fal's default resolver fetches its installer with it. `python:3.11-slim` doesn't ship it; the Dockerfile installs it.

Also: no `ENTRYPOINT`/`CMD` — fal supplies its own.

### When your base image's tag moves

This is the part nobody documents, and the behavior is **not** what most people assume:

> fal's build cache is keyed on **the content of your Dockerfile**, not on what your base-image tag resolves to today. Write `FROM python:3.11-slim` and repoint that tag upstream, and fal keeps serving the image it already built — the new base is never picked up, and nothing warns you.

Two ways to stay honest, both used here:

- **Pin `FROM …@sha256:<digest>`** (what [`Dockerfile`](Dockerfile) does). A base-image change becomes a visible edit to a file in your repo, which is also exactly what makes the cache do the right thing.
- **`fal deploy --no-cache`** when you deliberately want a rebuild against the current tag.

The opposite holds for the other container path: a bare `image = "repo/app:tag"` reference in `pyproject.toml` is re-resolved to a digest on *every* deploy, so it tracks a moving tag automatically. Two paths, opposite semantics — know which one you're on.

## Deploy

```bash
fal deploy app.py             # -> <username>/transcribe-runtime
fal deploy app_container.py   # -> <username>/transcribe-container
```

```bash
curl -X POST https://queue.fal.run/<your-username>/transcribe-runtime \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"audio_url": "https://storage.googleapis.com/falserverless/model_tests/whisper/dinner_conversation.mp3"}'
```

## Cost & tuning

- Both variants run on `L` (4 vCPU / 15 GB) — whisper-tiny is 39M parameters and needs no GPU. Rates: [fal pricing](https://fal.ai/pricing).
- The container variant's *build* is a one-time cost per Dockerfile content hash; later deploys of an unchanged Dockerfile skip the build entirely.
- `keep_alive=60` suits bursty transcription. Raise it if calls arrive in clusters; see 02.2 for the autoscaling math.

## Next steps

- **00.4 — bulletproof inputs**: `audio_url` goes straight to the model here; that example shows how to validate user media before inference.
- **01.3 — dev → staging → prod**: promote whichever build path you picked across environments.
- **02.1 — cut cold starts**: what to do about that image-pull wall clock.
- Concept docs: [custom container images](https://fal.ai/docs/documentation/development/use-custom-container-image) · [managing dependencies](https://fal.ai/docs/documentation/getting-started/manage-dependencies)
