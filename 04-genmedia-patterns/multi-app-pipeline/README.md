# Multi-app pipeline: agent, LLM, DiT, and SR on separate machines

A real generative video product is rarely one model. A request goes through an agent that
decides what to do, an LLM that writes the prompt, a diffusion transformer (DiT) that
renders the clip, and a super-resolution (SR) model that upscales it. These stages need
different hardware: the agent needs no GPU, the LLM fits on a smaller GPU, the DiT wants
the largest GPU you have, and the SR model is small but processes many frames. If you put
them all in one container on one GPU type, you pay H100 prices for an orchestrator that
only waits.

This example deploys each stage as its own private fal app, on its own machine type with
its own scaling. A CPU agent app chains them through the fal queue:

```
client ──> agent (CPU "S") ──> llm  (GPU-A100)  Qwen2.5-7B-Instruct: idea -> prompt
                           ──> dit  (GPU-H100)  WAN 2.1 T2V 1.3B:    prompt -> 832x480 clip
                           ──> sr   (GPU-A100)  Real-ESRGAN x2:      clip -> 1664x960 clip
```

## What you get

A short idea in; an upscaled mp4 out, plus the un-upscaled clip, the prompt the LLM wrote,
and per-stage timings. Below is a real response from a warm pipeline (all four apps
deployed on one account; 33 frames, 20 steps, seed 7). The CDN URLs have limited
retention, so the URLs are omitted here:

```json
{
  "video":      {"content_type": "video/mp4", "file_name": "upscaled.mp4", "file_size": 4728554},
  "base_video": {"content_type": "video/mp4", "file_name": "video.mp4", "file_size": 244986},
  "prompt": "A red panda curled up asleep on a weathered wooden branch under a canopy of dense green leaves, dappled sunlight filtering through.",
  "negative_prompt": "human, city, indoor setting",
  "seed": 7,
  "timings": {
    "llm": {"wall_seconds": 1.65,  "compute_seconds": 1.13,  "overhead_seconds": 0.52},
    "dit": {"wall_seconds": 11.79, "compute_seconds": 10.87, "overhead_seconds": 0.92},
    "sr":  {"wall_seconds": 6.04,  "compute_seconds": 5.07,  "overhead_seconds": 0.97}
  }
}
```

`ffprobe` on the two clips: `base_video` is 832x480, 33 frames, 16 fps. `video` is
1664x960, 33 frames, 16 fps. The whole request took 20.3 s from the client.

`overhead_seconds` is what each hop costs: the agent's wall time for the call minus the
time the stage itself reports. It covers queueing, the HTTP round trips, and the file
moving through the fal CDN. On warm runners it was 0.5-1.0 s per stage in this run. A cold
stage adds its boot time to its own overhead, and nothing else: in a run where every app
started cold, the request took 70.4 s, and the LLM stage's `overhead_seconds` was 31.3 s.

## Run it

The agent calls the stages by name, so deploy the three stages first. Each command
creates one private app on your account:

```bash
pip install fal==1.78.2 fal-client==1.0.3
fal auth login

git clone https://github.com/fal-ai-community/fal-serverless-examples.git
cd fal-serverless-examples/04-genmedia-patterns/multi-app-pipeline

fal deploy stage_llm.py    # -> <you>/video-pipeline-llm
fal deploy stage_dit.py    # -> <you>/video-pipeline-dit
fal deploy stage_sr.py     # -> <you>/video-pipeline-sr
```

Then run the agent. `fal run` gives you a temporary agent with a playground URL. It calls
the deployed stages:

```bash
fal run app.py
```

Or deploy it too and call it like any other fal app:

```bash
fal deploy app.py          # -> <you>/video-pipeline-agent
```

```python
import fal_client

result = fal_client.subscribe(
    "<you>/video-pipeline-agent",
    arguments={"idea": "a red panda napping on a branch", "seed": 7},
)
print(result["video"]["url"], result["timings"])
```

You can call each stage on its own too, which helps when you debug one of them:

```python
fal_client.subscribe("<you>/video-pipeline-sr", arguments={"video_url": "https://..."})
```

## How it works

### One app per stage

Each `stage_*.py` file is an ordinary `fal.App` with its own `machine_type`,
`requirements`, and `keep_alive`, and a fixed name (`name="video-pipeline-dit"`). Each
stage boots, scales, and fails independently:

- Ten requests in flight make the DiT scale to more H100s. The agent stays on one CPU
  runner because `max_multiplexing = 20` lets one runner wait on many requests at once.
- Changing the SR model is a `fal deploy stage_sr.py` and nothing else. The other stages
  keep their warm runners.
- Each stage has its own logs, metrics, and playground in the dashboard.

To serve a model that needs a whole node, replace a stage with a multi-GPU app. For
example, [`multi-gpu-video-custom-image`](../multi-gpu-video-custom-image/) runs WAN 2.2
A14B on 8 H100s from a custom Dockerfile. The agent only needs the new app's name and its
input fields.

### How the agent calls the stages

The agent uses `fal_client.AsyncClient.subscribe`, which is the same queue API that any
fal client uses. Two details make this work inside fal without any configuration:

- **Credentials.** fal injects `FAL_KEY` into every runner. The key is scoped to the
  app's owner, so the agent can call the owner's **private** stage apps, and nobody else
  can.
- **Names.** The agent builds `<owner>/<stage-name>` from `FAL_USER_NICKNAME`, which fal
  also injects. The same code works on any account, without hard-coded user names.

The queue API gives each hop retries and a request id that you can find in the logs. A
long DiT request does not hold a connection open, and a stage that is scaling up queues
the request instead of failing it.

### Large data moves as URLs

Stages return `File` outputs, which fal uploads to its CDN. The next stage gets a URL,
not bytes. A video never goes through the agent, so a small CPU agent can coordinate
stages that move hundreds of megabytes.

### Errors and timeouts

The agent validates what it can before it calls anything. For example, it rejects
`num_frames=34` with a 422 before it spends an LLM call.

If a stage returns a 4xx, the agent returns the same status, because the input was wrong.
Any other stage failure becomes a `502` that names the stage and app, for example
`llm stage (alice/video-pipeline-llm) failed: Application "video-pipeline-llm" not found`.
You know which app's logs to read. fal does not retry 4xx or 502 responses.

Each stage call has a 30-minute limit, which covers a cold DiT boot. The agent's own
`request_timeout` is set above the sum of the three. `skip_retry_conditions = ["timeout"]`
stops fal from retrying a timed-out agent request, because a retry would resubmit every
stage.

The SR stage checks its input before it decodes it: at most 200 MB, 1920x1080, and 241
frames. It reads, upscales, and writes four frames at a time, so memory does not grow with
the clip length.

## Deploy

The four apps are deployed separately. Deploy order matters only the first time: the
stages must exist before the agent receives traffic. After that, redeploy any app on its
own.

The commands above were run end to end from a fresh clone on 2026-11-20 in three states:
with the stages missing (the 502 above), with all four apps cold, and with all of them
warm.

To test a change before you deploy it, use the repository's deploy-check harness. The test
suite has free unit tests and one live test. The live test runs only under the
deploy-check harness, and only after the three stages are deployed on the harness
account. The harness checks only `app*.py` files, so it tests the agent against whatever
stage versions are deployed:

```bash
fal deploy stage_llm.py && fal deploy stage_dit.py && fal deploy stage_sr.py
HARNESS_REQUEST_TIMEOUT=1800 python harness/check.py --example 04-genmedia-patterns/multi-app-pipeline
```

## Cost & tuning

- **Pay for GPUs only where there is GPU work.** The agent runs on `S`, the cheapest
  machine type. In the run above it spent 20 s waiting while the GPU stages worked.
- **`keep_alive` per stage.** Each stage keeps its own warm window (300 s here). A stage
  that is used rarely, such as SR when most users skip it, can use a shorter window
  without making the others cold.
- **Hop overhead is about 1 s per stage when warm.** For a pipeline whose stages run for
  seconds to minutes, that is small. If one hop must be faster than that, put those two
  models in the same app.
- **Run independent stages at the same time.** This pipeline is a chain, but if your
  agent calls independent stages (for example, audio and video), start them with
  `asyncio.gather`.

## Next steps

- Make the agent decide more. Let the LLM stage return a plan (which stages, which
  settings), and let the agent follow it.
- Replace the DiT stage with [`multi-gpu-video-custom-image`](../multi-gpu-video-custom-image/)
  for a larger model on 8 GPUs.
- Stream progress to the client while the stages run. See
  [`3d-progressive-rendering`](../3d-progressive-rendering/) for SSE streaming.
