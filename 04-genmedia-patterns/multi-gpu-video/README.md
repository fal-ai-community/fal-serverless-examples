# Multi-GPU video inference with fal.distributed

Frontier video models outgrow single GPUs: WAN 2.2 A14B carries two 14-billion-parameter
expert transformers - about 70 GB of weights in bf16 - and even where they fit, one GPU
grinds through a clip. The traditional serving answer is a pile of orchestration hacks:
a torchrun subprocess, a ready-file handshake, an HTTP hop to a rank-0 server (the
[official WAN repo's multi-GPU instructions](https://github.com/Wan-Video/Wan2.2)
still work that way). This example serves the same class of model with none of that:
`fal.distributed` spawns one worker per GPU and initializes NCCL before your code runs,
and diffusers context parallelism splits every denoising step across four H100s -
one prompt, one video, four GPUs cooperating.

## What you get

A text prompt in; a CDN-hosted mp4 (16 fps) out, with the seed echoed so the same inputs
reproduce the same clip. This
[720p sample clip](assets/sample-720p-seed7.mp4)
is the real output of this example's validation run against a registered deploy
(720p, 81 frames, 27 steps, seed 7), response verbatim:

```json
{
  "video": {
    "url": "https://v3b.fal.media/files/b/0aa5f482/O2ywRcibrXZ1YRCIu27CA_wan22-1d96a465e70c4e76ba364350cd034fc4.mp4",
    "content_type": "video/mp4",
    "file_name": "wan22-1d96a465e70c4e76ba364350cd034fc4.mp4",
    "file_size": 841093
  },
  "seed": 7,
  "timings": {"generate": 282.2}
}
```

The `url` above is the CDN link that run actually returned; fal CDN files have
limited retention, so the durable copies are the two clips committed under
[`assets/`](assets/).

The CI deploy-check exercises the same endpoint at the fast defaults (480p, 33 frames,
6 steps) - [its output clip](assets/ci-defaults-480p.mp4)
comes back in about a minute on a warm runner. Timings vary with the node and whether
the shape is warm - see the measured table under Cost & tuning.

## Run it

```bash
pip install fal==1.78.2
```

```bash
fal auth login
```

```bash
fal run app.py
```

The first-ever boot on your account downloads the ~126 GB fp32 weights and stores a bf16
copy on [`/data`](https://fal.ai/docs/documentation/development/use-persistent-storage) (roughly half an hour,
once). Every boot after that loads the converted copy and is serving in a few minutes.
`fal run` prints the playground and API URLs when the app is up.

## How it works

### One worker per GPU

`DistributedRunner` spawns `world_size` worker processes (via `torch.multiprocessing`,
not a torchrun subprocess), sets `RANK`/`WORLD_SIZE`, and calls
`torch.distributed.init_process_group` **before** your `setup()` runs. Workers signal
readiness over the runner's own channel - no ready-files, no port juggling - and
`await runner.start()` fails loudly if any rank dies instead of hanging forever.

```python
class WanWorker(DistributedWorker):
    def setup(self, **kwargs):
        pipe = WanPipeline.from_pretrained(CONVERTED_DIR, torch_dtype=torch.bfloat16)
        cp = ContextParallelConfig(ulysses_degree=self.world_size)
        pipe.transformer.enable_parallelism(config=cp)
        pipe.transformer_2.enable_parallelism(config=cp)
```

### Sequence parallelism, not data parallelism

Data parallelism gives each GPU its own *whole* generation - more videos, same latency.
This example does the opposite: Ulysses sequence parallelism (diffusers'
`enable_parallelism`, experimental) splits the attention sequence of a *single*
generation across all four ranks - the right trade when the model itself is the
bottleneck. Two consequences show up in the code:

- **Every rank runs the same call with the same seed.** Context parallelism requires
  identical inputs on every rank; under data parallelism you would do the opposite
  (per-rank seeds for different outputs).
- **Only rank 0 returns a result.** The other ranks return `{}`; their contribution
  happens inside the pipeline through NCCL collectives.

See [Multi-GPU workloads](https://docs.fal.ai/serverless/distributed/overview) for the
full strategy taxonomy.

### The MoE VRAM budget

Sequence parallelism shards the *sequence*, not the weights - all ~70 GB would sit on
every rank. Two tricks keep 80 GB H100s comfortable:

- **bf16 pre-conversion on `/data`**: the HF repo stores fp32 (~126 GB); `setup()`
  converts once to bf16, so all four workers read half the bytes on every later boot
  (measured: worker load drops from ~14 min worst-case to ~2-4 min).
- **Sequenced CPU offload** (`enable_model_cpu_offload`): WAN 2.2's two experts run in
  disjoint denoising stages, so only the active component stays resident - peak is
  ~27-30 GB per GPU, with headroom all the way to 720p x 81 frames.

### The error contract

Worker exceptions do **not** raise on the caller side - `runner.invoke()` returns
`{"error": "..."}`, and the endpoint re-raises it as a 5xx (which fal does not bill and
retries on healthy hardware). The warmup in `setup()` asserts on that key so a broken
model never reports healthy - and it warms the default 480p/33-frame shape, because
attention kernels autotune per shape and a cold shape adds ~40 s to the first request.
One more honesty note: each runner handles one request at a time - fal's default
`max_multiplexing = 1` admits a single request per runner, and the distributed workers
process tasks serially - so this endpoint scales by adding runners (raise
`max_concurrency` or use `fal apps scale`), not by stacking requests on one.

## Deploy

```bash
fal deploy app.py
```

Call it through the queue - production settings take minutes, so don't use raw
synchronous HTTP:

```python
import fal_client

result = fal_client.subscribe(
    "<your-username>/multi-gpu-video",
    arguments={
        "prompt": "a red panda napping on a mossy branch, soft morning light",
        "resolution": "720p",
        "num_frames": 81,
        "num_inference_steps": 27,
        "seed": 7,
    },
)
print(result["video"]["url"])
```

```js
import { fal } from "@fal-ai/client";

const result = await fal.subscribe("<your-username>/multi-gpu-video", {
  input: {
    prompt: "a red panda napping on a mossy branch, soft morning light",
    resolution: "720p",
    num_frames: 81,
    num_inference_steps: 27,
  },
});
console.log(result.data.video.url);
```

```bash
curl -X POST https://queue.fal.run/<your-username>/multi-gpu-video \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "a red panda napping on a mossy branch, soft morning light", "resolution": "720p", "num_frames": 81, "num_inference_steps": 27}'
```

That 720p call is also this example's validation protocol: deploy, run it once, confirm
the returned mp4 plays. (Last run 2026-08-11 - it produced the sample clip at the top of
this page.)

## Cost & tuning

- **Multi-GPU bills `gpu_count x duration`.** This app runs 4x GPU-H100, so every
  runner-second costs four H100-seconds, from boot through teardown.
- **`keep_alive = 300`** keeps the runner warm for 5 minutes after the last request, so
  bursts share one boot while an idle app still scales to zero.
- **When to change `num_gpus`**: the Ulysses degree must divide the model's attention
  heads (WAN 2.2 A14B has 40, so 2, 4, or 8 GPUs work). More ranks cut per-clip latency
  at a higher per-second rate.
- **Fast previews vs final quality**: the defaults (480p, 33 frames, 6 steps) are the
  fast path; WAN's native quality settings (720p, 81 frames, 27 steps) use the same
  endpoint and seed semantics.

Measured on real 4x GPU-H100 runs (validation deploys 2026-08-04/05, registered-deploy
run 2026-08-11; fal 1.78.2, diffusers 0.39.0) - one-time
measurements, not a guarantee:

| Config | Wall-clock |
|---|---|
| 480p · 33 frames · 6 steps (defaults, warm shape) | ~62-95 s |
| 720p · 81 frames · 27 steps | ~270-282 s |

Node-to-node variance is real: worker boot ranged 15 s (page-cache-warm node) to 221 s
(cold node) on the bf16 copy, and host-to-device bandwidth varies with the machine.
Multi-GPU capacity can also queue - a 4-GPU allocation may wait when capacity is tight
(we have seen tens of minutes at peak), which is one more reason to call this endpoint
through the queue with generous client timeouts rather than synchronous HTTP.

## Next steps

- [Tier 04 - Genmedia patterns](../README.md) - the neighboring LoRA and 3D
  serving examples
- [Streaming progress over SSE](https://docs.fal.ai/serverless/distributed/streaming) -
  add preview frames from rank 0 with `add_streaming_result`
- [Tier 02 - Scaling & cold starts](../../02-scaling-cold-starts/README.md) - compiled
  caches and faster weight loading for very large models
