# A video model on 8 GPUs from your own Dockerfile

Teams that train their own video model usually already have an inference image: a CUDA
base, their own model code, and system packages such as ffmpeg. They also need more than
one GPU per request, because a large video DiT is too slow, or too big, on one GPU. This
example combines the two. It builds the environment from a digest-pinned CUDA Dockerfile
with `ContainerImage.from_dockerfile`, asks for `num_gpus = 8`, and uses
`fal.distributed` to split every denoising step of one video across all eight H100s.

The model is WAN 2.2 A14B (two 14B experts, about 70 GB in bf16), the same model as
[`multi-gpu-video`](../multi-gpu-video/), which runs it on 4 GPUs with fal's runtime
image. Read the two side by side: the model code is the same, and only the environment
and the GPU count change. To serve your own model, keep the structure and replace the
worker's `setup()` and `__call__()`.

## What you get

A text prompt in; a CDN-hosted mp4 (16 fps) out, with the seed echoed and the number of
GPUs that worked on it:

This [720p sample clip](assets/sample-720p-8gpu-seed7.mp4) is the real output of this
example's validation run against a registered deploy (720p, 81 frames, 27 steps, seed
7). Response verbatim (the CDN URL has limited retention; the committed clip is the
durable copy):

```json
{
  "video": {
    "url": "https://v3b.fal.media/files/b/0ab34845/5wzQuA4Jln4TBUgaRbCcH_wan22-d42db670b2fc4ad1a8b4699ec0f778e3.mp4",
    "content_type": "video/mp4",
    "file_name": "wan22-d42db670b2fc4ad1a8b4699ec0f778e3.mp4",
    "file_size": 840455
  },
  "seed": 7,
  "num_gpus": 8,
  "timings": {"generate": 155.5}
}
```

`ffprobe` on the clip shows 1280x720, 81 frames, 16 fps.

## Run it

```bash
pip install fal==1.78.2
fal auth login

git clone https://github.com/fal-ai-community/fal-serverless-examples.git
cd fal-serverless-examples/04-genmedia-patterns/multi-gpu-video-custom-image
fal run app.py
```

Use Python 3.11 locally. fal sends your app class to the runner as pickled bytecode, so
your Python version must match the Python in the image.

The first deploy builds the image on fal's builders (no local Docker, no registry). The
first boot on an account also downloads the ~126 GB fp32 weights and stores a bf16 copy
on [`/data`](https://fal.ai/docs/documentation/development/use-persistent-storage). Both
happen once. `multi-gpu-video` uses the same `/data` path, so if you ran it already, this
example reuses its copy.

## How it works

### The Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:17e2...
RUN apt-get install ... python3.11 curl ffmpeg ...
RUN python3.11 -m pip install torch==2.12.1 diffusers==0.39.0 ... pyzmq==27.1.0 ...
RUN python3.11 -m pip install fal==1.78.2
```

```python
class MultiGpuVideoCustomImage(fal.App, keep_alive=300):
    machine_type = "GPU-H100"
    num_gpus = 8
    image = ContainerImage.from_dockerfile(str(DOCKERFILE))
```

Four rules apply to any custom image on fal:

- **Python 3.11 in the image, the same version as on your machine.** A mismatch fails at
  runtime with an unclear error.
- **`curl` must be installed.** fal's runtime installer uses it.
- **Install `fal` last**, so that pip picks `pydantic`, `protobuf`, and `boto3` versions
  that fal supports.
- **Pin the base image by digest.** fal caches builds by the Dockerfile's text. If you
  use a moving tag, a new upstream image never reaches your build. With a digest, every
  base-image change is a visible edit.

There is no `ENTRYPOINT` or `CMD`; fal starts its own server in the image. If your image
already runs its own HTTP server, see
[Deploy an existing server](https://fal.ai/docs/documentation/development/migrate-external-docker-server)
instead.

`fal.distributed` needs `pyzmq`, which the `fal` package does not install, so the
Dockerfile installs it explicitly.

### Eight GPUs, one video

`num_gpus = 8` gives the app eight H100s on one machine (`cuda:0` to `cuda:7`).
`DistributedRunner` starts one `WanWorker` process per GPU and initializes NCCL before
the worker's `setup()` runs. Each request goes to all eight ranks with the same inputs
and seed. Ulysses sequence parallelism splits the attention sequence across the ranks,
and rank 0 writes and returns the mp4.

The Ulysses degree is the GPU count, and it must divide the model's attention-head
count. WAN has 40 heads, so 2, 4, or 8 GPUs work. `test_app.py` checks this rule. For your own model, check its head count
before you pick `num_gpus`.

See [`multi-gpu-video`](../multi-gpu-video/) for the parts that are the same in both
examples: the bf16 conversion on `/data`, CPU offload of the inactive expert, the
`{"error": ...}` worker contract, and the warmup.

## Deploy

```bash
fal deploy app.py
```

Call it through the queue; long clips take minutes:

```python
import fal_client

result = fal_client.subscribe(
    "<your-username>/multi-gpu-video-custom-image",
    arguments={
        "prompt": "a red panda napping on a mossy branch, soft morning light",
        "resolution": "720p",
        "num_frames": 81,
        "num_inference_steps": 27,
        "seed": 7,
    },
)
print(result["video"]["url"], result["num_gpus"])
```

## Cost & tuning

- **Multi-GPU bills `gpu_count x duration`.** Every runner-second of this app is eight
  H100-seconds, from boot to teardown. `keep_alive = 300` keeps a warm runner for 5
  minutes after the last request.
- **An 8-GPU allocation can wait for capacity.** A whole node is harder to get than one
  GPU. Call the app through the queue with a generous timeout.
- **One request at a time per runner.** All eight GPUs work on one video. To serve more
  videos at the same time, add runners (`max_concurrency`).

Measured on 8x GPU-H100 on 2026-11-20 (fal 1.78.2, diffusers 0.39.0, one registered
deploy, warm runner). These are single measurements, not a guarantee:

| Config | `generate` on 8 GPUs | `generate` on 4 GPUs ([`multi-gpu-video`](../multi-gpu-video/)) |
|---|---|---|
| 720p · 81 frames · 27 steps | 155.5 s | ~270-282 s |
| 480p · 33 frames · 6 steps (defaults) | 54.5-56.0 s | — |

Doubling the GPUs cut the full-quality clip from about 280 s to 156 s (1.8x). The short
480p default gains much less. At 6 steps the fixed per-request costs, such as moving the
two experts between CPU and GPU (`enable_model_cpu_offload`) and the VAE decode, are a
large share of the time, and more GPUs do not shorten them.

The deploy-check harness run on the same day built the image from scratch, converted the
weights to bf16 on `/data` (first boot on that account), launched 8 ranks, and passed the
smoke request and `test_app.py`.

## Next steps

- Put this app behind an agent with other models: [`multi-app-pipeline`](../multi-app-pipeline/)
  chains an LLM, a DiT, and an SR model, each as its own app.
- [Multi-GPU workloads](https://fal.ai/docs/documentation/serverless/distributed/overview):
  the other parallelism strategies `fal.distributed` supports.
- [Use a custom container image](https://fal.ai/docs/documentation/development/use-custom-container-image):
  build context, `COPY`, and `.dockerignore`, for when your model code lives in your repo.
