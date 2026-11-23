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
    "url": "https://v3b.fal.media/files/b/0ab392a3/cC-qJ0CzCd2vWCl3f28ls_wan22-8542e4f6bd2345fe85df6b3a3d134ab1.mp4",
    "content_type": "video/mp4",
    "file_name": "wan22-8542e4f6bd2345fe85df6b3a3d134ab1.mp4",
    "file_size": 840455
  },
  "seed": 7,
  "num_gpus": 8,
  "timings": {"generate": 106.6}
}
```

`ffprobe` on the clip shows 1280x720, 81 frames, 16 fps. The committed file is
byte-identical to the output of an earlier revision that used CPU offload, so the speed-ups
described below do not change the output.

## Run it

```bash
pip install fal==1.78.2
fal auth login

git clone https://github.com/fal-ai-community/fal-serverless-examples.git
cd fal-serverless-examples/04-genmedia-patterns/multi-gpu-video-custom-image
fal run app.py --auth private
```

`fal run` ignores `app_auth` and makes the temporary app public unless you pass
`--auth private`. Without it, anyone with the URL could run eight H100s on your bill.

On a healthy node, the app was serving about 2 minutes after the runner started: about
1 minute to pre-read the bf16 copy into the page cache, under 40 s for the 8 workers to
load it onto their GPUs, and under 20 s for the warmup.

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
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04@sha256:78ba...
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
- **Match the base image's CUDA to your framework's.** The `torch==2.12.1` wheel from
  PyPI brings its own CUDA 13.0 runtime, so the base is CUDA 13.0 too. If you compile
  extensions against the base image's CUDA, a mismatch shows up as a load error at runtime.
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
count. WAN has 40 heads, so 2, 4, or 8 GPUs work. `test_app.py` checks this rule. For
your own model, check its head count
before you pick `num_gpus`.

### When one GPU stops responding

If one rank fails to join a collective, the other seven wait for it. Their GPUs sit at 0%
while the runner keeps billing for eight. A single faulty GPU is enough to cause this. The
app limits both waits:

- The warmup in `setup()` has a 15-minute limit. If it hangs, `setup()` fails and fal
  replaces the runner.
- Each request has a 15-minute limit. If it hangs, the endpoint returns a 503. fal then
  terminates the runner and retries the request on a new one.

### Where the time goes, and what keeps it short

Three choices make this app much faster than the obvious version:

- **Every rank keeps the whole pipeline on its GPU.** Sequence parallelism shards the
  sequence, not the weights, so each rank holds both experts, the text encoder, and the
  VAE: about 64 GB. Peak use at 720p x 81 frames was 66 GiB of the H100's 80. diffusers'
  `enable_model_cpu_offload` would fit in less memory, but it moves the models to the GPU
  and back on every request. With offload, the default request took 56 s; without it,
  4.2 s.
- **Only rank 0 decodes.** After the last step every rank holds the same latents. The
  other seven ask for `output_type="latent"` and skip the VAE decode.
- **One parallel read before the workers start.** `setup()` reads the bf16 files once
  with 32 parallel `cat`s, so the 8 workers load from the page cache. Without it, the
  workers took 4 min 22 s to load on one boot; with it, about 1.5 minutes in total.

See [`multi-gpu-video`](../multi-gpu-video/) for the parts that are the same in both
examples: the bf16 conversion on `/data`, the `{"error": ...}` worker contract, and the
warmup. It uses the same resident-weights setup on 4 GPUs.

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
- **Convert the weights on a cheaper app first.** The first boot on an account spends
  about 35 minutes downloading and converting the weights on CPU, while eight H100s are
  billed. Running [`multi-gpu-video`](../multi-gpu-video/) once first does the same
  conversion on four GPUs, and this example reuses its copy.
- **One request at a time per runner.** All eight GPUs work on one video. To serve more
  videos at the same time, add runners (`max_concurrency`).

Measured on 8x GPU-H100 with this Dockerfile (CUDA 13.0.2) on 2026-11-22 (fal 1.78.2,
diffusers 0.39.0, one registered deploy, warm runner). These are single measurements, not
a guarantee:

| Config | This app | This app with CPU offload | 4 GPUs with CPU offload ([`multi-gpu-video`](../multi-gpu-video/)) |
|---|---|---|---|
| 720p · 81 frames · 27 steps | 106.6 s | 158.9 s | ~270-282 s |
| 480p · 33 frames · 6 steps (defaults) | 4.2 s | 56.1-56.2 s | ~62-95 s (request wall-clock) |

Times are `timings.generate`. At the defaults, a request takes about 6.5 s from the client.
An instrumented run split the 720p request into 97 s of denoising and 11.5 s of VAE decode
on rank 0. The first 720p request on a runner warmed at 480p took as long as the second,
so there is no per-shape warm-up cost.

Building the image with `fal deploy --no-cache` took about 3 minutes on fal's builders.
The one-time bf16 conversion on `/data` was observed once, on the account's first boot,
and was not timed separately. The deploy-check harness passed on the final Dockerfile:
it launched 8 ranks and passed the smoke request and `test_app.py`.

## Next steps

- Put this app behind an agent with other models: [`multi-app-pipeline`](../multi-app-pipeline/)
  chains an LLM, a DiT, and an SR model, each as its own app.
- [Multi-GPU workloads](https://fal.ai/docs/documentation/serverless/distributed/overview):
  the other parallelism strategies `fal.distributed` supports.
- [Use a custom container image](https://fal.ai/docs/documentation/development/use-custom-container-image):
  build context, `COPY`, and `.dockerignore`, for when your model code lives in your repo.
