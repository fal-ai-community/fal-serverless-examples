# 3D progressive rendering: watch SAM-3D build a model live

3D generation takes long enough that a spinner is a product problem: the user
stares at nothing for the better part of a minute. SAM-3D Objects happens to
be built from two diffusion stages whose intermediate states are *drawable* -
a voxel cloud that sharpens, then colorizes. This example streams those
states over SSE while the model runs, so your UI renders progress instead of
promising it. fal's hosted `fal-ai/sam-3/3d-objects` gives you the same model
request/response; deploy this when you want the progressive experience.

## What you get

Progressive voxel frames during both diffusion stages, a vertex-colored mesh
preview on small reconstructions, then the final assets by URL:

```json
{"type":"geometry","step":12,"total_steps":25,"progress":0.27,"voxels":"<base64>","count":8541,"grid_size":64}
{"type":"appearance","step":20,"total_steps":25,"progress":0.82,"voxels":"<base64>","count":8541,"grid_size":64}
{"type":"complete","progress":1.0,"glb":{"url":"https://…/model.glb"},"gaussian_splat":{"url":"https://…/splat.ply"},"seed":42}
```

## Run it

One-time: request access to the gated weights at
huggingface.co/facebook/sam-3d-objects (SAM License), then store a token:

```bash
fal secrets set HF_TOKEN "hf_…"
```

```bash
pip install fal==1.78.2   # keep pinned to this example's requirements.txt
fal auth login
fal run app.py::SAM3DStreamApp
```

Then either open the bundled viewer (`python -m http.server 8000`, open
viewer.html, paste the printed URL + a FAL key - local dev only), or consume
the stream in ~10 lines of Python:

```python
import httpx, json
from fal.sdk import get_credentials

with httpx.stream(
    "POST", "<your app URL>/stream",
    json={"image_url": "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"},
    headers=get_credentials().to_headers(), timeout=600,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            print(event["type"], event.get("progress"))
```

## How it works

Three ideas, in the order they appear in app.py:

1. **No fork.** The official pipeline's generators expose `generate_iter()`,
   a public per-solver-step iterator (`generate()` is just "loop it, keep the
   last value"). `stream_generate()` mirrors `InferencePipeline.run()` stage
   by stage and emits a frame per step. Stage-1 frames decode the
   intermediate latent to a 64³ occupancy grid; stage-2 frames decode
   per-voxel colors through the Gaussian head every `colors_every` steps.
2. **Progress frames are disposable; the terminal event is guaranteed.** The
   GPU worker and the SSE generator meet at a bounded queue: `put_progress`
   drops the oldest frame when the client is slow (the GPU never waits on the
   network), `put_terminal` always delivers. Every stream ends in exactly one
   `complete` or `error` - and disconnects cancel generation between steps
   instead of leaking a wedged worker. Heartbeats are SSE *comments*, sent
   every 15s so proxies don't kill quiet phases.
3. **Small frames inline, big assets by URL.** A voxel frame is 6 bytes per
   voxel (raw 64³ grid indices fit uint8 losslessly - `(v+0.5)/64-0.5` to
   dequantize; colors are scaled float 0-1 → byte exactly once, in one
   encoder). The GLB and splat never travel inline - the `complete` event
   carries fal CDN URLs (see Working with Files).

A rare fifth event, `mesh_preview`, carries a vertex-colored mesh inline for
small reconstructions - most SAM-3D outputs (we measured 306k vertices on the
sample image) exceed the 200k-vertex inline cap in `stream_generate`, so treat
it as a bonus, not a stage your client can depend on.

Concept docs: [Streaming Endpoints](https://docs.fal.ai/serverless/development/streaming) ·
[handling inputs & outputs](https://docs.fal.ai/serverless/development/handle-inputs-and-outputs) ·
[custom containers](https://docs.fal.ai/serverless/development/customize-environment-containers)

## Deploy

```bash
fal deploy app.py
```

Once registered, the fal clients stream it natively (they append `/stream`):

```python
import fal_client

for event in fal_client.stream("<username>/sam-3d-progressive",
                               arguments={"image_url": "https://…/photo.png"}):
    print(event["type"], event.get("progress"))
```

```ts
import { fal } from "@fal-ai/client";

const stream = await fal.stream("<username>/sam-3d-progressive", {
  input: { image_url: "https://…/photo.png" },
});
for await (const event of stream) console.log(event.type, event.progress);
```

## Cost & tuning

Measured on an H100 warm runner (spike numbers, bare pipeline): a full
25+25-step generation (vertex-color GLB, no texture bake) took **≈16s**,
averaging 158ms/step in stage 1 (geometry) and 112ms/step in stage 2
(appearance); each colored frame costs one extra Gaussian decode at
**≈19ms**. Cold setup (weights already cached on `/data`, pipeline
instantiation + model load) was **≈60s warm**; on a brand-new `/data` volume
the very first boot in this repo's history took over 600s before the runner
reported ready - a one-time anomaly, not the steady state, but worth watching
on a fresh deploy. Peak VRAM was **17.8GB**, comfortably under SAM-3D's own
stated ≥32GB prerequisite - the extra headroom is why this stays on H100
rather than a smaller tier.

On the live deployed endpoint (network + export + CDN upload included, not
just the bare pipeline) a full `/` request measured **9-29s** warm end to
end, and a full `/stream` run at default settings (`geometry_every=1`,
`colors_every=5`) measured **≈21s** end-to-end.

Worked example: at fal's current H100 pricing (list **$4.50/hr**, usage-based
discounts as low as **$1.89/hr**, as of 2026-08-11 — see
[fal.ai/pricing](https://fal.ai/pricing)), that's **$0.00125/s** at list
price. One `/stream` request (~21s measured) plus the default 60s
`keep_alive` idle tail is about 80 billed seconds ≈ **$0.10** at list price
(≈**$0.04** at the discounted rate) - most of that is the idle tail, not the
generation.

Knobs, in the order they matter:

| Knob | Tradeoff |
|---|---|
| `keep_alive` (default 60s) | How long the container stays warm (and billed) after the last request. Raise it for interactive sessions with bursty follow-up requests; the default favors batch-style, one-shot calls. |
| `geometry_every` / `colors_every` | Frame volume vs. decode cost. `geometry_every` frames are nearly free (an argwhere on an existing tensor); each `appearance` frame costs one Gaussian decode (≈19ms measured) - `colors_every` defaults to 5 for exactly that reason. |
| `stage1_steps` / `stage2_steps` | Diffusion step count: fewer steps is faster and streams fewer frames, at some cost to geometry/appearance fidelity. |
| `max_concurrency=1` | One `fal.App` instance holds one loaded pipeline on one GPU; the generators (`ss_gen`, `slat_gen`) are shared mutable objects, so this app deliberately serves one generation at a time per container rather than racing shared state. Scale throughput with more containers, not concurrent requests per container. |
| `machine_type = "GPU-H100"` | SAM-3D's own prerequisite is ≥32GB VRAM; we measured 17.8GB peak, but the 80GB H100 keeps both diffusion stages interactive with headroom for future settings (higher step counts, larger grids). |

## Next steps

- [Stream progressive results with SSE](../../03-streaming-realtime/) - the
  minimal version of this pattern (03.1)
- [LOD processor for 3D assets](../../05-media-processing/) - post-process
  the GLB this example produces (05.2)
- Hosted [fal-ai/sam-3/3d-objects](https://fal.ai/models/fal-ai/sam-3/3d-objects)
  for request/response without deploying; TRELLIS.2-4B / Hunyuan3D 3.x if you
  want maximum asset fidelity over progressive UX

---
SAM-3D Objects weights and code are Meta's, under the
[SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE);
this example downloads the gated checkpoint from the official
`facebook/sam-3d-objects` repository.
