"""The burst target for catalog slot 02.2 (SERV-1294).

Same model as 00.2 (FLUX.2 [klein] 4B), with two differences that exist so
one real burst can be measured without any extra credential:

  1. machine_type is PINNED to GPU-H100 rather than a fallback list. GPU-L40
     has no published price at all, so a silent fallback would make the
     dollar figures unattributable. burst.py asserts the machine it landed on.
  2. Every response self-reports which runner served it, whether that runner
     was cold, and how long its setup() took. That is the whole measurement:
     distinct served_by values give the runner ramp, was_cold marks who paid
     a cold start, and setup_duration_s IS the cold start, measured on the
     runner rather than inferred from a client-side timer.

The scaling knobs below deliberately reproduce the UN-TUNED config the
burst in trace.json was measured under - they are not a recommendation.
keep_alive=300 in particular is on no Pareto-optimal point in frontier.json
and README.md's "Cost & tuning" calls it a tax for a burst shaped like this
one; it stays at 300 here anyway, because changing it would desynchronise
this file from the committed trace and break a reader's ability to
redeploy this exact app and reproduce that measurement. See README.md's
"Cost & tuning" for what the frontier says to change, and why, once you've
reproduced the baseline. min_concurrency and concurrency_buffer are both 0,
which also keeps the CI deploy-check to exactly one runner - the only one
harness/check.py can observe and confirm stopped.
"""
import random
import time
import uuid

import fal
from fal.toolkit import Image, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate.",
        examples=["a calico cat riding a bicycle through a sunlit wheat field, film photo"],
    )
    image_size: ImageSizeInput = Field(
        default="landscape_4_3",
        description="A preset name or a custom {width, height}.",
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description=(
            "Denoising steps. klein is distilled for 4. Keep this fixed across "
            "a burst - varying it varies service time, which makes the "
            "measured trace uncalibratable."
        ),
    )
    seed: int | None = Field(
        default=None, description="Random when omitted; the response echoes it."
    )


class Output(BaseModel):
    image: Image = Field(description="The generated image, as a CDN URL.")
    seed: int = Field(description="The seed that produced this image.")
    served_by: str = Field(
        description="Opaque per-runner id. Distinct values in one burst = distinct runners."
    )
    was_cold: bool = Field(
        description="True if this was the first request this runner ever served."
    )
    setup_duration_s: float = Field(
        description="How long this runner's setup() took - its measured cold start."
    )


class TuneAutoscaling(fal.App, keep_alive=300):  # 300s: matches trace.json's measured config, see the docstring
    machine_type = "GPU-H100"  # PINNED: GPU-L40 has no published price
    app_auth = "private"
    startup_timeout = 1200  # the account's first-ever boot pulls ~16 GB to /data

    # The un-tuned config this example's burst was measured under - not a
    # recommendation. See README.md "Cost & tuning" for what the frontier
    # says to change instead, and why.
    min_concurrency = 0  # bursty traffic means long quiet periods: scale to zero
    max_concurrency = 4  # spend ceiling; the burst queues rather than surprising you
    concurrency_buffer = 0  # no idle buffer while ramping; README.md prices raising it

    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
    ]

    def setup(self) -> None:
        """Load -> warmup, timing itself so the runner can report its own cold start."""
        import os

        started = time.monotonic()
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # before any HF download

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )

        # Identity and timing for the measurement. requests_served is what
        # makes was_cold meaningful: only the first request on this runner
        # paid for the setup above.
        self.worker_id = uuid.uuid4().hex[:12]
        self.setup_duration_s = time.monotonic() - started
        self.requests_served = 0

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch

        was_cold = self.requests_served == 0
        self.requests_served += 1

        size = get_image_size(request.image_size)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        result = self.pipe(
            prompt=request.prompt,
            height=size.height,
            width=size.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0,  # klein is guidance-distilled; keep at 1.0
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        return Output(
            image=Image.from_pil(result.images[0], repository="fal_v3"),
            seed=seed,
            served_by=self.worker_id,
            was_cold=was_cold,
            setup_duration_s=round(self.setup_duration_s, 3),
        )
