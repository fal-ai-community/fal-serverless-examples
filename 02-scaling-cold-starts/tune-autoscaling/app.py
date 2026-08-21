"""The burst target: FLUX.2 klein, self-reporting runner id, coldness and setup time."""

import os
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
        examples=[
            "a calico cat riding a bicycle through a sunlit wheat field, film photo"
        ],
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
    sync_mode: bool = Field(
        default=False,
        description=(
            "Return the image as a data URI instead of a hosted CDN URL - "
            "skips the upload round-trip for latency-sensitive callers."
        ),
    )


class Output(BaseModel):
    image: Image = Field(
        description="The generated image: a CDN URL, or a data URI in sync_mode."
    )
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


class TuneAutoscaling(fal.App, keep_alive=300):
    machine_type = "GPU-H100"  # PINNED: GPU-L40 has no published price
    app_auth = "private"
    startup_timeout = 1200  # the account's first-ever boot pulls ~16 GB to /data

    # The un-tuned config trace.json was measured under - changing it desyncs
    # the committed trace; 0/0 also keeps the CI deploy-check to one runner.
    min_concurrency = 0
    max_concurrency = 4
    concurrency_buffer = 0

    # Authoritative; requirements.txt must mirror this exactly.
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
        started = time.perf_counter()
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

        self.worker_id = uuid.uuid4().hex[:12]
        self.setup_duration_s = time.perf_counter() - started
        self.requests_served = 0

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch

        was_cold = self.requests_served == 0
        self.requests_served += 1

        size = get_image_size(request.image_size)
        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        result = self.pipe(
            prompt=request.prompt,
            height=size.height,
            width=size.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0,  # klein is guidance-distilled; keep at 1.0
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        repository = "in_memory" if request.sync_mode else "fal_v3"
        return Output(
            image=Image.from_pil(result.images[0], repository=repository),
            seed=seed,
            served_by=self.worker_id,
            was_cold=was_cold,
            setup_duration_s=round(self.setup_duration_s, 3),
        )
