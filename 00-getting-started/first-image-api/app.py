"""Your first image API - FLUX.2 [klein] 4B on a fal GPU.

Catalog slot 00.2 (SERV-1293). A real text-to-image API: weights load once
per worker in setup(), a warmup generation runs before the first request,
and the endpoint returns a CDN-hosted image the playground renders.

Model: black-forest-labs/FLUX.2-klein-4B - Apache-2.0 and ungated (no HF
token anywhere). Careful: the 9B klein variants are gated + non-commercial;
do not swap them in here.
"""
import random

import fal
from fal.toolkit import Image, ImageSizeInput, get_image_size
from pydantic import BaseModel, Field

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. klein handles long, specific prompts well.",
        examples=["a calico cat riding a bicycle through a sunlit wheat field, film photo"],
    )
    image_size: ImageSizeInput = Field(
        default="landscape_4_3",
        description="A preset name or a custom {width, height} (multiples of 32 work best).",
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Denoising steps. klein is distilled for 4; more buys little.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Random when omitted. The response echoes the seed used - the same "
            "seed with the same inputs reproduces the same image."
        ),
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


class FirstImageApi(fal.App, keep_alive=300):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    startup_timeout = 1200  # the account's first-ever boot downloads ~16 GB to /data
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
        """Load -> (compile) -> warmup: everything slow happens once per worker."""
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # before any HF download

        import torch
        from diffusers import Flux2KleinPipeline

        # LOAD: weights land in the account-wide /data cache (fal points
        # HF_HOME there), so only the first boot ever downloads from the Hub.
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        # COMPILE would go here. Skipped in this first example - torch.compile
        # plus shared inductor caches are cold-start engineering, covered by
        # example 02.1 with the trade-off measured.

        # WARMUP: pay CUDA init and kernel autotune now, not on the first
        # user request. One step at full resolution is enough.
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch

        size = get_image_size(request.image_size)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        result = self.pipe(
            prompt=request.prompt,  # keyword-only in Flux2KleinPipeline
            height=size.height,
            width=size.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0,  # klein is guidance-distilled; keep at 1.0
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        repository = "in_memory" if request.sync_mode else "fal_v3"
        image = Image.from_pil(result.images[0], repository=repository)
        return Output(image=image, seed=seed)
