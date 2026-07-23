"""Cut cold starts - the BASELINE app: every cold start pays full price.

Catalog slot 02.1 (SERV-1286). The same SD-Turbo endpoint as app.py with
none of the levers pulled: weights are re-downloaded into container-local
disk on every boot (deliberately bypassing /data, to show what any
freshly-pulled container does), deserialized from safetensors, and
torch.compile recompiles its kernels from scratch every single time.

Nothing here is wrong, exactly - it is what a straightforward first draft
looks like. bench.py deploys this next to app.py and measures the
difference. This file is not deploy-checked by CI (the harness checks
app.py); it exists to be benchmarked against.
"""
import random
import tempfile
import time
from pathlib import Path

import fal
from fal.toolkit import Image
from pydantic import BaseModel, Field

MODEL_ID = "stabilityai/sd-turbo"
RESOLUTION = 512  # SD-Turbo's native resolution; fixed so warmup covers all shapes
# Only the fp16 weights + configs: identical bytes to what the optimized app
# packs, so the two apps differ in strategy, not payload.
DOWNLOAD_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "*/config.json",
    "*.fp16.safetensors",
]


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. SD-Turbo trades fidelity for speed.",
        examples=["a calico cat riding a bicycle, film photo"],
    )
    num_inference_steps: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Denoising steps. SD-Turbo is distilled for 1; up to 4 sharpens details.",
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
    generation_ms: int = Field(
        description="Server-side model time for this request (excludes upload)."
    )


class StageTiming(BaseModel):
    name: str = Field(description="Setup stage name.")
    seconds: float = Field(description="Wall-clock duration of the stage.")


class SetupReportInput(BaseModel):
    """No parameters - POST an empty object."""


class SetupReport(BaseModel):
    variant: str = Field(description="'baseline' or 'optimized'.")
    gpu: str = Field(description="GPU this worker booted on.")
    built_caches_this_boot: bool = Field(
        description="Always False here: the baseline builds nothing reusable."
    )
    stages: list[StageTiming] = Field(description="Per-stage setup timings, in order.")
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartBaseline(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # same pin as app.py so the comparison is like-for-like
    app_auth = "private"
    startup_timeout = 1200  # every boot downloads AND recompiles
    # Authoritative deploy-time deps - identical to app.py so both apps share
    # one cached environment and the benchmark isolates the three levers.
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
    ]

    def setup(self) -> None:
        """Download -> deserialize -> recompile: the full bill, every boot."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append({"name": name, "seconds": round(time.perf_counter() - start, 2)})

        t0 = time.perf_counter()
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import snapshot_download
        stage("imports", t0)

        # Weights land on container-local disk (an explicit local_dir bypasses
        # the /data-backed HF cache fal provides). The container is gone after
        # this worker dies, so the next cold start downloads all ~2.6 GB again.
        t0 = time.perf_counter()
        ephemeral_dir = Path(tempfile.mkdtemp(prefix="sd-turbo-ephemeral-"))
        snapshot_download(MODEL_ID, local_dir=ephemeral_dir, allow_patterns=DOWNLOAD_PATTERNS)
        stage("download weights (every boot)", t0)

        # Standard safetensors deserialization: read on the CPU, then copy
        # tensor-by-tensor onto the GPU.
        t0 = time.perf_counter()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            ephemeral_dir,  # a plain local directory - no Hub round-trip here
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        stage("load weights (safetensors -> GPU)", t0)

        # torch.compile with no shared cache: inductor compiles every kernel
        # from scratch on every worker, every cold start.
        t0 = time.perf_counter()
        self.pipe.unet = torch.compile(self.pipe.unet, dynamic=True)
        for steps in (1, 4):
            self.pipe(
                prompt="warmup",
                height=RESOLUTION,
                width=RESOLUTION,
                num_inference_steps=steps,
                guidance_scale=0.0,
            )
        stage("compile + warmup (recompiled every boot)", t0)

        self.setup_stages = stages
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        t0 = time.perf_counter()
        result = self.pipe(
            prompt=request.prompt,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo is distilled without CFG; keep at 0.0
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        generation_ms = int((time.perf_counter() - t0) * 1000)
        repository = "in_memory" if request.sync_mode else "fal_v3"
        image = Image.from_pil(result.images[0], repository=repository)
        return Output(image=image, seed=seed, generation_ms=generation_ms)

    @fal.endpoint("/setup-report")
    def setup_report(self, request: SetupReportInput) -> SetupReport:
        """What this worker's cold start actually spent its time on."""
        return SetupReport(
            variant="baseline",
            gpu=self.gpu_name,
            built_caches_this_boot=False,
            stages=[StageTiming(**s) for s in self.setup_stages],
            total_seconds=self.setup_total,
        )
