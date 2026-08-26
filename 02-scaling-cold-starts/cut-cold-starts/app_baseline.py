"""The baseline WAN 2.1 app: no cold-start levers pulled."""

import random
import tempfile
import time
from pathlib import Path
from typing import Literal

import fal
from fal.toolkit import File
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# Fixed native resolution: warmup at both frame extremes then covers every
# shape the compiled transformer will ever serve.
WIDTH, HEIGHT = 832, 480
WARMUP_FRAME_COUNTS = (17, 81)
FPS = 16


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. WAN 2.1 rewards detailed, cinematic prompts.",
        examples=["A calico cat surfing a small wave at golden hour, cinematic"],
    )
    negative_prompt: str = Field(
        default="",
        description="What to steer away from (artifacts, styles, objects).",
    )
    num_frames: int = Field(
        default=81,
        ge=17,
        le=81,
        description="Frames at 16 fps; must be 4k+1 (17, 21, ... 81). 81 = 5 s.",
        examples=[17],
    )
    num_inference_steps: int = Field(
        default=28,
        ge=1,
        le=40,
        description="Denoising steps - the quality/latency dial.",
        examples=[8],
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted; echoed back so results reproduce.",
        examples=[42],
    )

    @field_validator("num_frames")
    @classmethod
    def _frames_are_4k_plus_1(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("num_frames must be 4k+1 (17, 21, ..., 81)")
        return v


class Output(BaseModel):
    video: File = Field(
        description="The generated MP4 (832x480 @ 16 fps) on the fal CDN."
    )
    seed: int = Field(description="The seed that produced this video.")
    timings: dict[str, float] = Field(
        description="Server-side stage timings, seconds (excluding upload)."
    )


class SetupReportInput(BaseModel):
    """No parameters - POST an empty JSON object."""


class SetupReportOutput(BaseModel):
    variant: Literal["baseline", "optimized"] = Field(
        description="Which app variant produced this report."
    )
    gpu: str = Field(description="GPU this worker booted on.")
    built_caches_this_boot: bool = Field(
        description="Always False for the baseline: it has no caches to build."
    )
    timings: dict[str, float] = Field(
        description="Per-stage setup timings in order, seconds."
    )
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartBaseline(fal.App, keep_alive=60):
    machine_type = (
        "GPU-H100"  # same pinned GPU as app.py: apples-to-apples before/after
    )
    app_auth = "private"
    startup_timeout = 2700  # ~29 GB Hub download + CPU deserialize + full compile
    # Identical pins to app.py (shared cached env, benchmark isolates the
    # levers); requirements.txt mirrors this list exactly.
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "huggingface_hub==1.27.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        """Everything from scratch, every boot: download, deserialize, compile."""
        timings: dict[str, float] = {}
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            timings[name] = round(time.perf_counter() - start, 2)

        t0 = time.perf_counter()
        import torch
        from diffusers import WanPipeline

        stage("imports", t0)

        # Same serving dtypes as the optimized app's pack, so the two apps
        # serve identical pipelines.
        t0 = time.perf_counter()
        self.pipe = WanPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype={
                "transformer": torch.bfloat16,
                "text_encoder": torch.bfloat16,
                "vae": torch.float32,
                "default": torch.bfloat16,
            },
        )
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        stage("download + load weights (Hub, safetensors -> CPU -> GPU)", t0)

        # Warmup at both frame extremes marks the latent time dim dynamic;
        # output_type='latent' skips VAE decode (compile only sees transformer).
        t0 = time.perf_counter()
        self.pipe.transformer = torch.compile(self.pipe.transformer, dynamic=True)
        for frames in WARMUP_FRAME_COUNTS:
            self.pipe(
                prompt="warmup",
                height=HEIGHT,
                width=WIDTH,
                num_frames=frames,
                num_inference_steps=1,
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(0),
            )
        stage("compile + warmup (no cache)", t0)

        self.setup_timings = timings
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch
        from diffusers.utils import export_to_video

        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        t0 = time.perf_counter()
        frames = self.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=HEIGHT,
            width=WIDTH,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).frames[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            export_to_video(frames, str(path), fps=FPS)
            timings = {"generate_s": round(time.perf_counter() - t0, 2)}
            video = File.from_path(path, content_type="video/mp4")
        return Output(video=video, seed=seed, timings=timings)

    @fal.endpoint("/setup-report")
    def setup_report(self, request: SetupReportInput) -> SetupReportOutput:
        """What this worker's cold start actually spent its time on."""
        return SetupReportOutput(
            variant="baseline",
            gpu=self.gpu_name,
            built_caches_this_boot=False,
            timings=self.setup_timings,
            total_seconds=self.setup_total,
        )
