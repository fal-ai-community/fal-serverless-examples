"""Cut cold starts — the BASELINE app: no levers pulled.

Catalog slot 02.1 (SERV-1286). WAN 2.1 1.3B text-to-video the way everyone
writes it first: every worker boot downloads ~29 GB of fp32 shards from the
Hub, deserializes them through CPU, casts to serving dtypes, moves to GPU,
and compiles kernels from scratch. app.py is the same model and endpoints
with all three cold-start levers pulled; bench.py measures both.
"""
import random
import tempfile
import time
from pathlib import Path

import fal
from fal.toolkit import Video
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WIDTH, HEIGHT = 832, 480      # WAN 2.1 1.3B's native resolution, fixed so
WARMUP_FRAME_COUNTS = (17, 81)  # compile warmup covers every serving shape
FPS = 16


class TextToVideoInput(BaseModel):
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
        ge=4,
        le=50,
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


class TextToVideoOutput(BaseModel):
    video: Video = Field(
        description="The generated MP4 (832x480 @ 16 fps) on the fal CDN."
    )
    seed: int = Field(description="The seed that produced this video.")
    generation_ms: int = Field(
        description="Server-side model time (denoise + decode), excluding upload."
    )


class StageTiming(BaseModel):
    name: str = Field(description="Setup stage name.")
    seconds: float = Field(description="Wall-clock duration of the stage.")


class SetupReportInput(BaseModel):
    """No parameters - POST an empty JSON object."""


class SetupReport(BaseModel):
    variant: str = Field(description="'baseline' or 'optimized'.")
    gpu: str = Field(description="GPU this worker booted on.")
    built_caches_this_boot: bool = Field(
        description="Always False for the baseline: it has no caches to build."
    )
    stages: list[StageTiming] = Field(description="Per-stage setup timings, in order.")
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartBaseline(fal.App, keep_alive=60):
    machine_type = "GPU-H100"
    app_auth = "private"
    startup_timeout = 2700  # ~29 GB Hub download + CPU deserialize + full compile
    # Identical pins to app.py so both share one cached environment and the
    # benchmark isolates the levers, not pip-install noise. requirements.txt
    # mirrors this list exactly (static check enforces it).
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        """Everything from scratch, every boot: download, deserialize, compile."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append(
                {"name": name, "seconds": round(time.perf_counter() - start, 2)}
            )

        t0 = time.perf_counter()
        import torch
        from diffusers import WanPipeline
        stage("imports", t0)

        # The Hub repo ships fp32 (~29 GB); serving dtypes are bf16 for the
        # transformer and text encoder, fp32 for the VAE - the same dtypes
        # the optimized app packs, so the two apps serve identical pipelines.
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

        # Same compiled end state as app.py - but no shared cache, so every
        # boot pays the full kernel compilation. Warmup at both frame-count
        # extremes marks the latent time dim dynamic; output_type='latent'
        # skips VAE decode (compile only cares about transformer shapes).
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

        self.setup_stages = stages
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: TextToVideoInput) -> TextToVideoOutput:
        import torch
        from diffusers.utils import export_to_video

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
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
            generation_ms = int((time.perf_counter() - t0) * 1000)
            video = Video.from_path(path)
        return TextToVideoOutput(video=video, seed=seed, generation_ms=generation_ms)

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
