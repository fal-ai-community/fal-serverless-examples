"""Pipeline stage 2: WAN 2.1 1.3B on its own H100 turns a prompt into a clip."""

import random
import tempfile
import time
from pathlib import Path

import fal
from fal.toolkit import File
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
WIDTH, HEIGHT = 832, 480
FPS = 16


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate.",
        examples=["A red panda naps on a mossy branch, soft morning light, close-up"],
    )
    negative_prompt: str = Field(default="", description="What to avoid.")
    num_frames: int = Field(
        default=33,
        ge=17,
        le=81,
        description="Frames at 16 fps; must be 4k+1 (17, 21, ... 81).",
    )
    num_inference_steps: int = Field(
        default=20, ge=1, le=50, description="Denoising steps."
    )
    seed: int | None = Field(
        default=None, description="Random when omitted; echoed back."
    )

    @field_validator("num_frames")
    @classmethod
    def _frames_are_4k_plus_1(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError("num_frames must be 4k+1 (17, 21, ..., 81)")
        return v


class Output(BaseModel):
    video: File = Field(description="The generated MP4 (832x480 @ 16 fps).")
    seed: int = Field(description="The seed that produced this video.")
    compute_seconds: float = Field(description="Time spent inside this stage.")


class PipelineDiT(fal.App, name="video-pipeline-dit", keep_alive=300):
    # Swap this stage for multi-gpu-video-custom-image when one GPU is too slow.
    machine_type = "GPU-H100"
    app_auth = "private"
    startup_timeout = 1800
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "huggingface_hub==1.27.0",
        "ftfy==6.3.1",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    def setup(self) -> None:
        import torch
        from diffusers import WanPipeline

        self.pipe = WanPipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype={"vae": torch.float32, "default": torch.bfloat16},
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe(
            prompt="warmup",
            height=HEIGHT,
            width=WIDTH,
            num_frames=17,
            num_inference_steps=1,
            output_type="latent",
        )

    @fal.endpoint("/warm")
    def warm(self) -> dict:
        """No-op: the agent calls it so a cold runner boots while earlier stages run."""
        return {}

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
            compute = round(time.perf_counter() - t0, 2)
            video = File.from_path(path, content_type="video/mp4")
        return Output(video=video, seed=seed, compute_seconds=compute)
