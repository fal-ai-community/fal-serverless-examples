"""Multi-GPU video inference - WAN 2.2 A14B sharded across 4 GPUs with fal.distributed.

Catalog slot 04.5. One frontier text-to-video model, one video,
four GPUs. DistributedRunner spawns a worker per GPU and initializes
torch.distributed (NCCL) before setup() runs; diffusers context parallelism
(Ulysses sequence parallelism) then splits every denoising step's sequence
across the ranks, so all four GPUs cooperate on the same clip.

Model: Wan-AI/Wan2.2-T2V-A14B-Diffusers - Apache-2.0, ungated. The MoE pair
of 14B expert transformers is ~70 GB in bf16: this is what "needs multi-GPU"
actually looks like.
"""
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import fal
from fal.distributed import DistributedRunner, DistributedWorker
from fal.toolkit import File
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
NUM_GPUS = 4

# The HF repo stores fp32 (~126 GB). Each of the 4 workers reads the full
# weights (sequence parallelism replicates them), so we convert to bf16 once
# on /data and every later boot reads half the bytes. /data persists
# account-wide across machines and deploys.
CONVERTED_DIR = "/data/models/wan22-t2v-a14b-bf16"

# WAN generates at 16 fps; sizes are (height, width).
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "480p": (480, 832),
    "720p": (720, 1280),
}


class Input(BaseModel):
    prompt: str = Field(
        description="What happens in the video. Long, specific prompts steer WAN best.",
        examples=["a red panda napping on a mossy branch, soft morning light"],
    )
    negative_prompt: str = Field(
        default="",
        description="What to avoid in the video.",
    )
    resolution: Literal["480p", "720p"] = Field(
        default="480p",
        description="480p (832x480) for fast iteration; 720p (1280x720) for final quality.",
        examples=["480p"],
    )
    num_frames: int = Field(
        default=33,
        ge=9,
        le=81,
        description="Clip length in frames at 16 fps; must satisfy (n - 1) % 4 == 0. 81 is ~5 s.",
        examples=[33],
    )
    num_inference_steps: int = Field(
        default=6,
        ge=2,
        le=40,
        description="Denoising steps. 6 is a fast preview; WAN's native quality default is 27.",
        examples=[6],
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance for the high-noise expert.",
    )
    guidance_scale_2: float = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance for the low-noise expert.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random when omitted. Echoed back; same seed + same inputs reproduces the clip.",
    )

    @field_validator("num_frames")
    @classmethod
    def _frames_on_grid(cls, v: int) -> int:
        if (v - 1) % 4 != 0:
            raise ValueError(
                "num_frames must satisfy (num_frames - 1) % 4 == 0, e.g. 9, 17, 33, 49, 81"
            )
        return v


class Output(BaseModel):
    # No sync_mode on purpose: multi-hundred-MB clips make data-URI transport inappropriate.
    video: File = Field(description="The generated clip (mp4, 16 fps) on the fal CDN.")
    seed: int = Field(description="The seed that produced this video.")
    timings: dict[str, float] = Field(
        description="Wall-clock seconds per phase; 'generate' is the four-GPU denoise."
    )


class WanWorker(DistributedWorker):
    """One process per GPU. torch.distributed is already initialized here."""

    def setup(self, **kwargs) -> None:
        import torch
        from diffusers import ContextParallelConfig, WanPipeline

        # Every rank loads the full pipeline (sequence parallelism shards the
        # sequence, not the weights) from the bf16 copy setup() prepared.
        pipe = WanPipeline.from_pretrained(CONVERTED_DIR, torch_dtype=torch.bfloat16)

        # Ulysses attention across the runner's process group: each denoising
        # step's sequence is split over world_size ranks.
        cp = ContextParallelConfig(ulysses_degree=self.world_size)
        pipe.transformer.enable_parallelism(config=cp)
        pipe.transformer_2.enable_parallelism(config=cp)

        pipe.vae.enable_tiling()
        # The A14B MoE experts run in disjoint denoising stages; sequenced CPU
        # offload keeps only the active component resident (~30 GB of 80 GB).
        pipe.enable_model_cpu_offload(device=self.device)
        self.pipe = pipe

    def __call__(
        self,
        streaming: bool = False,  # force-injected by the runner - accept and ignore
        *,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_scale_2: float,
        seed: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import torch
        from diffusers.utils import export_to_video

        # Context parallelism needs identical inputs on every rank, so the
        # endpoint picks one seed and all ranks use it (unlike data
        # parallelism, where each rank would take a different seed).
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.perf_counter()
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            generator=generator,
            output_type="np",
        )
        generate_seconds = round(time.perf_counter() - t0, 1)

        if self.rank != 0:
            # Ranks cooperated via NCCL inside the pipeline call; only rank 0
            # materializes the artifact and replies.
            return {}

        path = f"/tmp/wan22-{uuid.uuid4().hex}.mp4"
        export_to_video(result.frames[0], path, fps=16)
        return {"video_path": path, "timings": {"generate": generate_seconds}}


class MultiGpuVideo(fal.App, keep_alive=300):
    machine_type = "GPU-H100"
    num_gpus = NUM_GPUS  # class attribute, not a class kwarg
    app_auth = "private"
    # First-ever boot on an account downloads ~126 GB AND converts to bf16
    # (~35 min total); every boot after that loads the converted copy (~8 min).
    startup_timeout = 3600
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "pyzmq==27.1.0",  # fal.distributed's transport; not a dependency of fal itself
        "hf-transfer==0.1.9",
        "ftfy==6.3.1",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
    ]

    async def setup(self) -> None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # One-time per account; see CONVERTED_DIR above. The directory only
        # ever appears via the atomic rename below, so existence implies a
        # complete copy.
        if not Path(CONVERTED_DIR).exists():
            import torch
            from diffusers import WanPipeline
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL_ID, revision=MODEL_REVISION)
            pipe = WanPipeline.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
            )
            # Unique staging dir: /data is account-wide, so concurrent first
            # boots must not write into the same path.
            tmp_dir = f"{CONVERTED_DIR}.tmp.{uuid.uuid4().hex}"
            pipe.save_pretrained(tmp_dir, safe_serialization=True)
            del pipe
            try:
                os.rename(tmp_dir, CONVERTED_DIR)
            except OSError:
                # Another boot won the race and already published its copy.
                shutil.rmtree(tmp_dir)

        self.runner = DistributedRunner(worker_cls=WanWorker, world_size=NUM_GPUS)
        await self.runner.start(timeout=1800)

        # Warm up the whole distributed path before the first request: worker
        # errors surface as an "error" key, not an exception. Warming at the
        # default request shape (33 frames @ 480p) matters - attention kernels
        # autotune per shape, and a cold shape adds ~40 s to the first request.
        warmup = await self.runner.invoke(
            _worker_payload(Input(prompt="warmup", num_inference_steps=2), seed=0)
        )
        assert "error" not in warmup, f"warmup failed: {warmup}"
        os.remove(warmup["video_path"])

    @fal.endpoint("/")
    async def generate(self, request: Input) -> Output:
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        result = await self.runner.invoke(_worker_payload(request, seed))
        if "error" in result:
            # 5xx: not billed, and the platform retries on healthy hardware.
            raise HTTPException(status_code=500, detail=f"generation failed: {result['error']}")

        try:
            video = File.from_path(result["video_path"], content_type="video/mp4")
        finally:
            # keep_alive runners live on: never leave multi-MB clips in /tmp.
            os.remove(result["video_path"])
        return Output(video=video, seed=seed, timings=result["timings"])


def _worker_payload(request: Input, seed: int) -> dict[str, Any]:
    height, width = RESOLUTIONS[request.resolution]
    return {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "height": height,
        "width": width,
        "num_frames": request.num_frames,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "guidance_scale_2": request.guidance_scale_2,
        "seed": seed,
    }
