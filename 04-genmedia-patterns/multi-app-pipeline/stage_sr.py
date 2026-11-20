"""Pipeline stage 3: Real-ESRGAN x2 on its own GPU upscales a clip frame by frame."""

import tempfile
import time
from pathlib import Path

import fal
from fal.toolkit import File, download_file, download_model_weights
from pydantic import BaseModel, Field

WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)
SCALE = 2
BATCH = 4
MAX_FRAMES = 241
# fal-hosted 1920x1080 sample; CI must not hit third-party servers.
SAMPLE_VIDEO = (
    "https://storage.googleapis.com/falserverless/example_outputs/pika/"
    "pika_i2v_v22_output.mp4"
)


class Input(BaseModel):
    video_url: str = Field(
        description="MP4 to upscale (at most 241 frames).",
        examples=[SAMPLE_VIDEO],
    )


class Output(BaseModel):
    video: File = Field(description="The upscaled MP4, 2x width and height.")
    width: int = Field(description="Output width in pixels.")
    height: int = Field(description="Output height in pixels.")
    compute_seconds: float = Field(description="Time spent inside this stage.")


class PipelineSR(fal.App, name="video-pipeline-sr", keep_alive=300):
    # RRDBNet is ~17M params; any 80 GB card fits a batch of 480p frames easily.
    machine_type = "GPU-A100"
    app_auth = "private"
    startup_timeout = 900
    requirements = [
        "fal==1.78.2",
        "torch==2.12.1",
        "torchvision==0.27.1",  # spandrel needs it; must match the torch pin
        "spandrel==0.4.2",
        "einops==0.8.1",
        "safetensors==0.6.2",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
        "numpy==2.3.5",
    ]

    def setup(self) -> None:
        import torch
        from spandrel import ModelLoader

        weights = download_model_weights(WEIGHTS_URL)
        descriptor = ModelLoader().load_from_file(weights)
        assert descriptor.scale == SCALE, f"unexpected scale {descriptor.scale}"
        self.model = descriptor.model.eval().half().to("cuda")
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 64, 64, device="cuda", dtype=torch.half))

    @fal.endpoint("/")
    def upscale(self, request: Input) -> Output:
        import imageio.v3 as iio
        import numpy as np
        import torch
        from fastapi import HTTPException

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            src = download_file(request.video_url, target_dir=tmp)
            fps = iio.immeta(src, plugin="FFMPEG").get("fps", 16)
            frames = iio.imread(src, plugin="FFMPEG")
            if len(frames) > MAX_FRAMES:
                raise HTTPException(
                    422, f"video has {len(frames)} frames; max is {MAX_FRAMES}"
                )

            out = []
            with torch.inference_mode():
                for i in range(0, len(frames), BATCH):
                    batch = torch.from_numpy(frames[i : i + BATCH]).to("cuda")
                    batch = batch.permute(0, 3, 1, 2).half().div_(255)
                    up = self.model(batch).clamp_(0, 1).mul_(255).round_()
                    out.append(up.byte().permute(0, 2, 3, 1).cpu().numpy())
            upscaled = np.concatenate(out)

            dst = Path(tmp) / "upscaled.mp4"
            iio.imwrite(
                dst,
                upscaled,
                plugin="FFMPEG",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=8,
            )
            compute = round(time.perf_counter() - t0, 2)
            video = File.from_path(dst, content_type="video/mp4")
        height, width = upscaled.shape[1:3]
        return Output(
            video=video, width=int(width), height=int(height), compute_seconds=compute
        )
