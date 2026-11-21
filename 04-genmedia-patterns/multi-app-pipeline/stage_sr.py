"""Pipeline stage 3: Real-ESRGAN x2 on its own GPU upscales a clip frame by frame."""

import tempfile
import time
from pathlib import Path

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, download_file, download_model_weights
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)
SCALE = 2
BATCH = 4
MAX_FRAMES = 241
MAX_DOWNLOAD_MB = 200
# 1080p in -> 4K out is the largest frame this stage accepts.
MAX_PIXELS = 1920 * 1080
# fal-hosted 1920x1080 sample; CI must not hit third-party servers.
SAMPLE_VIDEO = (
    "https://storage.googleapis.com/falserverless/example_outputs/pika/"
    "pika_i2v_v22_output.mp4"
)


class Input(BaseModel):
    video_url: str = Field(
        description="MP4 to upscale: at most 241 frames, 1920x1080, and 200 MB.",
        examples=[SAMPLE_VIDEO],
    )


class Output(BaseModel):
    video: File = Field(description="The upscaled MP4, 2x width and height.")
    width: int = Field(description="Output width in pixels.")
    height: int = Field(description="Output height in pixels.")
    num_frames: int = Field(description="Frames in the output clip.")
    compute_seconds: float = Field(description="Time spent inside this stage.")


def check_source(size: tuple[int, int], num_frames: int | None) -> None:
    """Reject inputs over the limits before decoding a single frame."""
    width, height = size
    if width * height > MAX_PIXELS:
        raise FieldException(
            field="video_url",
            message=f"{width}x{height} is larger than 1920x1080",
            type="video_too_large",
        )
    if num_frames is not None and num_frames > MAX_FRAMES:
        raise FieldException(
            field="video_url",
            message=f"video has {num_frames} frames; the limit is {MAX_FRAMES}",
            type="video_too_long",
        )


class PipelineSR(fal.App, name="video-pipeline-sr", keep_alive=300):
    # RRDBNet is ~17M params; any 80 GB card fits a batch of 1080p frames.
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

    def upscale_batch(self, frames: list[bytes], width: int, height: int) -> list:
        import numpy as np
        import torch

        batch = np.frombuffer(b"".join(frames), dtype=np.uint8)
        batch = batch.reshape(len(frames), height, width, 3)
        with torch.inference_mode():
            x = torch.from_numpy(batch.copy()).to("cuda").permute(0, 3, 1, 2)
            up = self.model(x.half().div_(255)).clamp_(0, 1).mul_(255).round_()
            return list(up.byte().permute(0, 2, 3, 1).contiguous().cpu().numpy())

    @fal.endpoint("/")
    def upscale(self, request: Input) -> Output:
        import imageio_ffmpeg

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                src = download_file(
                    request.video_url, target_dir=tmp, filesize_limit=MAX_DOWNLOAD_MB
                )
            except DownloadError as exc:
                raise FieldException(
                    field="video_url",
                    message=f"could not download: {exc}",
                    type="file_download_error",
                ) from exc

            reader = imageio_ffmpeg.read_frames(str(src))
            meta = next(reader)
            width, height = meta["size"]
            duration = meta.get("duration") or 0
            check_source((width, height), round(meta["fps"] * duration) or None)

            dst = Path(tmp) / "upscaled.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(dst),
                (width * SCALE, height * SCALE),
                fps=meta["fps"],
                codec="libx264",
                quality=8,
                macro_block_size=8,
            )
            writer.send(None)
            count, pending = 0, []
            try:
                for frame in reader:
                    count += 1
                    # Duration metadata can be wrong; the decoded count is what counts.
                    check_source((width, height), count)
                    pending.append(frame)
                    if len(pending) == BATCH:
                        for out in self.upscale_batch(pending, width, height):
                            writer.send(out)
                        pending = []
                if pending:
                    for out in self.upscale_batch(pending, width, height):
                        writer.send(out)
            finally:
                writer.close()
                reader.close()

            compute = round(time.perf_counter() - t0, 2)
            video = File.from_path(dst, content_type="video/mp4")
        return Output(
            video=video,
            width=width * SCALE,
            height=height * SCALE,
            num_frames=count,
            compute_seconds=compute,
        )
