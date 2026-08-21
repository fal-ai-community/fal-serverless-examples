"""Serve a 20+ GB video model (LTX-2.3 22B distilled) on one H200."""

import asyncio
import base64
import json
import os
import queue
import random
import shutil
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fal
import loader
from fal.toolkit import File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import torch

# (width, height); both must be divisible by 32 (pipeline requirement)
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "768x512": (768, 512),
    "512x768": (512, 768),
    "1024x576": (1024, 576),
}
FRAMES_PER_SECOND = 24
# VAE packs 8 frames per latent frame; outputs live on an 8k+1 grid.
TEMPORAL_FACTOR = 8


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. Longer, cinematic prompts work well.",
        examples=["a red fox trotting through fresh snow at golden hour"],
        max_length=2000,
    )
    resolution: Literal["768x512", "512x768", "1024x576"] = Field(
        default="768x512", description="Output size (width x height)."
    )
    num_frames: int = Field(
        default=57,
        ge=9,
        le=121,
        description=(
            f"Frame count at {FRAMES_PER_SECOND} fps; snapped to the model's 8k+1 grid."
        ),
    )
    seed: int | None = Field(
        default=None, description="Reproducible seed; random if unset."
    )
    preview_every: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Stream a preview every N denoising steps (/stream only).",
    )


class Output(BaseModel):
    # No sync_mode on purpose: videos are too big for data-URI transport.
    video: File = Field(description="Final video with audio, hosted on the fal CDN.")
    seed: int = Field(description="Seed actually used.")
    num_frames: int = Field(
        description="Effective frame count after snapping to the model's 8k+1 grid."
    )
    timings: dict[str, float] = Field(description="Server-side stage timings, seconds.")


# keep_alive=300: sessions come in bursts and an idle H200 bills.
class LargeVideoModel(fal.App, keep_alive=300):
    machine_type = "GPU-H200"  # ~70 GB resident bf16 weights need H200-class VRAM
    app_auth = "private"
    # first-ever boot downloads ~88 GB to /data (~400 s cold)
    startup_timeout = 1800
    local_python_modules = [
        "loader"
    ]  # pure-python, safe to import in the deploying env
    # taehv.py imports torch at module top, which local_python_modules would run
    # in CI's torch-less deploy env - ship it via app_files instead; the explicit
    # context_dir keeps the path example-relative under any loader.
    app_files = ["taehv.py"]
    app_files_context_dir = str(Path(__file__).resolve().parent)
    # Authoritative deploy-time deps; requirements.txt mirrors exactly (CI-enforced).
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        "torch==2.12.1",
        "diffusers==0.39.0",  # LTX2Pipeline needs >=0.37
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "av==16.0.1",  # needed by diffusers.utils.encode_video (mp4+audio mux)
        "hf-transfer==0.1.9",
        "huggingface_hub==1.27.0",
        "numpy==2.4.6",
    ]

    def setup(self) -> None:
        import torch
        from diffusers import LTX2Pipeline
        from taehv import TAEHV

        self.timings: dict[str, float] = {}
        loader.tune_hf_env()
        t0 = time.perf_counter()
        model_dir = loader.fetch_model()
        tae_path = loader.fetch_taehv()
        self.timings["fetch_s"] = round(time.perf_counter() - t0, 1)
        self.timings["page_cache_s"] = round(loader.warm_page_cache(model_dir), 1)

        t0 = time.perf_counter()
        self.pipe = LTX2Pipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.tiny_vae = TAEHV(str(tae_path)).to("cuda", torch.bfloat16)
        self.timings["load_s"] = round(time.perf_counter() - t0, 1)

        # Warm up at the default shape: autotune is shape-dependent, so a
        # smaller warmup shape would not spare real requests the compile tax.
        t0 = time.perf_counter()
        path, _, _, _ = self._generate(Input(prompt="warmup"), preview_cb=None)
        shutil.rmtree(Path(path).parent, ignore_errors=True)
        self.timings["warmup_s"] = round(time.perf_counter() - t0, 1)
        print(f"[setup] {self.timings}")

    def _generate(
        self, request: Input, preview_cb: Callable[..., dict] | None
    ) -> tuple[str, int, int, dict[str, float]]:
        """Run one distilled 8-step generation.

        Returns (mp4 path, seed, num_frames, timings).
        """
        import torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        width, height = RESOLUTIONS[request.resolution]
        num_frames = snap_frames(request.num_frames)
        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        try:
            video, audio = self.pipe(
                prompt=request.prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                sigmas=DISTILLED_SIGMA_VALUES,  # distilled schedule - fixed 8 steps
                num_inference_steps=8,
                guidance_scale=1.0,  # distilled = CFG off
                use_cross_timestep=True,  # LTX-2.3 conditioning
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="np",
                return_dict=False,
                callback_on_step_end=preview_cb,
            )
        except torch.cuda.OutOfMemoryError as exc:
            # Not fal.exceptions.GPUOutOfMemoryException: its 503 restarts the
            # runner, discarding 88 GB of warm weights over one oversized request.
            torch.cuda.empty_cache()
            raise fal.exceptions.AppException(
                message="Out of GPU memory - lower num_frames or resolution and retry.",
                status_code=422,
            ) from exc
        timings["generate_s"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        # Own dir per request so the file can be named output.mp4 - fal derives
        # the response file_name from the path; callers rmtree the parent after upload.
        out_path = str(Path(tempfile.mkdtemp(prefix="ltx2-")) / "output.mp4")
        waveform = audio[0]
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(waveform)
        encode_video(
            video[0],
            FRAMES_PER_SECOND,
            out_path,
            audio=waveform.float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
        )
        timings["encode_s"] = round(time.perf_counter() - t0, 1)
        return out_path, seed, num_frames, timings

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        path, seed, num_frames, timings = self._generate(request, preview_cb=None)
        try:
            # Explicit content_type: File.from_path defaults to
            # application/octet-stream, so browsers download instead of play.
            return Output(
                video=File.from_path(path, content_type="video/mp4"),
                seed=seed,
                num_frames=num_frames,
                timings=timings,
            )
        finally:
            shutil.rmtree(Path(path).parent, ignore_errors=True)

    @fal.endpoint("/stream")
    async def stream(self, request: Input) -> StreamingResponse:
        async def event_source():
            # Session and job start inside the generator: a never-iterated body
            # must not leave a worker thread and a GPU generation running.
            width, height = RESOLUTIONS[request.resolution]
            session = _PreviewSession(
                self,
                height,
                width,
                snap_frames(request.num_frames),
                request.preview_every,
            )
            loop = asyncio.get_running_loop()
            job = loop.run_in_executor(
                None, lambda: self._generate(request, preview_cb=session.on_step_end)
            )
            try:
                while True:
                    while not session.events.empty():
                        yield f"data: {json.dumps(session.events.get())}\n\n"
                    if job.done():
                        break
                    await asyncio.sleep(0.05)
                # Freeze the queue, then drain: every preview reaches the client
                # and nothing follows the terminal event.
                await loop.run_in_executor(
                    None, session.finish
                )  # blocking join off the loop
                while not session.events.empty():
                    yield f"data: {json.dumps(session.events.get())}\n\n"
                try:
                    path, seed, num_frames, timings = job.result()
                    # from_path_async, not from_path: the 1-2 s upload would block
                    # the event loop. Everything that can fail stays inside this
                    # try; only the yield is outside.
                    try:
                        final = Output(
                            video=await File.from_path_async(
                                path, content_type="video/mp4"
                            ),
                            seed=seed,
                            num_frames=num_frames,
                            timings=timings,
                        )
                    finally:
                        shutil.rmtree(Path(path).parent, ignore_errors=True)
                    result = json.dumps(
                        {"type": "result", **final.model_dump(mode="json")}
                    )
                except Exception as exc:
                    # The 200 is already sent: raising would truncate the stream and
                    # x-fal-sse-last-event-as-body would surface a preview as the body,
                    # so emit the failure as the last event. CancelledError is a
                    # BaseException, so cancellation still propagates.
                    traceback.print_exc()
                    if isinstance(exc, fal.exceptions.AppException):
                        detail, status = exc.message, exc.status_code
                    else:
                        detail, status = f"{type(exc).__name__}: {exc}", 500
                    err = json.dumps(
                        {"type": "error", "detail": detail, "status_code": status}
                    )
                    yield f"data: {err}\n\n"
                    return
                yield f"data: {result}\n\n"
            finally:
                session.close()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"x-fal-sse-last-event-as-body": "1"},
        )


def snap_frames(n: int) -> int:
    """LTX generates on an 8k+1 frame grid; snap down so users never 422 on it."""
    return max(9, ((n - 1) // TEMPORAL_FACTOR) * TEMPORAL_FACTOR + 1)


class _PreviewSession:
    """Latest-wins: callback never blocks, only stores latents; one worker decodes."""

    def __init__(
        self,
        app: LargeVideoModel,
        height: int,
        width: int,
        num_frames: int,
        every: int,
    ):
        self.app = app
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.every = every
        self.events: queue.Queue[dict] = queue.Queue()
        self.done = threading.Event()
        self._latest: tuple[int, int, torch.Tensor] | None = None  # (step, total, x0)
        self._lock = threading.Lock()
        self._prev: tuple[torch.Tensor, float] | None = None  # (latents, sigma)
        self._worker = threading.Thread(target=self._decode_loop, daemon=True)
        self._worker.start()

    # runs on the denoise thread
    def on_step_end(self, pipe, i, t, callback_kwargs):
        try:
            latents = callback_kwargs["latents"]
            sigma = float(pipe.scheduler.sigmas[i + 1])
            total = len(pipe.scheduler.sigmas) - 1
            if self._prev is not None and i % self.every == 0 and i < total - 1:
                prev_latents, prev_sigma = self._prev
                # Flow-matching latents move on straight paths: extrapolate two
                # consecutive states to sigma=0 for a clean-sample estimate.
                x0 = latents - sigma * (latents - prev_latents) / (sigma - prev_sigma)
                with self._lock:
                    self._latest = (i, total, x0.detach().clone())
            self._prev = (latents.detach().clone(), sigma)
        except Exception as exc:  # previews must never break generation
            print(f"[preview] callback error at step {i}: {exc!r}")
        return {}

    # runs on the decode worker thread
    def _decode_loop(self):
        while not self.done.is_set():
            with self._lock:
                job, self._latest = self._latest, None
            if job is None:
                time.sleep(0.02)
                continue
            step, total, x0 = job
            # Both threads use the default CUDA stream, so the callback's clone is
            # ordered before these decode kernels - no explicit sync needed.
            try:
                self.events.put(self._render_event(step, total, x0))
            except Exception as exc:
                print(f"[preview] decode error at step {step}: {exc!r}")

    def _render_event(self, step: int, total: int, x0) -> dict:
        import torch
        from diffusers import LTX2Pipeline
        from diffusers.utils import encode_video

        tae = self.app.tiny_vae
        latent_frames = (self.num_frames - 1) // TEMPORAL_FACTOR + 1
        with torch.inference_mode():
            lat = LTX2Pipeline._unpack_latents(
                x0, latent_frames, self.height // 32, self.width // 32, 1, 1
            )
            # TAEHV takes diffusion-space latents - do NOT apply
            # _denormalize_latents here (saturates its input clamp).
            rgb = tae.decode_video(  # NTCHW in, NTCHW [0,1] out, first 7 frames trimmed
                lat.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                parallel=True,
                show_progress_bar=False,
            )
            # half-res previews: ~4x smaller events
            rgb = torch.nn.functional.interpolate(
                rgb[0].float(), scale_factor=0.5, mode="area"
            )
            frames = (rgb.clamp(0, 1).permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        try:
            # .contiguous() is required by av.VideoFrame.from_ndarray
            encode_video(frames.contiguous(), FRAMES_PER_SECOND, path)
            payload = Path(path).read_bytes()
        finally:
            if os.path.exists(path):
                os.unlink(path)
        return {
            "type": "preview",
            "step": step,
            "total_steps": total,
            "progress": round((step + 1) / total, 3),
            # File-shaped like Output.video so clients parse one shape for
            # previews and the final result; only the url scheme differs.
            "video": {
                "url": "data:video/mp4;base64," + base64.b64encode(payload).decode(),
                "content_type": "video/mp4",
                "file_size": len(payload),
            },
            "width": frames.shape[2],
            "height": frames.shape[1],
            "frames": int(frames.shape[0]),
        }

    def finish(self, timeout: float = 30.0):
        """Stop the worker and wait; afterwards `events` is complete and frozen."""
        self.done.set()
        self._worker.join(timeout=timeout)

    def close(self):
        # Runs from the generator's `finally`, on client disconnect that is the
        # event loop thread - cap the join rather than stall /health.
        self.finish(timeout=5.0)
