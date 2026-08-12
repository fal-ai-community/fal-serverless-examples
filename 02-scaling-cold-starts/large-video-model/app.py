"""Serve a 20+ GB video model - LTX-2.3 22B distilled on one H200.

Catalog slot 02.3 (SERV-1303). The three lessons:
  1. loader.py - parallel download + page-cache pre-read for ~88 GB of weights
  2. GPU sizing - ~70 GB resident bf16 needs H200-class VRAM (see README math)
  3. /stream    - intermediate-step tiny-VAE previews over SSE
"""
import asyncio
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Literal

import fal
import loader
from fal.toolkit import File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# height, width - both divisible by 32 (pipeline requirement)
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "768x512": (512, 768),
    "512x768": (768, 512),
    "1024x576": (576, 1024),
}


def snap_frames(n: int) -> int:
    """LTX generates on an 8k+1 frame grid; snap down so users never 422 on it."""
    return max(9, ((n - 1) // 8) * 8 + 1)


class _PreviewSession:
    """Per-request preview state: x0 extrapolation + latest-wins decode handoff.

    The callback runs on the denoise thread and must never block: it stores
    the newest latents (dropping any undecoded predecessor) and returns.
    A single decode worker drains it; the SSE generator drains `events`.
    """

    def __init__(
        self,
        app: "LargeVideoModel",
        height: int,
        width: int,
        num_frames: int,
        every: int,
    ):
        import queue as _queue
        import threading

        self.app, self.height, self.width, self.num_frames = (
            app,
            height,
            width,
            num_frames,
        )
        self.every = every
        self.events: "_queue.Queue[dict]" = _queue.Queue()
        self.done = threading.Event()
        self._latest: tuple[int, int, object] | None = None  # (step, total, x0_latents)
        self._lock = threading.Lock()
        self._prev: tuple[object, float] | None = None  # (latents, sigma)
        self._worker = threading.Thread(target=self._decode_loop, daemon=True)
        self._worker.start()

    # -- denoise thread ------------------------------------------------------
    def on_step_end(self, pipe, i, t, callback_kwargs):
        try:
            latents = callback_kwargs["latents"]
            sigma = float(pipe.scheduler.sigmas[i + 1])
            total = len(pipe.scheduler.sigmas) - 1
            if self._prev is not None and i % self.every == 0 and i < total - 1:
                prev_latents, prev_sigma = self._prev
                # Flow-matching latents move on straight paths: extrapolate to sigma=0
                # from two consecutive states for a clean-sample estimate.
                x0 = latents - sigma * (latents - prev_latents) / (sigma - prev_sigma)
                with self._lock:
                    self._latest = (i, total, x0.detach().clone())
            self._prev = (latents.detach().clone(), sigma)
        except Exception as exc:  # previews must never break generation
            print(f"[preview] callback error at step {i}: {exc!r}")
        return {}

    # -- decode worker thread --------------------------------------------------
    def _decode_loop(self):
        import time as _time

        while not self.done.is_set():
            with self._lock:
                job, self._latest = self._latest, None
            if job is None:
                _time.sleep(0.02)
                continue
            step, total, x0 = job
            # Both threads submit to the default CUDA stream, so the callback's clone
            # is ordered before these decode kernels - no explicit sync needed (a
            # dedicated torch.cuda.Stream here would need torch.cuda.Event handoff).
            # The win of this thread is keeping the CPU x264 encode off the denoiser.
            try:
                self.events.put(self._render_event(step, total, x0))
            except Exception as exc:
                print(f"[preview] decode error at step {step}: {exc!r}")

    def _render_event(self, step: int, total: int, x0) -> dict:
        import base64
        import os
        import tempfile

        import torch
        from diffusers import LTX2Pipeline
        from diffusers.utils import encode_video

        tae = self.app.tiny_vae
        f_lat = (self.num_frames - 1) // 8 + 1
        with torch.inference_mode():
            lat = LTX2Pipeline._unpack_latents(
                x0, f_lat, self.height // 32, self.width // 32, 1, 1
            )
            # TAEHV consumes diffusion-space latents directly - do NOT apply
            # _denormalize_latents here (saturates its input clamp; spike-verified).
            rgb = tae.decode_video(  # NTCHW in, NTCHW [0,1] out, first 7 frames trimmed
                lat.permute(0, 2, 1, 3, 4).to(torch.bfloat16),
                parallel=True,
                show_progress_bar=False,
            )
            rgb = torch.nn.functional.interpolate(  # half-res previews: ~4x smaller events
                rgb[0].float(), scale_factor=0.5, mode="area"
            )
            frames = (rgb.clamp(0, 1).permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = tmp.name
        try:
            # encode_video accepts a torch uint8 [T,H,W,3] tensor; .contiguous()
            # matters (av.VideoFrame.from_ndarray requirement).
            encode_video(frames.contiguous(), 24, path)
            payload = base64.b64encode(open(path, "rb").read()).decode()
        finally:
            if os.path.exists(path):
                os.unlink(path)
        return {
            "type": "preview",
            "step": step,
            "total_steps": total,
            "progress": round((step + 1) / total, 3),
            "video": f"data:video/mp4;base64,{payload}",
            "width": frames.shape[2],
            "height": frames.shape[1],
            "frames": int(frames.shape[0]),
        }

    def finish(self, timeout: float = 30.0):
        """Stop the worker and wait for any in-flight render to land in `events`.

        After this returns the queue is complete and frozen, so one final drain
        cannot miss a preview that was still encoding when the generation ended.
        A render is ~0.1 s; the generous default is pure paranoia, and callers
        that cannot afford to block pass a smaller bound.
        """
        self.done.set()
        self._worker.join(timeout=timeout)

    def close(self):
        # Runs from the generator's `finally` - on a client disconnect that is the
        # event loop thread, so cap the join hard rather than stalling /health.
        self.finish(timeout=5.0)


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
        description="Frame count at 24 fps; snapped to the model's 8k+1 grid.",
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
    video: File = Field(description="Final video with audio, hosted on the fal CDN.")
    seed: int = Field(description="Seed actually used.")
    timings: dict[str, float] = Field(description="Server-side stage timings, seconds.")


# keep_alive=300: 5 min, because video sessions come in bursts and an idle H200 bills.
class LargeVideoModel(fal.App, keep_alive=300):
    machine_type = "GPU-H200"  # ~70 GB resident bf16 weights - see README "GPU sizing"
    app_auth = "private"
    startup_timeout = (
        1800  # first-ever boot downloads ~88 GB to /data (spike: 392 s cold)
    )
    local_python_modules = [
        "loader"
    ]  # pure-python, safe to import in the deploying env
    # taehv.py imports torch at module top; local_python_modules would make CI's
    # torch-less env import it at deploy time. app_files ships it by path instead;
    # the explicit context_dir keeps it example-relative regardless of the loader
    # (fal run sets local_file_path, harness/check.py's runpy loader does not).
    app_files = ["taehv.py"]
    app_files_context_dir = str(Path(__file__).resolve().parent)
    # Authoritative deploy-time deps; requirements.txt mirrors exactly (CI-enforced).
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
        "torch==2.12.1",
        "diffusers==0.39.0",  # LTX2Pipeline landed in 0.37; 0.39 is current stable
        "transformers==5.3.0",
        "accelerate==1.14.0",
        "av==16.0.1",  # diffusers.utils.encode_video (mp4 mux, incl. audio)
        "hf-transfer==0.1.9",
        "huggingface_hub==1.27.0",
        "numpy==2.4.6",
    ]

    def setup(self) -> None:
        """Load -> warm: every stage timed; these numbers are the example's story."""
        import torch
        from diffusers import LTX2Pipeline
        from taehv import TAEHV

        self.timings: dict[str, float] = {}
        loader.tune_hf_env()
        t0 = time.time()
        model_dir = loader.fetch_model()
        tae_path = loader.fetch_taehv()
        self.timings["fetch_s"] = round(time.time() - t0, 1)
        self.timings["page_cache_s"] = round(loader.warm_page_cache(model_dir), 1)

        t0 = time.time()
        self.pipe = LTX2Pipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.tiny_vae = TAEHV(str(tae_path)).to("cuda", torch.bfloat16)
        self.timings["load_s"] = round(time.time() - t0, 1)

        # Warm up at the smoke shape (Input defaults) so the first request pays
        # no compile/autotune tax - autotune is shape-dependent, so warming at a
        # smaller shape would not help the real requests.
        t0 = time.time()
        self._generate(Input(prompt="warmup"), preview_cb=None)
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        print(f"[setup] {self.timings}")

    def _generate(
        self, request: Input, preview_cb
    ) -> tuple[str, int, dict[str, float]]:
        """Run one distilled 8-step generation; returns (mp4 path, seed, timings)."""
        import tempfile

        import torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        height, width = RESOLUTIONS[request.resolution]
        num_frames = snap_frames(request.num_frames)
        seed = (
            request.seed if request.seed is not None else int(time.time_ns() % 2**31)
        )
        timings: dict[str, float] = {}

        t0 = time.time()
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
            # AppException -> {"detail": message} at this status, worker untouched.
            # Not fal.exceptions.GPUOutOfMemoryException: its 503 tells the platform
            # to restart the runner, which would throw away 88 GB of warm weights
            # over one oversized request.
            torch.cuda.empty_cache()
            raise fal.exceptions.AppException(
                message="Out of GPU memory - lower num_frames or resolution and retry.",
                status_code=422,
            ) from exc
        timings["generate_s"] = round(time.time() - t0, 1)

        t0 = time.time()
        # A whole directory per request so the file itself can be called
        # output.mp4: fal derives the response's `file_name` from the path, and
        # "tmp8f3k1z.mp4" is not a name any caller wants to see. Both endpoints
        # rmtree the parent once the upload is done.
        out_path = str(Path(tempfile.mkdtemp(prefix="ltx2-")) / "output.mp4")
        waveform = audio[0]
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(waveform)
        encode_video(
            video[0],
            24.0,
            out_path,
            audio=waveform.float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
        )
        timings["encode_s"] = round(time.time() - t0, 1)
        return out_path, seed, timings

    @fal.endpoint("/")
    def run(self, request: Input) -> Output:
        path, seed, timings = self._generate(request, preview_cb=None)
        try:
            # content_type is not sniffed from the suffix - File.from_path
            # defaults to application/octet-stream, which makes browsers
            # download the clip instead of playing it.
            return Output(
                video=File.from_path(path, content_type="video/mp4"),
                seed=seed,
                timings=timings,
            )
        finally:
            shutil.rmtree(Path(path).parent, ignore_errors=True)

    @fal.endpoint("/stream")
    async def stream(self, request: Input) -> StreamingResponse:
        async def event_source():
            # Built inside the generator: a response body that is never iterated
            # (client hangs up during headers) must not leave a worker thread and
            # a GPU generation running with nobody to read them.
            height, width = RESOLUTIONS[request.resolution]
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
                # The worker is often still encoding the last preview while the
                # final full-VAE mux runs. Freeze the queue, then drain it: every
                # preview reaches the client, and nothing follows the terminal event.
                await loop.run_in_executor(None, session.finish)  # join off the loop
                while not session.events.empty():
                    yield f"data: {json.dumps(session.events.get())}\n\n"
                try:
                    path, seed, timings = job.result()  # re-raises generation errors
                    # await, not File.from_path: the upload is 1-2 s of blocking I/O
                    # and this coroutine owns the event loop (SSE flushes, /health)
                    # meanwhile. Everything that can fail - generation, upload,
                    # serialization - is inside this try; only the yield is outside.
                    try:
                        final = Output(
                            video=await File.from_path_async(
                                path, content_type="video/mp4"
                            ),
                            seed=seed,
                            timings=timings,
                        )
                    finally:
                        shutil.rmtree(Path(path).parent, ignore_errors=True)
                    result = json.dumps(
                        {"type": "result", **json.loads(final.model_dump_json())}
                    )
                except Exception as exc:
                    # The 200 OK is long gone, so a raise here would just truncate the
                    # stream - and with x-fal-sse-last-event-as-body a non-streaming
                    # client would then read the last *preview* as the response body.
                    # Make the failure the last event instead. (CancelledError is a
                    # BaseException on 3.11+, so cancellation still propagates.)
                    traceback.print_exc()
                    if isinstance(exc, fal.exceptions.AppException):
                        detail, status = exc.message, exc.status_code
                    else:
                        detail, status = f"{type(exc).__name__}: {exc}", 500
                    yield f"data: {json.dumps({'type': 'error', 'detail': detail, 'status_code': status})}\n\n"
                    return
                yield f"data: {result}\n\n"
            finally:
                session.close()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"x-fal-sse-last-event-as-body": "1"},
        )
