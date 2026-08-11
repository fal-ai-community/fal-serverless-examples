"""Serve a 20+ GB video model - LTX-2.3 22B distilled on one H200.

Catalog slot 02.3 (SERV-1303). The three lessons:
  1. loader.py - parallel download + page-cache pre-read for ~88 GB of weights
  2. GPU sizing - ~70 GB resident bf16 needs H200-class VRAM (see README math)
  3. /stream    - intermediate-step tiny-VAE previews over SSE (Task 5)
"""
import time
from pathlib import Path
from typing import Literal

import fal
from fal.toolkit import File
from pydantic import BaseModel, Field

import loader

# height, width - both divisible by 32 (pipeline requirement)
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "768x512": (512, 768),
    "512x768": (768, 512),
    "1024x576": (576, 1024),
}


def snap_frames(n: int) -> int:
    """LTX generates on an 8k+1 frame grid; snap down so users never 422 on it."""
    return max(9, ((n - 1) // 8) * 8 + 1)


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
        default=57, ge=9, le=121,
        description="Frame count at 24 fps; snapped to the model's 8k+1 grid.",
    )
    seed: int | None = Field(default=None, description="Reproducible seed; random if unset.")
    preview_every: int = Field(
        default=1, ge=1, le=4,
        description="Stream a preview every N denoising steps (/stream only).",
    )


class Output(BaseModel):
    video: File = Field(description="Final video with audio, hosted on the fal CDN.")
    seed: int = Field(description="Seed actually used.")
    timings: dict[str, float] = Field(description="Server-side stage timings, seconds.")


class App(fal.App, keep_alive=300):  # 5 min: video sessions come in bursts; idle H200 bills
    machine_type = "GPU-H200"  # ~70 GB resident bf16 weights - see README "GPU sizing"
    app_auth = "private"
    startup_timeout = 1800  # first-ever boot downloads ~88 GB to /data (spike: 392 s cold)
    local_python_modules = ["loader"]  # pure-python, safe to import in the deploying env
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
        self.pipe = LTX2Pipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
        self.tiny_vae = TAEHV(str(tae_path)).to("cuda", torch.bfloat16)
        self.timings["load_s"] = round(time.time() - t0, 1)

        # Warm up at the smoke shape (Input defaults) so the first request pays
        # no compile/autotune tax - autotune is shape-dependent, so warming at a
        # smaller shape would not help the real requests.
        t0 = time.time()
        self._generate(Input(prompt="warmup"), preview_cb=None)
        self.timings["warmup_s"] = round(time.time() - t0, 1)
        print(f"[setup] {self.timings}")

    def _generate(self, request: Input, preview_cb) -> tuple[str, int, dict[str, float]]:
        """Run one distilled 8-step generation; returns (mp4 path, seed, timings)."""
        import tempfile

        import torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        height, width = RESOLUTIONS[request.resolution]
        num_frames = snap_frames(request.num_frames)
        seed = request.seed if request.seed is not None else int(time.time_ns() % 2**31)
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
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
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
        return Output(video=File.from_path(path), seed=seed, timings=timings)
