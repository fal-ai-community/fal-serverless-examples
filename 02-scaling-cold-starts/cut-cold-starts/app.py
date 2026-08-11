"""Cut cold starts — the OPTIMIZED app: /data + FlashPack + compiled caches.

Catalog slot 02.1 (SERV-1286). The same WAN 2.1 1.3B endpoint as
app_baseline.py, with every cold-start lever pulled:

  1. /data persistent storage - the ~29 GB fp32 repo is downloaded and
     repacked ONCE per account; every later boot skips the network entirely.
  2. FlashPack loading - the ~15 GB pack streams /data -> GPU directly
     (github.com/fal-ai/flashpack), no safetensors CPU deserialization.
  3. synchronized_inductor_cache - torch.compile kernels are built once,
     synced to /data, and every later boot unpacks them instead of compiling.

wan_flashpack.py holds the integration classes - the exact pattern from
FlashPack's own WAN 2.1 flagship example - and ships to the runner via
local_python_modules. bench.py measures this app against the baseline.
"""
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import fal
from fal.toolkit import Video
from pydantic import BaseModel, Field, field_validator

import wan_flashpack

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WIDTH, HEIGHT = 832, 480      # WAN 2.1 1.3B's native resolution, fixed so
WARMUP_FRAME_COUNTS = (17, 81)  # compile warmup covers every serving shape
FPS = 16

# Everything this app persists lives under one versioned /data prefix. Bump
# the version when MODEL_ID, serving dtypes, or the torch pin change - a new
# pin must never load an old pack or old kernels.
PACK_VERSION = "v1"
PACK_DIR = Path(f"/data/examples/cut-cold-starts/wan21-t2v-13b-bf16-{PACK_VERSION}")
INDUCTOR_CACHE_KEY = f"cut-cold-starts-wan21-13b/{PACK_VERSION}"
# Written (atomically) only after BOTH caches exist: the FlashPack pack on
# /data AND the synced kernel cache. A worker that crashes mid-setup leaves
# no marker, so the next boot honestly reports itself as cache-building.
READY_MARKER = PACK_DIR / f".ready-{INDUCTOR_CACHE_KEY.replace('/', '-')}"


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
        description="True when this boot found no readiness marker and built "
        "cache state (/data pack and/or compiled kernels). Steady-state boots "
        "find the marker, reuse both, and report False."
    )
    stages: list[StageTiming] = Field(description="Per-stage setup timings, in order.")
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartOptimized(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # pin ONE type: inductor caches are per GPU type
    app_auth = "private"
    startup_timeout = 2700  # covers the account's first-ever boot (download + pack + compile)
    local_python_modules = ["wan_flashpack"]
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
        """Pack once to /data, stream via FlashPack, compile against the shared cache."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append(
                {"name": name, "seconds": round(time.perf_counter() - start, 2)}
            )

        t0 = time.perf_counter()
        import torch
        from fal.toolkit import synchronized_inductor_cache

        # Builds the FlashPack WAN classes and installs the sys.modules entry
        # that transformers introspection and the pack's model_index.json
        # class resolution both need (see wan_flashpack.py's docstring).
        classes = wan_flashpack.get_classes()
        FlashPackWanPipeline = classes["FlashPackWanPipeline"]
        stage("imports", t0)

        self.built_caches_this_boot = not READY_MARKER.is_file()

        # LEVER 1 - first boot per ACCOUNT (not per worker): download the fp32
        # repo, rebuild at serving dtypes, pack with FlashPack. The pack is
        # written to LOCAL disk and published to /data as one bulk copy +
        # atomic rename: chunked writes straight to /data run at ~0.1 MB/s
        # (measured), and the rename means concurrently booting workers can
        # never observe a half-written pack - the loser of the race cleans up.
        if not PACK_DIR.is_dir():
            t0 = time.perf_counter()
            from diffusers.schedulers import UniPCMultistepScheduler
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            repo_dir = Path(snapshot_download(MODEL_ID))
            transformer = classes["FlashPackWanTransformer3DModel"].from_pretrained(
                repo_dir / "transformer", torch_dtype=torch.bfloat16
            )
            vae = classes["FlashPackAutoencoderKLWan"].from_pretrained(
                repo_dir / "vae", torch_dtype=torch.float32
            )
            text_encoder = classes["FlashPackUMT5EncoderModel"].from_pretrained(
                repo_dir / "text_encoder", torch_dtype=torch.bfloat16
            )
            build_pipe = FlashPackWanPipeline(
                tokenizer=AutoTokenizer.from_pretrained(repo_dir / "tokenizer"),
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                scheduler=UniPCMultistepScheduler.from_pretrained(repo_dir / "scheduler"),
            )
            local_dir = Path(tempfile.mkdtemp(prefix="wan-flashpack-build-")) / "pack"
            local_dir.mkdir(parents=True)
            build_pipe.save_pretrained_flashpack(str(local_dir))
            del build_pipe, transformer, vae, text_encoder
            stage("download + build FlashPack pack (first boot only)", t0)

            t0 = time.perf_counter()
            PACK_DIR.parent.mkdir(parents=True, exist_ok=True)
            for stale in PACK_DIR.parent.glob(".build-*"):  # crashed builds
                if time.time() - stale.stat().st_mtime > 3600:
                    shutil.rmtree(stale, ignore_errors=True)
            publish_dir = PACK_DIR.parent / f".build-{uuid.uuid4().hex}"
            shutil.copytree(local_dir, publish_dir)
            try:
                publish_dir.rename(PACK_DIR)
            except OSError:
                shutil.rmtree(publish_dir, ignore_errors=True)
                if not PACK_DIR.is_dir():
                    # Nobody else published either: a real storage failure,
                    # not a lost race - fail the boot.
                    raise
            shutil.rmtree(local_dir.parent, ignore_errors=True)
            stage("publish pack to /data (first boot only)", t0)

        # LEVER 2 - every boot: the pack streams /data -> GPU. The pack dir is
        # self-contained (model_index.json + per-component packs + tokenizer +
        # scheduler), so this one call rebuilds the whole pipeline.
        t0 = time.perf_counter()
        pipe = FlashPackWanPipeline.from_pretrained_flashpack(
            str(PACK_DIR), device_map="cuda"
        )
        for component_name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipe, component_name)
            leftover = [
                name
                for name, tensor in (
                    list(module.named_parameters()) + list(module.named_buffers())
                )
                if tensor.is_meta
            ]
            # Must run BEFORE any .to(): meta tensors ride through .to()
            # unchanged, so a later check could be masked.
            if leftover:
                raise RuntimeError(
                    f"{component_name}: tensors not materialized from pack: {leftover[:5]}"
                )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        stage("load weights (FlashPack, /data -> GPU)", t0)

        # LEVER 3 - compile against the shared inductor cache. First boot
        # compiles for real and syncs kernels to /data on context exit; every
        # later boot unpacks them and warmup is cache-hit execution. Warmup
        # MUST run inside the context or nothing is captured.
        t0 = time.perf_counter()
        with synchronized_inductor_cache(INDUCTOR_CACHE_KEY):
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
        stage("compile + warmup (shared inductor cache)", t0)

        # Both caches now exist; publish the readiness marker atomically. A
        # crash anywhere above leaves no marker -> next boot rebuilds honestly.
        if self.built_caches_this_boot:
            marker_tmp = READY_MARKER.with_name(
                f"{READY_MARKER.name}.tmp-{uuid.uuid4().hex}"
            )
            marker_tmp.write_text("ok")
            marker_tmp.rename(READY_MARKER)

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
            variant="optimized",
            gpu=self.gpu_name,
            built_caches_this_boot=self.built_caches_this_boot,
            stages=[StageTiming(**s) for s in self.setup_stages],
            total_seconds=self.setup_total,
        )
