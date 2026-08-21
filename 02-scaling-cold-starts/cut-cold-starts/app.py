"""Cut cold starts — the OPTIMIZED app: /data + FlashPack + compiled caches.

The same WAN 2.1 1.3B endpoint as app_baseline.py, with every
cold-start lever pulled:

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
from typing import Literal

import fal
from fal.toolkit import File, synchronized_inductor_cache
from pydantic import BaseModel, Field, field_validator

import wan_flashpack

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# Resolution is fixed at WAN 2.1 1.3B's native 832x480 so warmup at both frame
# extremes covers every shape the compiled transformer will ever serve.
WIDTH, HEIGHT = 832, 480
WARMUP_FRAME_COUNTS = (17, 81)
FPS = 16

# Everything this app persists lives under one versioned /data prefix. Bump
# the version when MODEL_ID, serving dtypes, or the torch pin change - a new
# pin must never load an old pack or old kernels.
PACK_VERSION = "v1"
PACK_DIR = Path(f"/data/examples/cut-cold-starts/wan21-t2v-13b-bf16-{PACK_VERSION}")
INDUCTOR_CACHE_KEY = f"cut-cold-starts-wan21-13b/{PACK_VERSION}"
STALE_BUILD_MAX_AGE_S = 3600
# Written (atomically) only after BOTH caches exist: the FlashPack pack on
# /data AND the synced kernel cache. A worker that crashes mid-setup leaves
# no marker, so the next boot honestly reports itself as cache-building.
READY_MARKER = PACK_DIR / f".ready-{INDUCTOR_CACHE_KEY.replace('/', '-')}"


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
        description="True when this boot found no readiness marker and built "
        "cache state (/data pack and/or compiled kernels). Steady-state boots "
        "find the marker, reuse both, and report False."
    )
    timings: dict[str, float] = Field(
        description="Per-stage setup timings in order, seconds."
    )
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartOptimized(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # pin ONE type: inductor caches are per GPU type
    app_auth = "private"
    startup_timeout = 2700  # covers the account's first-ever boot (download + pack + compile)
    local_python_modules = ["wan_flashpack"]
    requirements = [
        "fal==1.78.2",  # keep in sync with harness/requirements.txt
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
        timings: dict[str, float] = {}
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            timings[name] = round(time.perf_counter() - start, 2)

        t0 = time.perf_counter()
        import torch

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
        # atomic rename: chunked writes straight to /data are pathologically
        # slow (measured figure and provenance in the README), and the rename
        # means concurrently booting workers can never observe a half-written
        # pack - the loser of the race cleans up.
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
                if time.time() - stale.stat().st_mtime > STALE_BUILD_MAX_AGE_S:
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
        # scheduler), so the load stage below rebuilds the whole pipeline from
        # the pack: the text encoder explicitly (see the tie-restoration
        # comment below), then the remaining components via one pipeline call.
        t0 = time.perf_counter()
        # transformers ties encoder.embed_tokens.weight to shared.weight, and
        # the flagship classes deliberately leave the tied name out of the
        # pack (flashpack_ignore_names). flashpack's loader assigns packed
        # tensors by REPLACING each Parameter object, which silently breaks
        # that tie: shared gets a real tensor while encoder.embed_tokens
        # keeps the meta skeleton Parameter, and the pipeline loader's own
        # .to(device) then crashes ("Cannot copy out of meta tensor"). Load
        # the text encoder ourselves, restore the tie through the official
        # embeddings API (module-level sharing), and hand the finished
        # module to the pipeline loader - components passed as kwargs are
        # used as-is and skip the broken path.
        text_encoder = classes["FlashPackUMT5EncoderModel"].from_pretrained_flashpack(
            str(PACK_DIR / "text_encoder"), device="cuda"
        )
        text_encoder.set_input_embeddings(text_encoder.shared)
        pipe = FlashPackWanPipeline.from_pretrained_flashpack(
            str(PACK_DIR), device_map="cuda", text_encoder=text_encoder
        )
        for component_name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipe, component_name)
            leftover = [
                name
                for name, tensor in (
                    # remove_duplicate=False: tied parameters share one
                    # object only until an assignment replaces one side (see
                    # above) - dedup would mask exactly that broken tie.
                    list(module.named_parameters(remove_duplicate=False))
                    + list(module.named_buffers(remove_duplicate=False))
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

        self.setup_timings = timings
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
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
            timings = {"generate_s": round(time.perf_counter() - t0, 2)}
            video = File.from_path(path, content_type="video/mp4")
        return Output(video=video, seed=seed, timings=timings)

    @fal.endpoint("/setup-report")
    def setup_report(self, request: SetupReportInput) -> SetupReportOutput:
        """What this worker's cold start actually spent its time on."""
        return SetupReportOutput(
            variant="optimized",
            gpu=self.gpu_name,
            built_caches_this_boot=self.built_caches_this_boot,
            timings=self.setup_timings,
            total_seconds=self.setup_total,
        )
