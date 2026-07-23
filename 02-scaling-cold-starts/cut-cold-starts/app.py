"""Cut cold starts - the OPTIMIZED app: /data + FlashPack + compiled caches.

Catalog slot 02.1 (SERV-1286). SD-Turbo behind the same endpoint as
baseline_app.py, with every cold-start lever pulled:

  1. /data persistent storage - weights are downloaded and packed ONCE per
     account; every later cold start reads them from shared storage.
  2. FlashPack weight loading - tensors stream from /data straight to GPU
     (2.6 GB in ~2.6 s measured), constant-time and network-free.
  3. synchronized_inductor_cache - torch.compile kernels are compiled once,
     zipped to /data, and every later worker unpacks them instead of
     recompiling.

baseline_app.py is the same model and endpoint with none of the levers;
bench.py deploys both and prints the measured before/after table.
"""
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import fal
from fal.toolkit import Image
from pydantic import BaseModel, Field

MODEL_ID = "stabilityai/sd-turbo"
# Everything this app persists lives under one versioned /data prefix. Bump
# the version when the model, dtype, or torch pin changes - stale packs and
# kernel caches from an old pin must never be loaded by a new one.
PACK_DIR = Path("/data/examples/cut-cold-starts/sd-turbo-fp16-v1")
INDUCTOR_CACHE_KEY = "cut-cold-starts-sd-turbo/v1"
RESOLUTION = 512  # SD-Turbo's native resolution; fixed so warmup covers all shapes


class Input(BaseModel):
    prompt: str = Field(
        description="What to generate. SD-Turbo trades fidelity for speed.",
        examples=["a calico cat riding a bicycle, film photo"],
    )
    num_inference_steps: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Denoising steps. SD-Turbo is distilled for 1; up to 4 sharpens details.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Random when omitted. The response echoes the seed used - the same "
            "seed with the same inputs reproduces the same image."
        ),
    )
    sync_mode: bool = Field(
        default=False,
        description=(
            "Return the image as a data URI instead of a hosted CDN URL - "
            "skips the upload round-trip for latency-sensitive callers."
        ),
    )


class Output(BaseModel):
    image: Image = Field(
        description="The generated image: a CDN URL, or a data URI in sync_mode."
    )
    seed: int = Field(description="The seed that produced this image.")
    generation_ms: int = Field(
        description="Server-side model time for this request (excludes upload)."
    )


class StageTiming(BaseModel):
    name: str = Field(description="Setup stage name.")
    seconds: float = Field(description="Wall-clock duration of the stage.")


class SetupReportInput(BaseModel):
    """No parameters - POST an empty object."""


class SetupReport(BaseModel):
    variant: str = Field(description="'baseline' or 'optimized'.")
    gpu: str = Field(description="GPU this worker booted on.")
    built_caches_this_boot: bool = Field(
        description=(
            "True only on the account's first boot, which builds the /data "
            "FlashPack pack that every later cold start reuses."
        )
    )
    stages: list[StageTiming] = Field(description="Per-stage setup timings, in order.")
    total_seconds: float = Field(description="Total setup() wall-clock time.")


class ColdStartOptimized(fal.App, keep_alive=60):
    machine_type = "GPU-H100"  # pin ONE type: inductor caches are stored per GPU type
    app_auth = "private"
    startup_timeout = 1200  # covers the account's first-ever boot (download + pack + compile)
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    # Identical to baseline_app.py so both apps share one cached environment -
    # the benchmark then isolates the three levers, not pip install noise.
    requirements = [
        "fal==1.78.2",
        "flashpack==0.4.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
    ]

    def setup(self) -> None:
        """Pack once to /data, load via FlashPack, compile against the shared cache."""
        stages: list[dict] = []
        setup_start = time.perf_counter()

        def stage(name: str, start: float) -> None:
            stages.append({"name": name, "seconds": round(time.perf_counter() - start, 2)})

        t0 = time.perf_counter()
        import torch
        from accelerate import init_empty_weights
        from diffusers import (
            AutoencoderKL,
            EulerDiscreteScheduler,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from fal.toolkit import synchronized_inductor_cache
        from flashpack import assign_from_file, pack_to_file
        from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        stage("imports", t0)

        # FlashPack's diffusers integration works through small mixin
        # subclasses: from_pretrained_flashpack() reads <dir>/config.json +
        # <dir>/model.flashpack, builds the module skeleton on the meta device,
        # and streams tensors from /data directly onto the GPU.
        #
        # The text encoder deliberately does NOT get the transformers mixin.
        # transformers 5.x inspects the source module of the model's class at
        # __init__ time (sys.modules[cls.__module__]), and a subclass defined
        # inside a fal app file - which the platform loads by path, without an
        # importable module name - crashes that lookup with KeyError. The
        # direct function API below (skeleton + assign_from_file) does the
        # same work without a subclass.
        class PackedUNet(UNet2DConditionModel, FlashPackDiffusersModelMixin):
            pass

        class PackedVAE(AutoencoderKL, FlashPackDiffusersModelMixin):
            pass

        # LEVER 1 - first boot per ACCOUNT (not per worker): download once into
        # the /data HF cache, repack to FlashPack next to it. The pack is
        # written to LOCAL disk first and then copied to /data as one bulk
        # transfer: pack_to_file's chunked write pattern is pathologically
        # slow directly against /data's network filesystem (measured: a run
        # never finished in 40 min; a plain copy of the same bytes takes
        # seconds to ~2 min depending on the host). The copy lands in a temp
        # dir and is atomically renamed so concurrently booting workers can't
        # observe a half-written pack; the loser of the race just cleans up.
        self.built_caches_this_boot = False
        if not PACK_DIR.is_dir():
            t0 = time.perf_counter()
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, variant="fp16"
            )
            local_dir = Path(tempfile.mkdtemp(prefix="flashpack-build-")) / "pack"
            for name in ("unet", "vae"):
                component_dir = local_dir / name
                component_dir.mkdir(parents=True)
                getattr(pipe, name).save_config(component_dir)
                pack_to_file(getattr(pipe, name), str(component_dir / "model.flashpack"), None)
            text_encoder_dir = local_dir / "text_encoder"
            text_encoder_dir.mkdir(parents=True)
            pipe.text_encoder.config.save_pretrained(str(text_encoder_dir))
            pack_to_file(pipe.text_encoder, str(text_encoder_dir / "model.flashpack"), None)
            pipe.tokenizer.save_pretrained(str(local_dir / "tokenizer"))
            pipe.scheduler.save_pretrained(str(local_dir / "scheduler"))
            del pipe
            stage("pack weights to FlashPack (first boot only)", t0)

            t0 = time.perf_counter()
            for stale in PACK_DIR.parent.glob(".build-*"):  # crashed builds, if any
                if time.time() - stale.stat().st_mtime > 3600:
                    shutil.rmtree(stale, ignore_errors=True)
            publish_dir = PACK_DIR.parent / f".build-{uuid.uuid4().hex}"
            shutil.copytree(local_dir, publish_dir)
            try:
                publish_dir.rename(PACK_DIR)
            except OSError:
                shutil.rmtree(publish_dir)  # another worker finished first
            shutil.rmtree(local_dir.parent, ignore_errors=True)
            stage("publish pack to /data (first boot only)", t0)
            self.built_caches_this_boot = True

        # LEVER 2 - every boot: weights stream from /data to GPU via FlashPack.
        # No Hub round-trip, no safetensors CPU deserialization. The .to() call
        # moves the buffers that skeleton-init left on the CPU.
        t0 = time.perf_counter()
        unet = PackedUNet.from_pretrained_flashpack(str(PACK_DIR / "unet"), device="cuda")
        vae = PackedVAE.from_pretrained_flashpack(str(PACK_DIR / "vae"), device="cuda")
        # Text encoder via the direct function API: skeleton on the meta
        # device (buffers stay real - packs only hold state_dict tensors),
        # then stream the pack onto the GPU.
        with init_empty_weights(include_buffers=False):
            text_encoder = CLIPTextModel(
                CLIPTextConfig.from_pretrained(str(PACK_DIR / "text_encoder"))
            )
        assign_from_file(
            text_encoder, str(PACK_DIR / "text_encoder" / "model.flashpack"), device="cuda"
        )
        text_encoder.eval()
        for module in (unet, vae, text_encoder):
            module.to("cuda", torch.float16)
        leftover_meta = [
            name
            for module in (unet, vae, text_encoder)
            for name, tensor in list(module.named_parameters()) + list(module.named_buffers())
            if tensor.is_meta
        ]
        if leftover_meta:  # a pack/model mismatch must fail the boot, not the request
            raise RuntimeError(f"tensors not materialized from pack: {leftover_meta}")
        self.pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=CLIPTokenizer.from_pretrained(str(PACK_DIR / "tokenizer")),
            unet=unet,
            scheduler=EulerDiscreteScheduler.from_pretrained(str(PACK_DIR / "scheduler")),
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
        )
        self.pipe.set_progress_bar_config(disable=True)
        stage("load weights (FlashPack, /data -> GPU)", t0)

        # LEVER 3 - compile against the shared inductor cache. First boot
        # compiles for real and syncs the kernels to /data on exit; every
        # later boot unpacks them and warmup is just cache-hit execution.
        # Warmup MUST run inside the context or nothing gets captured.
        t0 = time.perf_counter()
        with synchronized_inductor_cache(INDUCTOR_CACHE_KEY):
            self.pipe.unet = torch.compile(self.pipe.unet, dynamic=True)
            for steps in (1, 4):
                self.pipe(
                    prompt="warmup",
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                )
        stage("compile + warmup (shared inductor cache)", t0)

        self.setup_stages = stages
        self.setup_total = round(time.perf_counter() - setup_start, 2)
        self.gpu_name = torch.cuda.get_device_name(0)

    @fal.endpoint("/")
    def generate(self, request: Input) -> Output:
        import torch

        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        t0 = time.perf_counter()
        result = self.pipe(
            prompt=request.prompt,
            height=RESOLUTION,
            width=RESOLUTION,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo is distilled without CFG; keep at 0.0
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        generation_ms = int((time.perf_counter() - t0) * 1000)
        repository = "in_memory" if request.sync_mode else "fal_v3"
        image = Image.from_pil(result.images[0], repository=repository)
        return Output(image=image, seed=seed, generation_ms=generation_ms)

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
