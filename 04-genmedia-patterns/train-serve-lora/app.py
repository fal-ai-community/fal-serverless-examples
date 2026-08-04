"""Serve a freshly-trained klein LoRA from /data - the serving half of 04.4.

The trainer (app_trainer.py) writes /data/loras/<lora_name>/; this app loads
adapters from there by name onto the DISTILLED klein-4B (4-step inference).
Train where gradients behave, serve where inference is fast.

One resident adapter, swapped on demand - deliberately the 20-line version
of 04.3's LRU fleet (see README "Next steps").
"""
import random
import time
from pathlib import Path

import fal
from fal.exceptions import FieldException
from fal.toolkit import Image, ImageSizeInput, download_file, get_image_size
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

import shared


class GenerateInput(BaseModel):
    prompt: str = Field(
        description=(
            "What to generate. Repeat your training trigger phrase to invoke "
            "the style (the demo adapter's is 'pprdrma style,')."
        ),
        examples=["pprdrma style, a tram crossing a city bridge at sunset"],
    )
    lora_name: str = Field(
        default=shared.DEMO_LORA,
        pattern=shared.SLUG_PATTERN,
        description="Adapter to serve, by trainer slug (/data/loras/<lora_name>/).",
    )
    lora_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="Adapter strength. 0 disables the style; ~0.7-1.2 is typical.",
    )
    image_size: ImageSizeInput = Field(
        default="square_hd",
        description="A preset name or a custom {width, height} (multiples of 16).",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, le=8,
        description="Denoising steps. klein is distilled for 4; more buys little.",
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted; the response echoes the seed used.",
    )
    sync_mode: bool = Field(
        default=False,
        description="Return a data URI instead of a hosted CDN URL.",
    )


class ServingInfo(BaseModel):
    lora_name: str = Field(description="The adapter that produced this image.")
    adapter_source: str = Field(
        description=(
            "'cached' = already on the GPU; 'disk' = loaded from /data for this "
            "request; 'seeded' = loaded from the /data copy that setup() "
            "bootstrapped from the published reference adapter."
        )
    )


class GenerateOutput(BaseModel):
    image: Image = Field(description="The generated image.")
    seed: int = Field(description="The seed that produced this image.")
    timings: dict = Field(description="Seconds spent in adapter handling vs inference.")
    serving: ServingInfo = Field(description="How this request was served.")


class KleinLoraServer(fal.App, keep_alive=300):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    startup_timeout = 1200  # first boot: ~16 GB of weights + the demo adapter
    app_files = ["shared.py"]  # Import Code: same module the trainer ships
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.19.1",
        "safetensors==0.8.0",
        "pillow==11.3.0",
        "numpy==2.4.0",
    ]

    def setup(self) -> None:
        """Load -> seed -> adopt -> warm up, all through the adapter path,
        so a healthy worker is proven able to serve LoRA traffic."""
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            shared.SERVE_MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        # Idempotent /data bootstrap (04.3's featured-user pattern): if the
        # demo adapter isn't on this account's /data yet, fetch the published
        # reference copy - CI stays green on a fresh account, and the example
        # works before you've run your first training.
        self._resident: str | None = None
        self._adapter_id: str | None = None
        self._counter = 0
        self._seeded: set[str] = set()
        demo_weights = shared.lora_weights_path(shared.DEMO_LORA)
        if not demo_weights.exists():
            tmp = shared.tmp_sibling(demo_weights)
            try:
                fetched = download_file(
                    shared.REFERENCE_ADAPTER_URL,
                    tmp.parent,
                    filesize_limit=shared.MAX_ADAPTER_MB,
                )
            except DownloadError as exc:
                raise RuntimeError(
                    f"could not seed the demo adapter from {shared.REFERENCE_ADAPTER_URL}: {exc}"
                ) from exc
            shared.validate_safetensors_headers(Path(fetched))
            shared.atomic_replace(Path(fetched), demo_weights)
            tmp.unlink(missing_ok=True)
            self._seeded.add(shared.DEMO_LORA)

        self._load_adapter(shared.DEMO_LORA)
        # Warmup THROUGH the adapter: proves the whole serving path at boot.
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )

    def _load_adapter(self, name: str) -> str:
        """Swap the resident adapter; returns the adapter_source tag."""
        if self._resident == name:
            return "cached"
        path = shared.lora_weights_path(name)
        if not path.exists():
            raise FieldException(
                field="lora_name",
                message=(
                    f"no adapter at /data/loras/{name}/ - train one first "
                    "(app_trainer.py /train) or check the slug"
                ),
                type="lora_not_found",
            )
        try:
            shared.validate_safetensors_headers(path)
        except shared.ArtifactError as exc:
            raise FieldException(
                field="lora_name", message=str(exc), type="lora_invalid_format"
            ) from None

        if self._adapter_id is not None:
            self.pipe.delete_adapters(self._adapter_id)  # frees the old slot's VRAM
            self._resident = None
            self._adapter_id = None
        adapter_id = f"{name}-{self._counter}"  # loaded names can't be reused
        self._counter += 1
        try:
            self.pipe.load_lora_weights(
                str(path), adapter_name=adapter_id, use_safetensors=True
            )
        except RuntimeError as exc:
            if "size mismatch for" in str(exc):
                # The diagnostic that answers most bad-LoRA support tickets.
                raise FieldException(
                    field="lora_name",
                    message=(
                        "adapter tensors don't fit this model - the file was "
                        "likely trained for a different architecture or size "
                        "(e.g. a 9B or FLUX.2-dev LoRA on klein-4B)"
                    ),
                    type="lora_incompatible",
                ) from None
            raise  # CUDA/OOM faults stay 5xx: retryable, not the caller's fault
        if adapter_id not in getattr(self.pipe.transformer, "peft_config", {}):
            # diffusers "loads" foreign-prefix files as a silent no-op; make it loud.
            raise FieldException(
                field="lora_name",
                message="file contains no FLUX.2 transformer LoRA keys",
                type="lora_invalid_format",
            )
        self._log_key_audit(path, adapter_id)
        self._resident = name
        self._adapter_id = adapter_id
        return "seeded" if name in self._seeded else "disk"

    def _log_key_audit(self, path: Path, adapter_id: str) -> None:
        """'N/M modules loaded' - the other half of LoRA debuggability."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        sd = type(self.pipe).lora_state_dict(str(path))
        file_modules = {
            k.removeprefix("transformer.").removesuffix(".lora_A.weight")
            for k in sd
            if k.endswith(".lora_A.weight")
        }
        injected = {
            n
            for n, m in self.pipe.transformer.named_modules()
            if isinstance(m, BaseTunerLayer) and adapter_id in m.lora_A
        }
        loaded = len(file_modules & injected)
        print(f"[lora] {loaded}/{len(file_modules)} adapter modules loaded from {path}")
        if loaded < len(file_modules):
            print(f"[lora] unused modules (first 5): {sorted(file_modules - injected)[:5]}")

    @fal.endpoint("/")
    def generate(self, request: GenerateInput) -> GenerateOutput:
        import torch

        t0 = time.monotonic()
        source = self._load_adapter(request.lora_name)
        self.pipe.set_adapters(self._adapter_id, adapter_weights=request.lora_scale)
        t_lora = time.monotonic() - t0

        size = get_image_size(request.image_size)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        t1 = time.monotonic()
        result = self.pipe(
            prompt=request.prompt,
            height=size.height,
            width=size.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0,  # klein is guidance-distilled; the arg is ignored
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        t_inference = time.monotonic() - t1

        repository = "in_memory" if request.sync_mode else "fal_v3"
        image = Image.from_pil(result.images[0], repository=repository)
        return GenerateOutput(
            image=image,
            seed=seed,
            timings={"lora": round(t_lora, 3), "inference": round(t_inference, 3)},
            serving=ServingInfo(lora_name=request.lora_name, adapter_source=source),
        )
