"""Train a FLUX.2-klein LoRA on fal serverless - the trainer half of 04.4.

The interesting part is the settings block: this is the long-job pattern
the docs imply but never assemble in one place.

  keep_alive=0                       a training job pays for exactly what it
                                     uses; no idle GPU, no adapter-state leak
  request_timeout=7200               training outlives the 3600s default;
                                     7200 is the account-level maximum
  skip_retry_conditions=["timeout"]  a timed-out training must NOT be silently
                                     re-run (and re-billed) by the queue
  max_concurrency=1                  one training per GPU - concurrent jobs on
                                     one runner would fight for VRAM

`POST /` is a dataset preflight: the exact validation head of `/train`,
no GPU time. (Hosted fal trainers validate inline instead - this split
exists so CI can exercise the input path without paying for a train.)
Training math in `/train` is the official diffusers klein DreamBooth
script's core, inlined - see the pinned source link at TRAINING_SOURCE.
"""
import tempfile
from pathlib import Path

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

import shared

TRAINING_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/v0.39.0/"
    "examples/dreambooth/train_dreambooth_lora_flux2_klein.py"
)

# The product-matching hidden defaults (fixed on purpose - the public surface
# stays small; change them here, not per-request):
RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
RESOLUTION = 1024        # BFL style-LoRA guidance; drop to 512 for cheap iteration
LR_WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
TRAIN_SEED = 42
LOG_EVERY = 25


class PreflightInput(BaseModel):
    image_data_url: str = Field(
        description=(
            "URL of a ZIP of 10-40 style images (png/jpg/webp), each optionally "
            "paired with a same-name .txt caption describing its CONTENT "
            "(the style itself belongs in trigger_phrase)."
        ),
        examples=[shared.DATASET_ZIP_URL],
    )
    default_caption: str | None = Field(
        default=None,
        description="Caption used for any image without a sidecar .txt.",
    )
    trigger_phrase: str | None = Field(
        default=None,
        description=(
            "Prepended to every caption at training time (plain text prepend, "
            "no token surgery). Use a non-word like 'pprdrma style,' and repeat "
            "it in serving prompts to invoke the style."
        ),
    )


class PreflightReport(BaseModel):
    ready: bool = Field(description="True - errors surface as typed 422s instead.")
    num_images: int = Field(description="Images that passed decoding and size gates.")
    num_sidecar_captions: int = Field(description="Images with their own .txt caption.")
    example_caption: str = Field(description="The final caption of the first image.")
    warnings: list[str] = Field(description="Non-fatal dataset advice.")


class TrainInput(PreflightInput):
    lora_name: str = Field(
        pattern=shared.SLUG_PATTERN,
        description=(
            "Adapter slug: becomes /data/loras/<lora_name>/ on your account, "
            "which the serving app (app.py) loads by name."
        ),
        examples=["my-style"],
    )
    steps: int = Field(
        default=1000,
        ge=100,
        le=10_000,
        multiple_of=100,
        description="Training steps. Style LoRAs peak around 750-1500.",
    )
    learning_rate: float = Field(
        default=5e-5,
        ge=1e-6,
        le=1e-3,
        description="AdamW learning rate (cosine schedule, 100 warmup steps).",
    )


class TrainOutput(BaseModel):
    lora_file: File = Field(
        description=(
            "The trained adapter (diffusers-native keys) on the CDN - portable: "
            "drop-in for 04.3's /register or fal's hosted klein LoRA endpoints."
        )
    )
    data_path: str = Field(description="Where the adapter landed on /data.")
    steps: int = Field(description="Steps actually trained.")
    final_loss: float = Field(description="Mean loss over the last 25 steps.")
    config: dict = Field(description="Echo of the training inputs (config.json sidecar).")


def _fetch_dataset(url: str, work_dir: Path) -> Path:
    """Bounded, SSRF-guarded download (both built into download_file)."""
    try:
        return download_file(url, work_dir, filesize_limit=shared.MAX_ZIP_MB)
    except DownloadError as exc:
        detail = str(exc)
        code = "file_too_large" if ("exceed" in detail or "limit" in detail) else "file_download_error"
        raise FieldException(
            field="image_data_url",
            message=f"could not download (limit {shared.MAX_ZIP_MB} MB): {detail}",
            type=code,
        ) from None


def _prepare_or_422(zip_path: Path, work_dir: Path, request: PreflightInput):
    try:
        return shared.prepare_dataset(
            zip_path,
            work_dir,
            default_caption=request.default_caption,
            trigger_phrase=request.trigger_phrase,
        )
    except shared.DatasetError as exc:
        raise FieldException(field=exc.field, message=exc.message, type=exc.code) from None


class KleinLoraTrainer(fal.App, keep_alive=0):
    machine_type = "GPU-H100"  # training wants speed; L40 works (see README)
    app_auth = "private"
    startup_timeout = 1200     # first-ever boot downloads ~16 GB of base weights
    request_timeout = 7200     # account-level max; a 10k-step run needs it
    skip_retry_conditions = ["timeout"]
    max_concurrency = 1
    app_files = ["shared.py"]  # Import Code: ship the shared module with the app
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
    ]

    def setup(self) -> None:
        """Load the BASE model (gradients need the undistilled weights).
        Hard-fail: a trainer that can't load its model must not look healthy."""
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            shared.TRAIN_MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

    @fal.endpoint("/")
    def preflight(self, request: PreflightInput) -> PreflightReport:
        """Validate a dataset without spending a training run on it.
        This is byte-for-byte the head of /train."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            zip_path = _fetch_dataset(request.image_data_url, work / "dl")
            ds = _prepare_or_422(zip_path, work, request)
        return PreflightReport(
            ready=True,
            num_images=ds.num_images,
            num_sidecar_captions=ds.num_sidecar_captions,
            example_caption=ds.items[0][1],
            warnings=ds.warnings,
        )
