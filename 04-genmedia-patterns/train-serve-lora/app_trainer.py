"""Train a FLUX.2-klein LoRA on fal serverless - the trainer half of train-serve-lora."""
import copy
import json
import os
import random
import tempfile
import time
from collections import deque
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

# Fixed on purpose: hyperparameters change here, never per-request.
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


class PreflightOutput(BaseModel):
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
        description="AdamW learning rate (constant after 100 warmup steps).",
    )


class TrainOutput(BaseModel):
    diffusers_lora_file: File = Field(
        description=(
            "The trained adapter (diffusers-native keys) on the CDN - portable: "
            "drop-in for per-user-loras' /register or fal's hosted klein LoRA endpoints."
        )
    )
    data_path: str = Field(description="Where the adapter landed on /data.")
    steps: int = Field(description="Steps actually trained.")
    final_loss: float = Field(description=f"Mean loss over the last {LOG_EVERY} steps.")
    config: dict = Field(description="Echo of the training inputs (config.json sidecar).")


class KleinLoraTrainer(fal.App, keep_alive=0):
    machine_type = "GPU-H100"  # training wants speed; L40 works (see README)
    app_auth = "private"
    startup_timeout = 1200     # first-ever boot downloads ~16 GB of base weights
    request_timeout = 7200     # account-level max; a 10k-step run needs it
    skip_retry_conditions = ["timeout"]  # a timed-out training must never be re-run (re-billed) by the queue
    max_concurrency = 1                  # concurrent trainings on one runner fight for VRAM
    # Must be local_python_modules, not app_files: ephemeral `fal run` can't
    # resolve an app_files import at deserialization time (verified live).
    local_python_modules = ["shared"]
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.20.0",
        "safetensors==0.8.0",
        "pillow==11.3.0",
        "numpy==2.4.6",
    ]

    def setup(self) -> None:
        """Load the BASE model (gradients need the undistilled weights); hard-fail so an unhealthy trainer never looks ready."""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            shared.TRAIN_MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

    @fal.endpoint("/")
    def preflight(self, request: PreflightInput) -> PreflightOutput:
        """Validate a dataset without spending a training run on it - byte-for-byte the head of /train."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            zip_path = _fetch_dataset(request.image_data_url, work / "dl")
            ds = _prepare_or_422(zip_path, work, request)
        return PreflightOutput(
            ready=True,
            num_images=ds.num_images,
            num_sidecar_captions=ds.num_sidecar_captions,
            example_caption=ds.items[0][1],
            warnings=ds.warnings,
        )

    @fal.endpoint("/train")
    def train(self, request: TrainInput) -> TrainOutput:
        """Fine-tune a LoRA and publish it to /data (atomic) + the CDN."""
        import numpy as np
        import torch
        from diffusers import Flux2KleinPipeline
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import (
            cast_training_params,
            compute_density_for_timestep_sampling,
            compute_loss_weighting_for_sd3,
            free_memory,
        )
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        from PIL import Image as PILImage

        pipe = self.pipe
        device = "cuda"
        t_start = time.perf_counter()

        # 1. dataset (same head as /)
        with tempfile.TemporaryDirectory(prefix="train-") as td:
            staging = Path(td)
            zip_path = _fetch_dataset(request.image_data_url, staging / "dl")
            ds = _prepare_or_422(zip_path, staging, request)
            print(f"[train] dataset ok: {ds.num_images} images, "
                  f"{ds.num_sidecar_captions} sidecar captions, {len(ds.warnings)} warnings")

            # 2. precompute (once)
            # Text uses the PIPELINE's default encoder layers, not the script's
            # CLI default: serving re-encodes with pipeline defaults and the
            # LoRA must train against the embeddings it will be served with.
            pipe.text_encoder.to(device)
            pipe.vae.to(device)
            embeds, text_ids_list = [], []
            with torch.no_grad():
                for _, caption in ds.items:
                    pe, ti = pipe.encode_prompt(prompt=caption, device=device)
                    embeds.append(pe)
                    text_ids_list.append(ti)

            # klein's VAE has no scaling/shift factor; latents normalize with its BatchNorm stats.
            bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
            bn_std = torch.sqrt(
                pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
            ).to(device)
            latents = []
            with torch.no_grad():
                for path, _ in ds.items:
                    img = PILImage.open(path)  # prepare_dataset already made it RGB, EXIF-normalized
                    scale = RESOLUTION / min(img.size)
                    img = img.resize(
                        (round(img.width * scale), round(img.height * scale)),
                        PILImage.BILINEAR,
                    )
                    left = (img.width - RESOLUTION) // 2
                    top = (img.height - RESOLUTION) // 2
                    img = img.crop((left, top, left + RESOLUTION, top + RESOLUTION))
                    px = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
                    px = (px.permute(2, 0, 1) * 2 - 1).unsqueeze(0)
                    px = px.to(device, dtype=pipe.vae.dtype)
                    lat = pipe.vae.encode(px).latent_dist.mode()
                    lat = Flux2KleinPipeline._patchify_latents(lat)
                    lat = (lat - bn_mean) / bn_std
                    latents.append(lat)
            # Biggest VRAM lever: only the transformer stays on the GPU for the loop.
            pipe.text_encoder.to("cpu")
            pipe.vae.to("cpu")
            free_memory()
            print(f"[train] precompute done in {time.perf_counter() - t_start:.0f}s "
                  f"({len(latents)} latents cached)")

        # 3. LoRA on the transformer
        transformer = pipe.transformer
        transformer.requires_grad_(False)
        # Wider than the script's attention-only default, which trains adapters
        # too weak to express a style at lora_scale 1.0 (validated empirically).
        # range(24) is the 9B block count; peft ignores names absent in the 4B.
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.to_qkv_mlp_proj",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in", "ff.linear_out",
            "ff_context.linear_in", "ff_context.linear_out",
        ] + [f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)]
        torch.manual_seed(TRAIN_SEED)  # covers LoRA gaussian init, dropout, timestep sampling
        transformer.add_adapter(
            LoraConfig(
                r=RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
        )
        # LoRA master weights must train in fp32: in bf16, AdamW's ~5e-5 steps
        # round away against lora_A's ~0.0625 init (bf16 ULP there is ~2.4e-4).
        cast_training_params([transformer], dtype=torch.float32)
        transformer.enable_gradient_checkpointing()
        transformer.train()
        lora_params = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=request.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )
        # Constant-after-warmup on purpose: cosine decays to ~0 and wastes the
        # final third of a style run (validated empirically).
        lr_sched = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=LR_WARMUP_STEPS,
            num_training_steps=request.steps,
        )
        sched = copy.deepcopy(pipe.scheduler)  # never mutate the pipeline's copy

        def get_sigmas(timesteps, n_dim, dtype):
            sigmas = sched.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = sched.timesteps.to(device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        # klein-4B ships guidance_embeds: false (verified); the gate makes a
        # variant that flips it fail loudly, not train a different conditioning.
        if transformer.config.guidance_embeds:
            guidance = torch.full([1], 1.0, device=device)
            guidance = guidance.expand(1)
        else:
            guidance = None

        # 4. the loop
        gen = torch.Generator(device=device).manual_seed(TRAIN_SEED)
        rng = random.Random(TRAIN_SEED)
        order: list[int] = []
        window: deque[float] = deque(maxlen=LOG_EVERY)
        try:
            for step in range(1, request.steps + 1):
                if not order:  # epoch boundary: reshuffle, no replacement
                    order = list(range(len(latents)))
                    rng.shuffle(order)
                i = order.pop()
                model_input = latents[i]
                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device)
                noise = torch.randn(
                    model_input.shape, device=device, dtype=model_input.dtype, generator=gen
                )

                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=1,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * sched.config.num_train_timesteps).long()
                timesteps = sched.timesteps[indices].to(device)
                sigmas = get_sigmas(timesteps, model_input.ndim, model_input.dtype)

                noisy = (1.0 - sigmas) * model_input + sigmas * noise
                packed = Flux2KleinPipeline._pack_latents(noisy)

                model_pred = transformer(
                    hidden_states=packed,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=embeds[i],
                    txt_ids=text_ids_list[i],
                    img_ids=model_input_ids,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed.size(1)]
                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
                    model_pred, model_input_ids
                )

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas
                )
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                ).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, MAX_GRAD_NORM)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

                window.append(loss.item())
                if step == 1 or step % LOG_EVERY == 0:
                    print(
                        f"[train] step {step}/{request.steps} "
                        f"loss {sum(window) / len(window):.4f} "
                        f"lr {lr_sched.get_last_lr()[0]:.2e} "
                        f"elapsed {time.perf_counter() - t_start:.0f}s"
                    )
            # CUDA OOM propagates as a retryable 5xx on purpose, never a
            # permanent 422 blaming the user's dataset.

            # 5. gate -> publish
            transformer.eval()
            lora_layers = get_peft_model_state_dict(transformer)
            shared.LORA_ROOT.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=shared.LORA_ROOT) as td:
                # temp dir on /data itself => same filesystem => atomic rename
                # No peft metadata: diffusers infers lora_alpha == rank, which
                # round-trips at scale 1.0 ONLY because RANK == LORA_ALPHA.
                Flux2KleinPipeline.save_lora_weights(
                    save_directory=td, transformer_lora_layers=lora_layers
                )
                tmp_weights = Path(td) / shared.LORA_WEIGHT_NAME
                shapes = shared.validate_safetensors_headers(tmp_weights)
                shared.assert_diffusers_lora_keys(shapes)

                config = {
                    "model": shared.TRAIN_MODEL_ID,
                    "steps": request.steps,
                    "learning_rate": request.learning_rate,
                    "trigger_phrase": request.trigger_phrase,
                    "default_caption": request.default_caption,
                    "rank": RANK,
                    "lora_alpha": LORA_ALPHA,
                    "resolution": RESOLUTION,
                }
                # Order matters: CDN copy, then config, then weights LAST -
                # the weights file appearing is the readiness signal.
                lora_file = File.from_path(str(tmp_weights), repository="fal_v3")
                cfg_tmp = shared.tmp_sibling(shared.lora_config_path(request.lora_name))
                cfg_tmp.write_text(json.dumps(config, indent=2))
                shared.atomic_replace(cfg_tmp, shared.lora_config_path(request.lora_name))
                shared.atomic_replace(
                    tmp_weights, shared.lora_weights_path(request.lora_name)
                )
        finally:
            # Always drop the adapter, even with keep_alive=0.
            try:
                transformer.delete_adapters("default")
            except Exception:
                pass

        print(f"[train] published /data/loras/{request.lora_name} "
              f"in {time.perf_counter() - t_start:.0f}s total")
        return TrainOutput(
            diffusers_lora_file=lora_file,
            data_path=str(shared.lora_weights_path(request.lora_name)),
            steps=request.steps,
            final_loss=round(sum(window) / len(window), 4),
            config=config,
        )


def _fetch_dataset(url: str, work_dir: Path) -> Path:
    """Bounded, SSRF-guarded download (both built into download_file)."""
    try:
        return download_file(url, work_dir, filesize_limit=shared.MAX_ZIP_MB)
    except DownloadError as exc:
        detail = str(exc)
        # The SDK names the limit only for declared-Content-Length overflows;
        # mid-stream overflows get a generic message, hence the substring check.
        code = "file_too_large" if ("exceed" in detail or "limit" in detail) else "file_download_error"
        raise FieldException(
            field="image_data_url",
            message=f"could not download (limit {shared.MAX_ZIP_MB} MB): {detail}",
            type=code,
        ) from None


def _prepare_or_422(
    zip_path: Path, work_dir: Path, request: PreflightInput
) -> shared.PreparedDataset:
    try:
        return shared.prepare_dataset(
            zip_path,
            work_dir,
            default_caption=request.default_caption,
            trigger_phrase=request.trigger_phrase,
        )
    except shared.DatasetError as exc:
        raise FieldException(field=exc.field, message=exc.message, type=exc.code) from None
