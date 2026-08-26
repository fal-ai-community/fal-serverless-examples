"""Serve thousands of per-user LoRAs with routing hints on FLUX.2 [klein] 4B."""

import gc
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import fal
from fal.exceptions import FieldException
from fal.toolkit import Image, ImageSizeInput, get_image_size
from fal.toolkit.kv import KVStore
from fastapi import Response
from pydantic import BaseModel, Field

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
ADAPTER_DIR = Path("/data/examples/per-user-loras/adapters")
KV_DB_NAME = "per-user-loras"
HINT_PREFIX = "lora:"
MAX_RESIDENT_ADAPTERS = 8  # LRU bound = provide_hints() bound (16 KB header budget)
MAX_LORA_BYTES = 300 * 1024 * 1024
SYNTHETIC_RANK = 8
USER_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"
USER_ID_RE = re.compile(USER_ID_PATTERN)
AdapterSource = Literal["gpu-cache", "disk", "downloaded"]
_HF_REPO_RE = re.compile(r"^[\w][\w.-]*/[\w][\w.-]*$")
_OOM_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

# Entries stay licensed (Apache-2.0/MIT) and span all 4 diffusers-0.39 key formats.
_FLEET = "https://huggingface.co/rehan-fal/klein-style-fleet/resolve/main"
FEATURED_USERS: dict[str, dict] = {
    "ukiyo-e": {
        "url": f"{_FLEET}/ukiyo-e/pytorch_lora_weights.safetensors",
        "scale": 1.0,
        "trigger": "ukiyo-e woodblock print",
    },
    "stained-glass": {
        "url": f"{_FLEET}/stained-glass/pytorch_lora_weights.safetensors",
        "scale": 1.0,
        "trigger": "stained glass window",
    },
    "claymation": {
        "url": f"{_FLEET}/claymation/pytorch_lora_weights.safetensors",
        "scale": 1.0,
        "trigger": "claymation stop-motion",
    },
    "neon-noir": {
        "url": f"{_FLEET}/neon-noir/pytorch_lora_weights.safetensors",
        "scale": 1.0,
        "trigger": "neon noir",
    },
    "blueprint": {
        "url": f"{_FLEET}/blueprint/pytorch_lora_weights.safetensors",
        "scale": 1.0,
        "trigger": "architectural blueprint",
    },
    "wild-africa": {
        "url": "https://huggingface.co/zfrsgtcu/flux-wild-africa/resolve/main/African_Documentary_V2.safetensors",
        "scale": 1.0,
        "trigger": "afrodoc",
    },
    "mikkoani": {
        "url": "https://huggingface.co/mikkoph/mikkoani-klein4b/resolve/main/klein4b-mikkoani.safetensors",
        "scale": 1.0,
        "trigger": "@mikkoani illustration",
    },
    "spritesheet": {
        "url": "https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora/resolve/main/flux-spritesheet-lora.safetensors",
        "scale": 1.0,
        "trigger": "2x2 sprite sheet",
    },
}


class Input(BaseModel):
    user_id: str = Field(
        pattern=USER_ID_PATTERN,
        description=(
            "Which user's LoRA to apply. The pattern keeps hints short, "
            "latin-1 and comma-free - hint strings ride an HTTP header."
        ),
        examples=["ukiyo-e"],
        # ui.hint marks the routing-hint field; the playground reads it from OpenAPI.
        json_schema_extra={"ui": {"hint": True}},
    )
    prompt: str = Field(
        description=(
            "What to generate. Include the user's trigger word if their LoRA has one."
        ),
        examples=["ukiyo-e woodblock print of a red fox in a forest clearing"],
    )
    image_size: ImageSizeInput = Field(
        default="square_hd",
        description="A preset name or a custom {width, height}.",
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Denoising steps. klein is distilled for 4; more buys little.",
    )
    seed: int | None = Field(
        default=None,
        description="Random when omitted. The response echoes the seed used.",
    )
    sync_mode: bool = Field(
        default=False,
        description="Return the image as a data URI instead of a hosted CDN URL.",
    )


class ServingInfo(BaseModel):
    worker_id: str = Field(
        description="Stable id of the runner that served this request."
    )
    adapter_source: AdapterSource = Field(
        description=(
            "'gpu-cache' = adapter was already loaded (hint hit or recent use); "
            "'disk' = loaded from /data; 'downloaded' = fetched/generated first."
        )
    )
    hint: str = Field(
        description=(
            "The routing hint for this user - echo it via the hint= client option."
        )
    )
    resident_adapters: int = Field(
        description="Adapters currently in this runner's GPU LRU."
    )


class Output(BaseModel):
    image: Image = Field(description="The generated image.")
    seed: int = Field(description="The seed that produced this image.")
    timings: dict[str, float] = Field(
        description="Seconds per phase: 'lora' (adapter resolve+load) and 'inference'."
    )
    serving: ServingInfo = Field(
        description=(
            "Where and how this request was served - the routing story, per request."
        )
    )


class RegisterInput(BaseModel):
    user_id: str = Field(
        pattern=USER_ID_PATTERN,
        description="The user to register.",
        examples=["demo-user-1"],
    )
    lora_url: str | None = Field(
        default=None,
        description=(
            "HTTPS URL or Hugging Face repo id (owner/repo) of a FLUX.2-klein-4B "
            f"LoRA (.safetensors, max {MAX_LORA_BYTES // 2**20} MB). Omit to generate "
            "a small synthetic adapter - useful for load-testing the routing "
            "behavior at scale."
        ),
        examples=["fal/flux-2-klein-4b-spritesheet-lora"],
    )
    scale: float = Field(
        default=1.0,
        ge=0.0,
        le=4.0,
        description="Scale factor for LoRA application (0.0 to 4.0).",
    )


class RegisterOutput(BaseModel):
    user_id: str = Field(description="The registered user.")
    kind: Literal["registered", "synthetic"] = Field(
        description=(
            "Whether real weights were downloaded or a synthetic adapter was generated."
        )
    )
    adapter_bytes: int = Field(description="Size of the stored adapter file.")
    hint: str = Field(description="Routing hint clients should send for this user.")


def hint_for(user_id: str) -> str:
    """Routing hint - validated user_id keeps it latin-1, comma-free, <=45 bytes."""
    return f"{HINT_PREFIX}{user_id}"


def validate_lora_source(source: str) -> Literal["url", "hf_repo"]:
    """Classify a lora_url as a direct HTTPS URL or an HF repo id, or 422."""
    if _HF_REPO_RE.match(source):
        return "hf_repo"
    parsed = urlparse(source)
    if parsed.scheme == "https" and parsed.hostname:
        return "url"
    if parsed.scheme == "http":
        raise FieldException(
            field="lora_url",
            message="HTTP links are not supported for model weights. Use HTTPS.",
            type="lora_download_failed",
        )
    raise FieldException(
        field="lora_url",
        message=(
            f"Not a valid HTTPS URL or Hugging Face repo id (owner/repo): {source!r}"
        ),
        type="lora_download_failed",
    )


def looks_like_safetensors(head: bytes, file_size: int) -> bool:
    """Cheap validity gate: 8-byte little-endian header length, then JSON."""
    if len(head) < 9:
        return False
    header_len = int.from_bytes(head[:8], "little")
    return head[8:9] == b"{" and 0 < header_len <= file_size - 8


def is_oom_error(exc: BaseException | None) -> bool:
    """Walk the cause chain for GPU/host OOM - OOM stays retryable, never a 422."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "OutOfMemoryError":
            return True
        message = str(exc).lower()
        if any(marker in message for marker in _OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def lora_incompatible_message(exc: BaseException) -> str | None:
    """Map diffusers size/match errors to a caller-friendly message; None otherwise."""
    msg = str(exc).lower()
    if "size" in msg and "match" in msg:
        return (
            "LoRA weights are incompatible with the current model. "
            "The LoRA was likely trained for a different model architecture or size."
        )
    return None


def synthetic_seed(user_id: str) -> int:
    return int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:4], "big") % (
        2**31
    )


class PerUserLoras(fal.App, keep_alive=300):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    min_concurrency = 0
    max_concurrency = 3
    startup_timeout = (
        1200  # first boot downloads base model + featured adapters to /data
    )
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "hf-transfer==0.1.9",
        "huggingface_hub==1.27.0",
        "diffusers==0.39.0",
        "transformers==5.3.0",
        "torch==2.12.1",
        "accelerate==1.14.0",
        "peft==0.20.0",
        "safetensors==0.8.0",
        "httpx==0.28.1",
    ]

    def setup(self) -> None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = (
            "1"  # must be set before any HF download
        )

        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        # Lock first: provide_hints() assumes the lock exists once _resident does.
        self._cache_lock = threading.Lock()
        self._resident: OrderedDict[str, float] = OrderedDict()  # user_id -> scale
        self._worker_id = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        self.kv = KVStore(KV_DB_NAME)
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

        # Seeding must stay idempotent: multiple runners boot concurrently.
        for uid, meta in FEATURED_USERS.items():
            path = self._adapter_path(uid)
            if not path.is_file():
                print(f"[setup] downloading featured adapter {uid}")
                self._download_to(meta["url"], path)
            self._kv_put_user(
                uid,
                {
                    "kind": "featured",
                    "url": meta["url"],
                    "scale": meta["scale"],
                    "path": str(path),
                    "trigger": meta["trigger"],
                },
            )

        # Warm up the full LoRA path so no adapter-broken runner enters the pool.
        scale, _ = self._activate_adapter("ukiyo-e")
        self.pipe.set_adapters(["ukiyo-e"], adapter_weights=[scale])
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )
        print(
            f"[setup] worker {self._worker_id} ready; resident={list(self._resident)}"
        )

    @staticmethod
    def _adapter_path(user_id: str) -> Path:
        return ADAPTER_DIR / f"{user_id}.safetensors"

    def _kv_get_user(self, user_id: str) -> dict:
        raw = self.kv.get(f"user:{user_id}")
        if raw is None:
            raise FieldException(
                field="user_id",
                message=f"User '{user_id}' is not registered. POST /register first.",
                type="user_not_registered",
            )
        return json.loads(raw)

    def _kv_put_user(self, user_id: str, record: dict) -> None:
        # KV has no atomics; last-write-wins is the intended re-register semantics.
        self.kv.set(f"user:{user_id}", json.dumps(record))

    def _download_to(self, url: str, dest: Path) -> None:
        """Size-capped stream download; tmp + atomic rename hides partials."""
        import httpx

        tmp = dest.with_suffix(f".tmp-{os.getpid()}-{time.monotonic_ns()}")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                declared = int(r.headers.get("content-length", 0))
                if declared > MAX_LORA_BYTES:
                    raise FieldException(
                        field="lora_url",
                        message=(
                            f"LoRA is {declared} bytes; "
                            f"the limit is {MAX_LORA_BYTES // 2**20} MB."
                        ),
                        type="lora_too_large",
                    )
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > MAX_LORA_BYTES:
                            raise FieldException(
                                field="lora_url",
                                message=(
                                    "LoRA exceeds the "
                                    f"{MAX_LORA_BYTES // 2**20} MB limit."
                                ),
                                type="lora_too_large",
                            )
                        f.write(chunk)
            with open(tmp, "rb") as f:
                head = f.read(9)
            if not looks_like_safetensors(head, tmp.stat().st_size):
                raise FieldException(
                    field="lora_url",
                    message=(
                        "Downloaded file is not a .safetensors file. If this is a "
                        "model page URL, use the direct file download link."
                    ),
                    type="lora_invalid_format",
                )
            os.rename(tmp, dest)
        except httpx.HTTPError as exc:
            raise FieldException(
                field="lora_url",
                message=f"Failed to download LoRA from '{url}': {exc}",
                type="lora_download_failed",
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)

    def _resolve_hf_repo(self, repo_id: str) -> str:
        """owner/repo -> direct resolve URL of its single .safetensors file."""
        from huggingface_hub import HfApi

        try:
            files = [
                f
                for f in HfApi().list_repo_files(repo_id)
                # ComfyUI re-exports fail load_lora_weights.
                if f.endswith(".safetensors") and "comfy" not in f
            ]
        except Exception as exc:
            raise FieldException(
                field="lora_url",
                message=f"Could not list Hugging Face repo '{repo_id}': {exc}",
                type="lora_download_failed",
            ) from exc
        if len(files) != 1:
            raise FieldException(
                field="lora_url",
                message=(
                    f"Repo '{repo_id}' must contain exactly one non-comfy "
                    f".safetensors file (found {len(files)}). "
                    "Pass the direct file URL instead."
                ),
                type="lora_download_failed",
            )
        return f"https://huggingface.co/{repo_id}/resolve/main/{files[0]}"

    def _ensure_adapter_file(self, user_id: str, record: dict) -> tuple[Path, bool]:
        """Return (path, downloaded_now). Self-heals a missing /data file."""
        path = Path(record["path"])
        if path.is_file():
            return path, False
        if record["kind"] == "synthetic":
            self._write_synthetic_adapter(user_id, path)
        elif record.get("url"):
            self._download_to(record["url"], path)
        else:
            raise FieldException(
                field="user_id",
                message=(
                    f"Adapter file for '{user_id}' is missing and has no source URL. "
                    "Re-register the user."
                ),
                type="user_not_registered",
            )
        return path, True

    def _write_synthetic_adapter(self, user_id: str, dest: Path) -> None:
        """Seeded structurally-real LoRA; near-zero lora_B = subtle shift, no junk."""
        import torch
        from safetensors.torch import save_file

        gen = torch.Generator(device="cpu").manual_seed(synthetic_seed(user_id))
        state: dict[str, torch.Tensor] = {}
        suffixes = (".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0")
        for name, module in self.pipe.transformer.named_modules():
            if not name.endswith(suffixes) or not hasattr(module, "in_features"):
                continue
            r, n_in, n_out = SYNTHETIC_RANK, module.in_features, module.out_features
            state[f"transformer.{name}.lora_A.weight"] = (
                torch.randn((r, n_in), generator=gen, dtype=torch.float32) * 0.02
            )
            state[f"transformer.{name}.lora_B.weight"] = (
                torch.randn((n_out, r), generator=gen, dtype=torch.float32) * 1e-3
            )
        if not state:
            raise RuntimeError(
                "no attention projections found to target - diffusers naming changed?"
            )
        tmp = dest.with_suffix(f".tmp-{os.getpid()}-{time.monotonic_ns()}")
        try:
            save_file(state, str(tmp))
            os.rename(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)  # nothing on /data is auto-pruned

    def _activate_adapter(self, user_id: str) -> tuple[float, AdapterSource]:
        """GPU-LRU hit, else KV -> /data -> load_lora_weights -> insert+evict."""
        with self._cache_lock:
            if user_id in self._resident:
                self._resident.move_to_end(user_id)
                return self._resident[user_id], "gpu-cache"

        record = self._kv_get_user(user_id)
        path, downloaded = self._ensure_adapter_file(user_id, record)

        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
        with self._cache_lock:
            if user_id in self._resident:  # raced another request
                self._resident.move_to_end(user_id)
                return self._resident[user_id], "gpu-cache"
            try:
                self.pipe.load_lora_weights(state_dict, adapter_name=user_id)
            except Exception as exc:
                if is_oom_error(exc):
                    raise  # retryable 5xx, never a 422
                msg = lora_incompatible_message(exc) or str(exc)
                raise FieldException(
                    field="user_id",
                    message=f"Failed to load LoRA for '{user_id}'. {msg}",
                    type="lora_incompatible",
                ) from exc
            self._resident[user_id] = float(record.get("scale", 1.0))
            if len(self._resident) > MAX_RESIDENT_ADAPTERS:
                evicted, _ = self._resident.popitem(last=False)
                print(f"[lru] evicting adapter {evicted}")
                self.pipe.delete_adapters(evicted)
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            print(
                f"[lru] loaded adapter {user_id} "
                f"(source={'downloaded' if downloaded else 'disk'})"
            )
            return self._resident[user_id], ("downloaded" if downloaded else "disk")

    def provide_hints(self) -> list[str]:
        # May race setup() on any response: return [] until cache exists; never crash.
        resident = getattr(self, "_resident", None)
        if resident is None:
            return []
        with self._cache_lock:
            return [hint_for(uid) for uid in resident]

    @fal.endpoint("/")
    def generate(self, request: Input, response: Response) -> Output:
        import torch

        t0 = time.perf_counter()
        scale, source = self._activate_adapter(request.user_id)
        self.pipe.set_adapters([request.user_id], adapter_weights=[scale])
        lora_s = time.perf_counter() - t0

        size = get_image_size(request.image_size)
        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        t1 = time.perf_counter()
        try:
            result = self.pipe(
                prompt=request.prompt,
                height=size.height,
                width=size.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=1.0,  # klein is guidance-distilled
                generator=torch.Generator("cuda").manual_seed(seed),
            )
        except (RuntimeError, ValueError) as exc:
            if is_oom_error(exc):
                raise
            msg = lora_incompatible_message(exc)
            if msg is None:
                raise
            raise FieldException(
                field="user_id",
                message=msg,
                type="lora_incompatible",
            ) from exc
        inference_s = time.perf_counter() - t1

        hint = hint_for(request.user_id)
        # Clients echo this header as hint= on the next request, never computing hints.
        response.headers["x-fal-runner-hint"] = hint

        repository: Literal["in_memory", "fal_v3"] = (
            "in_memory" if request.sync_mode else "fal_v3"
        )
        return Output(
            image=Image.from_pil(result.images[0], repository=repository),
            seed=seed,
            timings={"lora": round(lora_s, 4), "inference": round(inference_s, 4)},
            serving=ServingInfo(
                worker_id=self._worker_id,
                adapter_source=source,
                hint=hint,
                resident_adapters=len(self._resident),
            ),
        )

    @fal.endpoint("/register")
    def register(self, request: RegisterInput, response: Response) -> RegisterOutput:
        if request.user_id in FEATURED_USERS:
            # setup() re-seeds featured KV records each boot; overrides silently revert.
            raise FieldException(
                field="user_id",
                message=(
                    f"'{request.user_id}' is a featured demo user and cannot "
                    "be re-registered. Pick a different user_id."
                ),
                type="user_reserved",
            )
        path = self._adapter_path(request.user_id)
        kind: Literal["registered", "synthetic"]
        if request.lora_url is None:
            kind = "synthetic"
            prior = self.kv.get(f"user:{request.user_id}")
            prior_synthetic = (
                prior is not None and json.loads(prior).get("kind") == "synthetic"
            )
            # Key off prior kind, not the file: a real registration left real weights.
            if not (prior_synthetic and path.is_file()):
                self._write_synthetic_adapter(request.user_id, path)
            record = {
                "kind": kind,
                "url": None,
                "scale": request.scale,
                "path": str(path),
            }
        else:
            kind = "registered"
            source_kind = validate_lora_source(request.lora_url)
            url = (
                self._resolve_hf_repo(request.lora_url)
                if source_kind == "hf_repo"
                else request.lora_url
            )
            self._download_to(url, path)  # re-register replaces the file
            record = {
                "kind": kind,
                "url": url,
                "scale": request.scale,
                "path": str(path),
            }
        self._kv_put_user(request.user_id, record)
        # Registration invalidates any older resident copy of this user.
        with self._cache_lock:
            if request.user_id in self._resident:
                self._resident.pop(request.user_id)
                self.pipe.delete_adapters(request.user_id)
        hint = hint_for(request.user_id)
        response.headers["x-fal-runner-hint"] = hint
        return RegisterOutput(
            user_id=request.user_id,
            kind=kind,
            adapter_bytes=path.stat().st_size,
            hint=hint,
        )
