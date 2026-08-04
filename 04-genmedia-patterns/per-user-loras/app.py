"""Serve thousands of per-user LoRAs with routing hints - FLUX.2 [klein] 4B.

Catalog slot 04.3 (SERV-1306). One shared warm fleet serves every user's
personal LoRA: adapters live on /data, a KV store maps user -> adapter, each
runner keeps a small GPU LRU of loaded adapters, and provide_hints() tells
fal's router which users this runner already has hot. Hints are best-effort
affinity - any runner can serve any user; a miss just pays the adapter load.
"""
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
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
MAX_RESIDENT_ADAPTERS = 8   # LRU bound = provide_hints() bound (16 KB header budget)
MAX_LORA_BYTES = 300 * 1024 * 1024
SYNTHETIC_RANK = 8
USER_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"
USER_ID_RE = re.compile(USER_ID_PATTERN)

# Real, license-verified klein-4B LoRAs (Apache-2.0 / MIT), spanning all four
# key-format families diffusers 0.39 loads: ai-toolkit, kohya, raw-PEFT
# (fal-trainer output), diffusers-native. The demo's "thousands" come from
# synthetic registrations; these eight are the visually-distinct featured users.
FEATURED_USERS: dict[str, dict] = {
    "numzoo": {
        "url": "https://huggingface.co/goumsss/numzoo-flux2-klein-lora/resolve/main/numzoo_klein.safetensors",
        "scale": 1.0, "trigger": "NUMZOO",
    },
    "yarn-art": {
        "url": "https://huggingface.co/Barath/flux2-klein-yarn-art-lora/resolve/main/klein4b_yarn_art_lora_v1.safetensors",
        "scale": 1.0, "trigger": "Y4RN_ART",
    },
    "baroque": {
        "url": "https://huggingface.co/noema-art/dutchbaroqueflux-klein/resolve/main/dutchbaroqueflux_klein.safetensors",
        "scale": 1.0, "trigger": "dutchbaroque",
    },
    "koni-anime": {
        "url": "https://huggingface.co/Sawata97/flux2_4b_koni_animestyle/resolve/main/Flux_klein_4b_anime_Koni.safetensors",
        "scale": 1.0, "trigger": "anime_style",
    },
    "wild-africa": {
        "url": "https://huggingface.co/zfrsgtcu/flux-wild-africa/resolve/main/African_Documentary_V2.safetensors",
        "scale": 1.0, "trigger": "afrodoc",
    },
    "mikkoani": {
        "url": "https://huggingface.co/mikkoph/mikkoani-klein4b/resolve/main/klein4b-mikkoani.safetensors",
        "scale": 1.0, "trigger": "@mikkoani illustration",
    },
    "spritesheet": {
        "url": "https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora/resolve/main/flux-spritesheet-lora.safetensors",
        "scale": 1.0, "trigger": "2x2 sprite sheet",
    },
    "liqglass": {
        "url": "https://huggingface.co/Andy-ML-And-AI/LIQGLASS/resolve/main/pytorch_lora_weights.safetensors",
        "scale": 1.0, "trigger": "LIQGLASS",
    },
}


class Input(BaseModel):
    user_id: str = Field(
        pattern=USER_ID_PATTERN,
        description=(
            "Which user's LoRA to apply. The pattern keeps hints short, "
            "latin-1 and comma-free - hint strings ride an HTTP header."
        ),
        examples=["numzoo"],
        ui={"hint": True},
    )
    prompt: str = Field(
        description="What to generate. Include the user's trigger word if their LoRA has one.",
        examples=["NUMZOO a fox portrait, pastel colors, storybook style"],
    )
    image_size: ImageSizeInput = Field(
        default="square_hd",
        description="A preset name or a custom {width, height}.",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, le=8,
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
    worker_id: str = Field(description="Stable id of the runner that served this request.")
    adapter_source: str = Field(
        description=(
            "'gpu-cache' = adapter was already loaded (hint hit or recent use); "
            "'disk' = loaded from /data; 'downloaded' = fetched/generated first."
        )
    )
    hint: str = Field(description="The routing hint for this user - echo it via the hint= client option.")
    resident_adapters: int = Field(description="Adapters currently in this runner's GPU LRU.")


class Output(BaseModel):
    image: Image = Field(description="The generated image.")
    seed: int = Field(description="The seed that produced this image.")
    timings: dict[str, float] = Field(
        description="Seconds per phase: 'lora' (adapter resolve+load) and 'inference'."
    )
    serving: ServingInfo = Field(
        description="Where and how this request was served - the routing story, per request."
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
            "LoRA (.safetensors, max 300 MB). Omit to generate a small synthetic "
            "adapter - useful for load-testing the routing behavior at scale."
        ),
        examples=["fal/flux-2-klein-4b-spritesheet-lora"],
    )
    scale: float = Field(
        default=1.0, ge=0.0, le=4.0,
        description="Scale factor for LoRA application (0.0 to 4.0).",
    )


class RegisterOutput(BaseModel):
    user_id: str = Field(description="The registered user.")
    kind: str = Field(description="'registered' (real weights) or 'synthetic'.")
    adapter_bytes: int = Field(description="Size of the stored adapter file.")
    hint: str = Field(description="Routing hint clients should send for this user.")


def hint_for(user_id: str) -> str:
    """The routing hint for a user. Short and fixed-shape by construction:
    user_id is pattern-validated, so the hint is latin-1, comma-free and
    at most 45 bytes. For URL-keyed fleets use a digest instead:
    f"lora:{sha256(url.encode()).hexdigest()[:16]}"."""
    return f"{HINT_PREFIX}{user_id}"


_HF_REPO_RE = re.compile(r"^[\w][\w.-]*/[\w][\w.-]*$")
_OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed", "cudnn_status_alloc_failed")


def validate_lora_source(source: str) -> str:
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
        message=f"Not a valid HTTPS URL or Hugging Face repo id (owner/repo): {source!r}",
        type="lora_download_failed",
    )


def looks_like_safetensors(head: bytes, file_size: int) -> bool:
    """Cheap validity gate: 8-byte little-endian header length, then JSON."""
    if len(head) < 9:
        return False
    header_len = int.from_bytes(head[:8], "little")
    return head[8:9] == b"{" and 0 < header_len <= file_size - 8


def is_oom_error(exc: BaseException | None) -> bool:
    """Walk the cause chain looking for GPU/host OOM. OOM must propagate as a
    retryable failure - never be swallowed into a permanent 422."""
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


def synthetic_seed(user_id: str) -> int:
    return int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:4], "big") % (2**31)


class PerUserLoras(fal.App, keep_alive=300):
    machine_type = ["GPU-L40", "GPU-H100"]  # tried in order until capacity is found
    app_auth = "private"
    min_concurrency = 0
    max_concurrency = 3
    startup_timeout = 1200  # first boot downloads base model + featured adapters to /data
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
    ]

    def setup(self) -> None:
        """Load -> seed featured users -> warmup THROUGH the LoRA path."""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # before any HF download

        import torch
        from diffusers import Flux2KleinPipeline

        # LOAD: base weights land in the account-wide /data HF cache, so only
        # the first boot ever downloads from the Hub.
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")

        # Serving state. _resident is the GPU adapter LRU *and* the exact set
        # provide_hints() advertises - the LRU bound is the hint-list bound.
        self._resident: OrderedDict[str, float] = OrderedDict()  # user_id -> scale
        self._cache_lock = threading.Lock()
        self._worker_id = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        self.kv = KVStore(KV_DB_NAME)
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

        # SEED the featured users: idempotent - files persist on /data across
        # runners and boots; KV writes are last-write-wins with fixed values.
        for uid, meta in FEATURED_USERS.items():
            path = self._adapter_path(uid)
            if not path.is_file():
                print(f"[setup] downloading featured adapter {uid}")
                self._download_to(meta["url"], path)
            self._kv_put_user(uid, {
                "kind": "featured", "url": meta["url"], "scale": meta["scale"],
                "path": str(path), "trigger": meta["trigger"],
            })

        # WARMUP through the full serving path: hint-miss -> disk load ->
        # set_adapters -> one real generation. Validates LoRA serving at boot
        # (a runner that can't serve adapters must never enter the pool).
        scale, _ = self._activate_adapter("numzoo")
        self.pipe.set_adapters(["numzoo"], adapter_weights=[scale])
        self.pipe(
            prompt="warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
        )
        print(f"[setup] worker {self._worker_id} ready; resident={list(self._resident)}")

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
        # KV has no atomics: concurrent registration is last-write-wins,
        # which is the right semantics for "re-register replaces".
        self.kv.set(f"user:{user_id}", json.dumps(record))

    def _download_to(self, url: str, dest: Path) -> None:
        """Stream-download with a hard size cap; temp file + atomic rename so
        concurrent runners never see partial files. (00.4 bulletproof-inputs
        covers the fully hardened version of this gate.)"""
        import httpx

        tmp = dest.with_suffix(f".tmp-{os.getpid()}-{time.monotonic_ns()}")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                declared = int(r.headers.get("content-length", 0))
                if declared > MAX_LORA_BYTES:
                    raise FieldException(
                        field="lora_url",
                        message=f"LoRA is {declared} bytes; the limit is {MAX_LORA_BYTES} (300 MB).",
                        type="lora_too_large",
                    )
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > MAX_LORA_BYTES:
                            raise FieldException(
                                field="lora_url",
                                message=f"LoRA exceeds the {MAX_LORA_BYTES}-byte (300 MB) limit.",
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
                f for f in HfApi().list_repo_files(repo_id)
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
                    f".safetensors file (found {len(files)}). Pass the direct file URL instead."
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
        """A small, deterministic, structurally-real LoRA for this user.

        Targets the attention projections of the transformer's double blocks
        (module names introspected at runtime, so this tracks diffusers'
        actual naming). lora_B is near-zero: outputs shift subtly per user,
        never garbage. Every synthetic adapter shares one config, so the
        whole synthetic fleet is structurally uniform.
        """
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
        save_file(state, str(tmp))
        os.rename(tmp, dest)

    def _activate_adapter(self, user_id: str) -> tuple[float, str]:
        """GPU-LRU hit, else KV -> /data -> load_lora_weights -> insert+evict.
        Returns (scale, adapter_source)."""
        with self._cache_lock:
            if user_id in self._resident:
                self._resident.move_to_end(user_id)
                return self._resident[user_id], "gpu-cache"

        record = self._kv_get_user(user_id)          # 422 if unknown
        path, downloaded = self._ensure_adapter_file(user_id, record)

        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
        with self._cache_lock:
            if user_id in self._resident:            # raced another request
                self._resident.move_to_end(user_id)
                return self._resident[user_id], "gpu-cache"
            try:
                self.pipe.load_lora_weights(state_dict, adapter_name=user_id)
            except Exception as exc:
                if is_oom_error(exc):
                    raise                             # retryable 5xx, never a 422
                msg = str(exc)
                if "size" in msg.lower() and "match" in msg.lower():
                    msg = (
                        "LoRA weights are incompatible with the current model. "
                        "The LoRA was likely trained for a different model architecture or size."
                    )
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
                import gc

                import torch
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[lru] loaded adapter {user_id} (source={'downloaded' if downloaded else 'disk'})")
            return self._resident[user_id], ("downloaded" if downloaded else "disk")

    def provide_hints(self) -> list[str]:
        # The middleware calls this on every response - including ones that
        # race setup(). Return [] until the cache exists; never crash here.
        resident = getattr(self, "_resident", None)
        if not resident:
            return []
        return [hint_for(uid) for uid in resident]

    @fal.endpoint("/")
    def generate(self, request: Input, response: Response) -> Output:
        import random

        import torch

        t0 = time.perf_counter()
        scale, source = self._activate_adapter(request.user_id)
        self.pipe.set_adapters([request.user_id], adapter_weights=[scale])
        lora_s = time.perf_counter() - t0

        size = get_image_size(request.image_size)
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
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
            msg = str(exc)
            if "size" in msg.lower() and "match" in msg.lower():
                raise FieldException(
                    field="user_id",
                    message=(
                        "LoRA weights are incompatible with the current model. "
                        "The LoRA was likely trained for a different model architecture or size."
                    ),
                    type="lora_incompatible",
                ) from exc
            raise
        inference_s = time.perf_counter() - t1

        hint = hint_for(request.user_id)
        # Echo the hint so callers never compute it: capture this header and
        # pass it back as hint= (X-Fal-Runner-Hint) on the next request.
        response.headers["x-fal-runner-hint"] = hint

        repository = "in_memory" if request.sync_mode else "fal_v3"
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
        path = self._adapter_path(request.user_id)
        if request.lora_url is None:
            kind = "synthetic"
            if not path.is_file():
                self._write_synthetic_adapter(request.user_id, path)
            record = {"kind": kind, "url": None, "scale": request.scale, "path": str(path)}
        else:
            kind = "registered"
            source_kind = validate_lora_source(request.lora_url)
            url = (
                self._resolve_hf_repo(request.lora_url)
                if source_kind == "hf_repo"
                else request.lora_url
            )
            self._download_to(url, path)  # re-register replaces the file
            record = {"kind": kind, "url": url, "scale": request.scale, "path": str(path)}
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
