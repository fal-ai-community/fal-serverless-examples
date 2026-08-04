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
        raise NotImplementedError  # Task 3

    def provide_hints(self) -> list[str]:
        # The middleware calls this on every response - including ones that
        # race setup(). Return [] until the cache exists; never crash here.
        resident = getattr(self, "_resident", None)
        if not resident:
            return []
        return [hint_for(uid) for uid in resident]

    @fal.endpoint("/")
    def generate(self, request: Input, response: Response) -> Output:
        raise NotImplementedError  # Task 3

    @fal.endpoint("/register")
    def register(self, request: RegisterInput, response: Response) -> RegisterOutput:
        raise NotImplementedError  # Task 4
