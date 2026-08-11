"""3D progressive rendering: stream SAM-3D reconstruction as it happens.

Catalog slot 04.7 (SERV-1319). One H100 app, two endpoints: `/` returns the
final GLB + Gaussian splat as URLs; `/stream` emits SSE while the two
diffusion stages run - voxels appear live, then the final assets arrive by
URL. The pattern to steal: model progress -> bounded queue -> SSE generator,
where progress frames are DISPOSABLE (drop-oldest) and the single terminal
event is GUARANTEED.
"""
import base64
import json
import queue
import secrets
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import FAL_PERSISTENT_DIR, File
from fal.toolkit import Image as FalImage
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Stage-1 voxel coords are indices in a fixed 64-cube grid, so uint8 is
# LOSSLESS - no per-frame renormalization, no jitter between frames.
GRID_SIZE = 64
HEARTBEAT_SECONDS = 15.0  # well under typical 60-100s proxy idle timeouts
SSE_HEARTBEAT = ": keepalive\n\n"


class GenerationCancelled(Exception):
    """Raised inside the worker when the client has disconnected."""


def encode_voxels(coords, colors=None) -> str:
    """Pack voxels as base64 of N x [x,y,z,r,g,b] uint8 rows (6 bytes/voxel).

    coords: (N,3) integer grid indices in [0, GRID_SIZE).
    colors: (N,3) floats in [0,1], or None for gray. The float->uint8 scaling
    happens HERE and only here - one encoder, one color contract.
    """
    import numpy as np

    coords = np.asarray(coords)
    packed = np.empty((len(coords), 6), dtype=np.uint8)
    packed[:, :3] = coords.astype(np.uint8)
    if colors is None:
        packed[:, 3:] = 128
    else:
        colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
        packed[:, 3:] = np.round(colors * 255).astype(np.uint8)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_voxels(payload: str) -> list[tuple[int, int, int, int, int, int]]:
    """Reference decoder (pure stdlib) - what a client does with `voxels`."""
    raw = base64.b64decode(payload)
    if len(raw) % 6:
        raise ValueError(f"voxel payload length {len(raw)} is not a multiple of 6")
    return [tuple(raw[i : i + 6]) for i in range(0, len(raw), 6)]


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


def put_progress(q: queue.Queue, event: dict) -> None:
    """Enqueue a disposable progress frame. A slow client must never stall
    the GPU: when the queue is full, drop the OLDEST frame and move on."""
    while True:
        try:
            q.put_nowait(event)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def put_terminal(q: queue.Queue, event: dict | None) -> None:
    """Enqueue a terminal event (or the None end-of-stream sentinel) with
    guaranteed delivery, evicting progress frames if the queue is full."""
    while True:
        try:
            q.put(event, timeout=0.1)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def stream_frames(q: queue.Queue, worker_done: threading.Event,
                  heartbeat_s: float = HEARTBEAT_SECONDS):
    """Drain the queue into SSE frames. Heartbeats are SSE comments (clients'
    event parsers never see them). Ends at the None sentinel; if the worker
    died without one, emit a terminal error so 'finished' is never ambiguous
    with 'connection died'."""
    while True:
        try:
            event = q.get(timeout=heartbeat_s)
        except queue.Empty:
            if worker_done.is_set() and q.empty():
                yield sse_format({
                    "type": "error", "code": "worker_died",
                    "message": "generation worker exited unexpectedly",
                })
                return
            yield SSE_HEARTBEAT
            continue
        if event is None:
            return
        yield sse_format(event)


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

SAMPLE_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"
HF_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
WEIGHTS_DIR = FAL_PERSISTENT_DIR / "sam-3d-objects" / HF_REVISION
DOCKERFILE = Path(__file__).parent / "Dockerfile"


class GenerateInput(BaseModel):
    image_url: str = Field(
        description="Image containing the object to reconstruct in 3D.",
        examples=[SAMPLE_IMAGE],
    )
    mask_url: str | None = Field(
        default=None,
        description=(
            "Optional binary mask isolating the object (white = object). "
            "Without it the whole image is reconstructed as one object - "
            "fine for object-centric photos, poor for cluttered scenes."
        ),
    )
    seed: int | None = Field(
        default=None, description="Random seed; omit for a random one."
    )
    stage1_steps: int | None = Field(
        default=None, ge=4, le=50,
        description="Geometry diffusion steps (default: model config, 25).",
    )
    stage2_steps: int | None = Field(
        default=None, ge=4, le=50,
        description="Appearance diffusion steps (default: model config, 25).",
    )


class StreamInput(GenerateInput):
    geometry_every: int = Field(
        default=1, ge=1, le=10,
        description="Emit a geometry frame every N steps (1 = smoothest).",
    )
    colors_every: int = Field(
        default=5, ge=1, le=25,
        description=(
            "Emit a colored appearance frame every N steps. Each frame costs "
            "a Gaussian decode, so this is a latency knob, not just polish."
        ),
    )


class AssetOutput(BaseModel):
    glb: File = Field(description="Final textured mesh (vertex colors), GLB.")
    gaussian_splat: File = Field(description="Gaussian splat, PLY.")
    seed: int = Field(description="Seed actually used.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class SAM3DStreamApp(
    fal.App,
    name="sam-3d-progressive",
    keep_alive=60,  # raise for interactive use; see README "Cost & tuning"
    kind="container",
    image=ContainerImage.from_dockerfile(str(DOCKERFILE), builder="depot"),
    max_concurrency=1,
    min_concurrency=0,
):
    # SAM-3D's own prerequisite is >=32 GB VRAM, which rules out most
    # smaller tiers; H100 keeps both diffusion stages interactive.
    machine_type = "GPU-H100"
    app_auth = "private"
    request_timeout = 600

    def setup(self):
        import torch
        from huggingface_hub import snapshot_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Gated checkpoint: requires one-time access approval on
        # huggingface.co/facebook/sam-3d-objects and an HF_TOKEN fal secret.
        # Pinned revision + /data cache: repeat cold starts skip the download.
        snapshot_download(
            "facebook/sam-3d-objects",
            revision=HF_REVISION,
            local_dir=str(WEIGHTS_DIR),
        )

        # Mirrors the official notebook/inference.py `Inference` wrapper
        # (that helper is not part of the pip-installed package).
        # FAL_PERSISTENT_DIR is not a real pathlib.Path - OmegaConf.load
        # rejects it with "Unexpected file type" (spike finding).
        config_path = Path(str(WEIGHTS_DIR)) / "checkpoints" / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        self.pipeline = instantiate(config)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[sam-3d-progressive] pipeline ready")

    # -- shared input/output plumbing --------------------------------------

    def _load_rgba(self, input: GenerateInput):
        """Fetch image (+ optional mask) and merge to the RGBA the pipeline
        expects. Runs BEFORE any StreamingResponse is returned, so bad inputs
        are honest HTTP 422s, never mid-stream error events."""
        import numpy as np

        try:
            rgb = np.array(FalImage(url=input.image_url).to_pil().convert("RGB"))
        except Exception as exc:
            raise FieldException(
                "image_url", "Could not fetch or decode the image."
            ) from exc

        if input.mask_url:
            try:
                mask_pil = FalImage(url=input.mask_url).to_pil().convert("L")
            except Exception as exc:
                raise FieldException(
                    "mask_url", "Could not fetch or decode the mask."
                ) from exc
            mask = (np.array(mask_pil.resize((rgb.shape[1], rgb.shape[0]))) > 127)
            mask = mask.astype(np.uint8) * 255
        else:
            mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        return np.concatenate([rgb, mask[..., None]], axis=-1)

    def _export_assets(self, outputs: dict, request: Request):
        """Final assets travel by URL, never inline - the terminal event and
        the JSON endpoint share this rule."""
        with TemporaryDirectory() as tmp:
            splat_path = str(Path(tmp) / "splat.ply")
            outputs["gs"].save_ply(splat_path)
            splat = File.from_path(splat_path, multipart=True, request=request)

            glb_path = str(Path(tmp) / "model.glb")
            outputs["glb"].export(glb_path)
            glb = File.from_path(
                glb_path, content_type="model/gltf-binary",
                multipart=True, request=request,
            )
        return glb, splat

    # -- endpoints ----------------------------------------------------------

    @fal.endpoint("/")
    def generate(self, input: GenerateInput, request: Request) -> AssetOutput:
        """Plain request/response: the whole pipeline, assets by URL.
        Use /stream to watch it happen."""
        rgba = self._load_rgba(input)
        seed = input.seed if input.seed is not None else secrets.randbits(31)
        outputs = self.pipeline.run(
            rgba, None, seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=input.stage1_steps,
            stage2_inference_steps=input.stage2_steps,
        )
        glb, splat = self._export_assets(outputs, request)
        return AssetOutput(glb=glb, gaussian_splat=splat, seed=seed)
