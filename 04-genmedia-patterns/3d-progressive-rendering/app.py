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
import random
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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
MAX_PENDING_FRAMES = 8
SSE_HEARTBEAT = ": keepalive\n\n"

SAMPLE_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"
HF_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
WEIGHTS_DIR = FAL_PERSISTENT_DIR / "sam-3d-objects" / HF_REVISION
DOCKERFILE = Path(__file__).parent / "Dockerfile"


class GenerationCancelled(Exception):
    """Raised inside the worker when the client has disconnected."""


class Input(BaseModel):
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


class StreamInput(Input):
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


class Output(BaseModel):
    # No sync_mode on purpose: these assets run to hundreds of MB, so
    # data-URI transport is never appropriate here.
    model_glb: File = Field(description="Final textured mesh (vertex colors), GLB.")
    gaussian_splat: File = Field(description="Gaussian splat, PLY.")
    seed: int = Field(description="Seed actually used.")


def resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, 2**32 - 1)


def encode_voxels(coords: "np.ndarray", colors: "np.ndarray | None" = None) -> str:
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
                  heartbeat_s: float = HEARTBEAT_SECONDS) -> Iterator[str]:
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


def stream_generate(pipeline: Any, rgba: "np.ndarray", seed: int,
                    params: StreamInput, emit: Callable[[dict], None],
                    cancelled: threading.Event) -> dict:
    """The streaming driver: mirror InferencePipeline.run() stage by stage,
    emitting a frame per solver step via `emit`.

    The pipeline's generators expose generate_iter(), a public per-solver-step
    iterator (generate() is literally "loop generate_iter, keep the last
    value") - so progressive output needs no model fork.
    """
    import numpy as np
    import torch
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )

    SH_C0 = 0.28209479177387814  # SH degree-0 constant: rgb = 0.5 + C0 * f_dc

    def check_cancelled():
        if cancelled.is_set():
            raise GenerationCancelled()

    models = pipeline.models
    with pipeline.device:
        emit({"type": "status", "stage": "preprocessing", "progress": 0.03,
              "message": "Preparing pipeline inputs"})
        # InferencePipelinePointMap conditions stage 1 on a MoGe depth
        # pointmap - without it the condition embedder KeyErrors on
        # 'pointmap'. Stage 2 does not take it.
        pointmap_dict = pipeline.compute_pointmap(rgba, None)
        ss_input = pipeline.preprocess_image(
            rgba, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"]
        )
        slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        torch.manual_seed(seed)

        # ---- stage 1: geometry (sparse structure) -------------------------
        ss_gen, ss_dec = models["ss_generator"], models["ss_decoder"]
        ss_gen.no_shortcut = True
        ss_gen.reverse_fn.strength = pipeline.ss_cfg_strength
        ss_gen.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
        prev_ss_steps = ss_gen.inference_steps
        if params.stage1_steps:
            ss_gen.inference_steps = params.stage1_steps
        total_ss_steps = ss_gen.inference_steps

        image = ss_input["image"]
        bs = image.shape[0]
        if pipeline.is_mm_dit():
            latent_shape = {
                k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                for k, v in ss_gen.reverse_fn.backbone.latent_mapping.items()
            }
        else:
            latent_shape = (bs,) + (4096, 8)
        cond_args, cond_kwargs = pipeline.get_condition_input(
            pipeline.condition_embedders["ss_condition_embedder"],
            ss_input, pipeline.ss_condition_input_mapping,
        )

        coords = None
        step = 0
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.shape_model_dtype):
                for _t, x_t, _ in ss_gen.generate_iter(
                    latent_shape, image.device, *cond_args, **cond_kwargs
                ):
                    step += 1
                    check_cancelled()
                    if step % params.geometry_every != 0 and step != total_ss_steps:
                        continue
                    shape_latent = x_t["shape"] if isinstance(x_t, dict) else x_t
                    ss = ss_dec(
                        shape_latent.permute(0, 2, 1).contiguous().view(bs, 8, 16, 16, 16)
                    )
                    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
                    grid_xyz = coords[:, 1:].cpu().numpy()
                    emit({
                        "type": "geometry", "step": step, "total_steps": total_ss_steps,
                        "progress": round(0.05 + 0.45 * step / total_ss_steps, 3),
                        "voxels": encode_voxels(grid_xyz),
                        "count": int(len(grid_xyz)), "grid_size": GRID_SIZE,
                    })
        finally:
            # Restore even on cancellation/exception: these generators are
            # shared across requests, and a leaked step count would silently
            # apply to every later request on this warm container.
            ss_gen.inference_steps = prev_ss_steps

        # Final stage-1 coords, pruned/downsampled exactly like run() does.
        if pipeline.downsample_ss_dist > 0:
            coords = prune_sparse_structure(
                coords, max_neighbor_axes_dist=pipeline.downsample_ss_dist
            )
        coords, _ = downsample_sparse_structure(coords)

        # ---- stage 2: appearance (structured latent) ----------------------
        slat_gen = models["slat_generator"]
        slat_gen.no_shortcut = True
        slat_gen.reverse_fn.strength = pipeline.slat_cfg_strength
        prev_slat_steps = slat_gen.inference_steps
        if params.stage2_steps:
            slat_gen.inference_steps = params.stage2_steps
        total_slat_steps = slat_gen.inference_steps

        slat_latent_shape = (bs,) + (coords.shape[0], 8)
        slat_cond_args, slat_cond_kwargs = pipeline.get_condition_input(
            pipeline.condition_embedders["slat_condition_embedder"],
            slat_input, pipeline.slat_condition_input_mapping,
        )
        slat_cond_args = slat_cond_args + (coords.cpu().numpy(),)
        grid_xyz = coords[:, 1:].cpu().numpy()
        n_vox = coords.shape[0]

        def voxel_colors(x_t) -> "np.ndarray | None":
            """Decode an intermediate slat to per-voxel float [0,1] colors via
            the Gaussian decoder's degree-0 SH. Costs one decode - which is
            why colors_every is a knob. Returns None if the gaussian layout
            doesn't map cleanly onto voxels (then frames stay gray)."""
            st = sp.SparseTensor(coords=coords, feats=x_t[0]).to(image.device)
            st = st * pipeline.slat_std.to(image.device) + pipeline.slat_mean.to(image.device)
            g = models["slat_decoder_gs"](st)[0]
            fdc = getattr(g, "_features_dc", None)
            if fdc is None:
                return None
            fdc = fdc.reshape(fdc.shape[0], -1)[:, :3]
            if fdc.shape[0] == n_vox:
                per_voxel = fdc
            elif fdc.shape[0] % n_vox == 0:
                per_voxel = fdc.view(n_vox, -1, 3).mean(dim=1)
            else:
                return None
            return (0.5 + SH_C0 * per_voxel).clamp(0, 1).float().cpu().numpy()

        step = 0
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
                for _t, x_t, _ in slat_gen.generate_iter(
                    slat_latent_shape, image.device, *slat_cond_args, **slat_cond_kwargs
                ):
                    step += 1
                    check_cancelled()
                    # Track every step's latent BEFORE the throttle: the final
                    # decode below needs the LAST one even when its frame is
                    # skipped by colors_every.
                    slat_latent = x_t
                    if step % params.colors_every != 0 and step != total_slat_steps:
                        continue
                    colors = voxel_colors(x_t)
                    emit({
                        "type": "appearance", "step": step, "total_steps": total_slat_steps,
                        "progress": round(0.50 + 0.40 * step / total_slat_steps, 3),
                        "voxels": encode_voxels(grid_xyz, colors),
                        "count": int(n_vox), "grid_size": GRID_SIZE,
                    })
        finally:
            # Same restore rationale as stage 1.
            slat_gen.inference_steps = prev_slat_steps
        slat = sp.SparseTensor(coords=coords, feats=slat_latent[0]).to(image.device)
        slat = slat * pipeline.slat_std.to(image.device) + pipeline.slat_mean.to(image.device)

        # ---- finalize: decode + GLB, exactly like run() --------------------
        emit({"type": "status", "stage": "finalizing", "progress": 0.90,
              "message": "Decoding mesh + gaussians, exporting GLB"})
        check_cancelled()
        outputs = pipeline.decode_slat(slat, pipeline.decode_formats)
        outputs = pipeline.postprocess_slat_output(
            outputs, with_mesh_postprocess=False, with_texture_baking=False,
            use_vertex_color=True,
        )

        glb = outputs.get("glb")
        if glb is not None and hasattr(glb, "vertices"):
            verts = np.asarray(glb.vertices, dtype=np.float32)
            faces = np.asarray(glb.faces, dtype=np.uint32)
            colors_u8 = None
            visual = getattr(glb, "visual", None)
            if visual is not None and getattr(visual, "vertex_colors", None) is not None:
                colors_u8 = np.asarray(visual.vertex_colors, dtype=np.uint8)[:, :3]
            # A frame, not an asset: cap the inline preview.
            if len(verts) <= 200_000:
                emit({
                    "type": "mesh_preview",
                    "vertices": base64.b64encode(verts.tobytes()).decode("ascii"),
                    "faces": base64.b64encode(faces.tobytes()).decode("ascii"),
                    "colors": None if colors_u8 is None
                    else base64.b64encode(colors_u8.tobytes()).decode("ascii"),
                    "vertex_count": int(len(verts)),
                    "face_count": int(len(faces)),
                })
        return outputs


class SAM3DStreamApp(
    fal.App,
    name="sam-3d-progressive",
    keep_alive=60,  # raise for interactive use; see README "Cost & tuning"
):
    # SAM-3D's own prerequisite is >=32 GB VRAM, which rules out most
    # smaller tiers; H100 keeps both diffusion stages interactive.
    machine_type = "GPU-H100"
    app_auth = "private"
    request_timeout = 600
    # The multi-GB CUDA image needs the faster depot builder.
    image = ContainerImage.from_dockerfile(str(DOCKERFILE), builder="depot")
    max_concurrency = 1
    min_concurrency = 0

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
        # rejects it with "Unexpected file type".
        config_path = Path(str(WEIGHTS_DIR)) / "checkpoints" / "pipeline.yaml"
        config = OmegaConf.load(config_path)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_path.parent)
        self.pipeline = instantiate(config)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # No warmup generation on purpose: compile_model=False leaves no
        # compile cache to prime, and a full two-stage pass would add ~16s to
        # every cold start that the streaming UX already makes visible.
        print("[sam-3d-progressive] pipeline ready")

    # -- shared input/output plumbing --------------------------------------

    def _load_rgba(self, request: Input) -> "np.ndarray":
        """Fetch image (+ optional mask) and merge to the RGBA the pipeline
        expects. Runs BEFORE any StreamingResponse is returned, so bad inputs
        are honest HTTP 422s, never mid-stream error events."""
        import numpy as np

        try:
            rgb = np.array(FalImage(url=request.image_url).to_pil().convert("RGB"))
        except Exception as exc:
            raise FieldException(
                field="image_url",
                message="Could not fetch or decode the image.",
                type="file_download_error",
            ) from exc

        if request.mask_url:
            try:
                mask_pil = FalImage(url=request.mask_url).to_pil().convert("L")
            except Exception as exc:
                raise FieldException(
                    field="mask_url",
                    message="Could not fetch or decode the mask.",
                    type="file_download_error",
                ) from exc
            mask = (np.array(mask_pil.resize((rgb.shape[1], rgb.shape[0]))) > 127)
            mask = mask.astype(np.uint8) * 255
        else:
            mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        return np.concatenate([rgb, mask[..., None]], axis=-1)

    def _export_assets(self, outputs: dict, fastapi_request: Request) -> tuple[File, File]:
        """Final assets travel by URL, never inline - the terminal event and
        the JSON endpoint share this rule."""
        with TemporaryDirectory() as tmp:
            splat_path = str(Path(tmp) / "splat.ply")
            outputs["gs"].save_ply(splat_path)
            splat = File.from_path(splat_path, multipart=True, request=fastapi_request)

            glb_path = str(Path(tmp) / "model.glb")
            outputs["glb"].export(glb_path)
            glb = File.from_path(
                glb_path, content_type="model/gltf-binary",
                multipart=True, request=fastapi_request,
            )
        return glb, splat

    # -- endpoints ----------------------------------------------------------

    @fal.endpoint("/")
    def generate(self, request: Input, fastapi_request: Request) -> Output:
        """Plain request/response: the whole pipeline, assets by URL.
        Use /stream to watch it happen."""
        rgba = self._load_rgba(request)
        seed = resolve_seed(request.seed)
        outputs = self.pipeline.run(
            rgba,
            mask=None,  # the mask is already merged into rgba's alpha
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=request.stage1_steps,
            stage2_inference_steps=request.stage2_steps,
        )
        glb, splat = self._export_assets(outputs, fastapi_request)
        return Output(model_glb=glb, gaussian_splat=splat, seed=seed)

    @fal.endpoint("/stream")
    def stream(self, request: StreamInput, fastapi_request: Request) -> StreamingResponse:
        """SSE progressive rendering. Every stream ends in exactly one
        terminal event: complete (asset URLs) or error (sanitized)."""
        rgba = self._load_rgba(request)  # bad inputs 422 BEFORE we stream
        seed = resolve_seed(request.seed)

        q: queue.Queue[dict | None] = queue.Queue(maxsize=MAX_PENDING_FRAMES)
        cancelled = threading.Event()
        worker_done = threading.Event()

        def worker() -> None:
            try:
                put_progress(q, {"type": "status", "stage": "loading",
                                 "progress": 0.01, "message": "Starting"})
                outputs = stream_generate(
                    self.pipeline, rgba, seed, request,
                    emit=lambda ev: put_progress(q, ev),
                    cancelled=cancelled,
                )
                glb, splat = self._export_assets(outputs, fastapi_request)
                put_terminal(q, {
                    "type": "complete", "progress": 1.0,
                    "model_glb": {"url": glb.url,
                                  "content_type": "model/gltf-binary"},
                    "gaussian_splat": {"url": splat.url,
                                       "content_type": "application/octet-stream"},
                    "seed": seed,
                })
            except GenerationCancelled:
                pass  # client left; nobody is reading
            except Exception:
                traceback.print_exc()  # full detail server-side ONLY
                put_terminal(q, {"type": "error", "code": "generation_failed",
                                 "message": "3D generation failed; check app logs."})
            finally:
                # Sentinel BEFORE the done flag: if worker_done were set
                # first, a heartbeat timeout in the gap would misread a
                # finished stream as worker_died.
                put_terminal(q, None)  # end-of-stream sentinel, guaranteed
                worker_done.set()

        threading.Thread(target=worker, daemon=True).start()

        def frames() -> Iterator[str]:
            try:
                yield from stream_frames(q, worker_done)
            finally:
                # Runs on normal completion AND client disconnect: signal the
                # worker, which checks between solver steps.
                cancelled.set()

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={
                # Optional fal-gateway nicety: plain synchronous calls to
                # /stream get the LAST event (complete/error) as the body.
                "x-fal-sse-last-event-as-body": "1",
                "Cache-Control": "no-cache",
            },
        )
