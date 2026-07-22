"""Media in, media out - files, CDN, and persistent storage on fal.

Catalog slot 00.3 (SERV-1302). No model here on purpose: this example is
about how media moves through a fal app - URL in, safe bounded download,
/data caching, processing on a cheap CPU machine, and typed outputs on
fal's CDN with the content type you asked for.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, Image, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

MAX_DOWNLOAD_MB = 20
MAX_PIXELS = 89_478_485  # decode gate: PIL's own bomb error only fires at 2x this
THUMBNAIL_EDGE = 256
CACHE_ROOT = Path("/data/cache/media-in")


class Input(BaseModel):
    image_url: str = Field(
        # The `_url` suffix gives the playground a drag-and-drop upload widget.
        description="Image to process: paste a URL or upload a file.",
        examples=["https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"],
    )
    max_size: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description=(
            "Upper bound for the longest edge, in pixels. Aspect ratio is "
            "preserved and images are never upscaled."
        ),
    )
    output_format: Literal["jpeg", "png", "webp"] = Field(
        default="webp",
        description=(
            "jpeg = small lossy files, png = lossless with transparency, "
            "webp = modern balance of both. Sets the output content type."
        ),
    )
    expires_in_seconds: int | None = Field(
        default=None,
        ge=60,
        description=(
            "Optional retention for the uploaded outputs; omit to use your "
            "account's default media lifecycle."
        ),
    )
    sync_mode: bool = Field(
        default=False,
        description=(
            "Return outputs as data URIs instead of hosted CDN URLs - "
            "skips the upload round-trip for latency-sensitive callers."
        ),
    )


class Output(BaseModel):
    image: Image = Field(description="The processed image, hosted on fal's CDN.")
    thumbnail: Image = Field(description=f"A {THUMBNAIL_EDGE}px thumbnail of the same image.")
    report: File = Field(
        description="A JSON report of what happened (dimensions, bytes, timings)."
    )


class MediaInMediaOut(fal.App, keep_alive=60):
    machine_type = "M"  # 2 vCPUs / 2 GB - image I/O needs no GPU
    app_auth = "private"
    # Authoritative deploy-time deps; requirements.txt mirrors this exactly.
    requirements = [
        "fal==1.78.2",
        "pillow==11.3.0",
    ]

    def setup(self) -> None:
        """Nothing to load - media apps are I/O-bound, not model-bound."""

    @fal.endpoint("/")
    def process(self, request: Input) -> Output:
        from PIL import Image as PILImage
        from PIL import ImageOps

        # MEDIA IN - a bounded, SSRF-guarded download (both built into
        # download_file), cached on /data persistent storage. The cache is
        # keyed by URL hash so a repeat request for the same URL skips the
        # transfer entirely - across workers, deploys, and CI runs.
        t0 = time.monotonic()
        cache_dir = CACHE_ROOT / hashlib.sha256(request.image_url.encode()).hexdigest()[:16]
        cached = next(cache_dir.glob("*"), None) if cache_dir.is_dir() else None
        if cached is not None:
            source_path = cached
        else:
            try:
                source_path = download_file(
                    request.image_url, cache_dir, filesize_limit=MAX_DOWNLOAD_MB
                )
            except DownloadError as exc:
                raise FieldException(
                    field="image_url",
                    message=f"could not download (limit {MAX_DOWNLOAD_MB} MB): {exc}",
                )
        download_ms = int((time.monotonic() - t0) * 1000)

        # PROCESS - plain Pillow work with the gotchas handled the way
        # production apps do: decode errors are a typed 422, oversized pixel
        # counts are rejected before decode allocates, EXIF rotation is
        # best-effort, and exotic modes (CMYK, palette) are normalized to
        # something the chosen encoder can actually write.
        t1 = time.monotonic()
        try:
            source = PILImage.open(source_path)
            if source.width * source.height > MAX_PIXELS:
                raise FieldException(
                    field="image_url",
                    message=(
                        f"image too large ({source.width}x{source.height}); "
                        f"max ~{MAX_PIXELS // 1_000_000} megapixels"
                    ),
                )
            source.load()
        except FieldException:
            raise
        except Exception:
            raise FieldException(field="image_url", message="not a decodable image")
        try:
            source = ImageOps.exif_transpose(source)
        except Exception:
            pass  # best-effort: bad EXIF metadata must not reject a decodable image
        original_size = (source.width, source.height)
        if source.mode not in ("RGB", "RGBA"):
            source = source.convert(
                "RGBA" if request.output_format in ("png", "webp") else "RGB"
            )
        if request.output_format == "jpeg" and source.mode == "RGBA":
            source = source.convert("RGB")  # JPEG has no alpha channel
        processed = source.copy()
        processed.thumbnail((request.max_size, request.max_size), PILImage.LANCZOS)
        thumb = processed.copy()  # derive from the smaller image, not the original
        thumb.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), PILImage.LANCZOS)
        processed_ms = int((time.monotonic() - t1) * 1000)

        # MEDIA OUT - format choice becomes the file's content type; an optional
        # per-file retention rides along as an object lifecycle preference; and
        # sync_mode swaps the CDN for inline data URIs.
        repository = "in_memory" if request.sync_mode else "fal_v3"
        save_kwargs = None
        if request.expires_in_seconds is not None:
            save_kwargs = {
                "object_lifecycle_preference": {
                    "expiration_duration_seconds": request.expires_in_seconds
                }
            }
        t2 = time.monotonic()
        image = Image.from_pil(
            processed,
            format=request.output_format,
            repository=repository,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )
        thumbnail = Image.from_pil(
            thumb,
            format=request.output_format,
            repository=repository,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )
        upload_ms = int((time.monotonic() - t2) * 1000)

        report_data = {
            "source": {
                "url": request.image_url,
                "width": original_size[0],
                "height": original_size[1],
                "cache_hit": cached is not None,
            },
            "output_format": request.output_format,
            "download_ms": download_ms,  # near-zero on a cache hit
            "processed_ms": processed_ms,
            "upload_ms": upload_ms,  # image + thumbnail; 0-ish in sync_mode
        }
        report = File.from_bytes(
            json.dumps(report_data, indent=2).encode(),
            content_type="application/json",
            file_name="report.json",
            repository=repository,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )
        return Output(image=image, thumbnail=thumbnail, report=report)
