"""Media in, media out - files, CDN, and persistent storage on fal.

No model here on purpose: this example is
about how media moves through a fal app - URL in, safe bounded download,
/data caching, processing on a cheap CPU machine, and typed outputs on
fal's CDN with the content type you asked for.
"""

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import DownloadError, File, Image, download_file
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
    thumbnail: Image = Field(
        description=f"A thumbnail of the same image, at most {THUMBNAIL_EDGE}px."
    )
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

    @fal.endpoint("/")
    def process(self, request: Input) -> Output:
        from PIL import Image as PILImage
        from PIL import ImageOps

        # MEDIA IN - cache on /data keyed by URL hash; repeats skip the transfer.
        t0 = time.perf_counter()
        cache_dir = (
            CACHE_ROOT / hashlib.sha256(request.image_url.encode()).hexdigest()[:16]
        )
        # download_file stages partial downloads as same-directory .tmp files;
        # never serve one as the cached image.
        cached = next(
            (p for p in sorted(cache_dir.glob("*")) if ".tmp" not in p.name), None
        )
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
                    type="file_download_error",
                ) from exc
        download_ms = int((time.perf_counter() - t0) * 1000)

        # PROCESS - the pixel gate sits between open() and load(): reject
        # oversized images before decode allocates.
        t1 = time.perf_counter()
        try:
            source: PILImage.Image = PILImage.open(source_path)
            if source.width * source.height > MAX_PIXELS:
                raise FieldException(
                    field="image_url",
                    message=(
                        f"image too large ({source.width}x{source.height}); "
                        f"max ~{MAX_PIXELS // 1_000_000} megapixels"
                    ),
                    type="image_too_large",
                )
            source.load()
        except FieldException:
            raise
        except PILImage.DecompressionBombError:
            # open() itself rejects images over 2x MAX_PIXELS, before our gate runs
            raise FieldException(
                field="image_url",
                message=f"image too large; max ~{MAX_PIXELS // 1_000_000} megapixels",
                type="image_too_large",
            ) from None
        except Exception:
            raise FieldException(
                field="image_url",
                message="not a decodable image",
                type="unsupported_image_format",
            ) from None
        # best-effort: bad EXIF metadata must not reject a decodable image
        with contextlib.suppress(Exception):
            source = ImageOps.exif_transpose(source)
        original_size = (source.width, source.height)
        if request.output_format == "jpeg":
            if source.mode != "RGB":
                source = source.convert("RGB")  # JPEG has no alpha channel
        elif source.mode not in ("RGB", "RGBA"):
            source = source.convert("RGBA")
        source.thumbnail(
            (request.max_size, request.max_size), PILImage.Resampling.LANCZOS
        )
        processed = source
        thumb = processed.copy()  # derive from the smaller image, not the original
        thumb.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), PILImage.Resampling.LANCZOS)
        processed_ms = int((time.perf_counter() - t1) * 1000)

        # MEDIA OUT
        repository: Literal["in_memory", "fal_v3"] = (
            "in_memory" if request.sync_mode else "fal_v3"
        )
        save_kwargs = None
        if request.expires_in_seconds is not None:
            save_kwargs = {
                "object_lifecycle_preference": {
                    "expiration_duration_seconds": request.expires_in_seconds
                }
            }
        t2 = time.perf_counter()
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
        upload_ms = int((time.perf_counter() - t2) * 1000)

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
