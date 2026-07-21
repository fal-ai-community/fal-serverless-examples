"""Media in, media out - files, CDN, and persistent storage on fal.

Catalog slot 00.3 (SERV-1302). No model here on purpose: this example is
about how media moves through a fal app - URL in, safe bounded download,
/data caching, processing on a cheap CPU machine, and typed outputs on
fal's CDN with the content type you asked for.
"""
import io
import json
import time
from typing import Literal

import fal
from fal.exceptions import FieldException
from fal.toolkit import File, Image, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

MAX_DOWNLOAD_MB = 20
THUMBNAIL_EDGE = 256


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
        description="Longest edge of the processed image, in pixels.",
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

        # MEDIA IN - a bounded download with /data caching built in: a relative
        # target_dir lands under /data (persistent, account-wide), and a repeat
        # request for the same URL reuses the cached file instead of re-downloading.
        t0 = time.monotonic()
        try:
            source_path = download_file(
                request.image_url, "cache/media-in", filesize_limit=MAX_DOWNLOAD_MB
            )
        except DownloadError as exc:
            raise FieldException(
                field="image_url",
                message=f"could not download (limit {MAX_DOWNLOAD_MB} MB): {exc}",
            )
        download_ms = int((time.monotonic() - t0) * 1000)

        # PROCESS - plain Pillow work: honor EXIF rotation, bound the size.
        t1 = time.monotonic()
        try:
            source = PILImage.open(source_path)
            source = ImageOps.exif_transpose(source)
        except Exception:
            raise FieldException(field="image_url", message="not a decodable image")
        original_size = (source.width, source.height)
        if request.output_format == "jpeg":
            source = source.convert("RGB")  # JPEG has no alpha channel
        processed = source.copy()
        processed.thumbnail((request.max_size, request.max_size), PILImage.LANCZOS)
        thumb = source.copy()
        thumb.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), PILImage.LANCZOS)
        processed_ms = int((time.monotonic() - t1) * 1000)

        # MEDIA OUT - format choice becomes the file's content type; an optional
        # per-file retention rides along as an object lifecycle preference.
        save_kwargs = None
        if request.expires_in_seconds is not None:
            save_kwargs = {
                "object_lifecycle_preference": {
                    "expiration_duration_seconds": request.expires_in_seconds
                }
            }
        image = Image.from_pil(
            processed,
            format=request.output_format,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )
        thumbnail = Image.from_pil(
            thumb,
            format=request.output_format,
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )

        report_data = {
            "source": {
                "url": request.image_url,
                "width": original_size[0],
                "height": original_size[1],
                "cached_at": str(source_path),
            },
            "output_format": request.output_format,
            "download_ms": download_ms,  # near-zero when /data already has the file
            "processed_ms": processed_ms,
        }
        report = File.from_bytes(
            json.dumps(report_data, indent=2).encode(),
            content_type="application/json",
            file_name="report.json",
            save_kwargs=save_kwargs,
            fallback_save_kwargs=save_kwargs,
        )
        return Output(image=image, thumbnail=thumbnail, report=report)
