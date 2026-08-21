"""Bulletproof inputs - typed 422s, bomb gates, and safe URL ingestion."""
import base64
import binascii
import hashlib
import zipfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import fal
from fal.exceptions import FieldException
from fal.toolkit import download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

# Defaults mirror fal's production ingestion.
DEFAULT_MAX_DOWNLOAD_MB = 100
DEFAULT_MAX_PIXELS = 89_478_485  # PIL's own bomb threshold; ~9460x9460
DEFAULT_MAX_ARCHIVE_ENTRIES = 10_000
DEFAULT_MAX_UNCOMPRESSED_MB = 512
ALLOWED_SCHEMES = {"http", "https", "data"}
ALLOWED_PORTS = {80, 443}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
# Scratch downloads stay off the persistent /data volume.
DOWNLOAD_DIR = Path("/tmp/bulletproof-inputs")


class Input(BaseModel):
    image_url: str = Field(
        description="Image to validate. http(s) URL or a data: URI.",
        examples=["https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"],
    )
    archive_url: str | None = Field(
        default=None,
        description="Optional zip archive to inspect (never extracted unvetted).",
    )
    max_download_mb: int = Field(
        default=DEFAULT_MAX_DOWNLOAD_MB,
        ge=1,
        le=1024,
        description="Download size cap in MB (download_file has no default cap).",
    )
    max_pixels: int = Field(
        default=DEFAULT_MAX_PIXELS,
        ge=1024,
        le=DEFAULT_MAX_PIXELS,
        description="Decode budget in pixels, checked before the image is decoded.",
    )
    max_archive_entries: int = Field(
        default=DEFAULT_MAX_ARCHIVE_ENTRIES,
        ge=1,
        description="Entry-count cap for archives.",
    )
    max_uncompressed_mb: int = Field(
        default=DEFAULT_MAX_UNCOMPRESSED_MB,
        ge=1,
        description="Declared-uncompressed-size cap for archives.",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png", description="Unused by validation - here to show a native enum 422."
    )


class ArchiveStats(BaseModel):
    entries: int = Field(description="Number of entries in the archive.")
    uncompressed_mb: float = Field(description="Declared uncompressed size in MB.")
    compression_ratio: float = Field(
        description="Uncompressed-to-compressed ratio, returned so callers can "
        "flag bomb-like archives (ratios over ~100); not a gate here."
    )


class Output(BaseModel):
    accepted: bool = Field(
        description="Always true in a 200 response - rejections never reach "
        "this payload; they surface as typed 422s."
    )
    format: str = Field(description="Detected image format.")
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    file_size: int = Field(description="Downloaded size in bytes.")
    gates: list[str] = Field(description="Gates this request passed, in order.")
    archive: ArchiveStats | None = Field(
        default=None, description="Archive stats when one was supplied."
    )


class BulletproofInputs(fal.App, keep_alive=60):
    machine_type = "M"  # validation is CPU work
    app_auth = "private"
    requirements = [
        "fal==1.78.2",
        "pillow==11.3.0",
    ]

    def setup(self) -> None:
        """Nothing to load - the gates are pure Python."""

    @fal.endpoint("/")
    def validate(self, request: Input) -> Output:
        from PIL import Image as PILImage

        gates = []

        check_url_policy(request.image_url, "image_url")
        gates.append("url_policy")

        # download_file's filesize_limit defaults to None - the cap must be passed.
        path = download_guarded(request.image_url, request.max_download_mb, "image_url")
        size_bytes = path.stat().st_size
        gates.append("download_cap")

        try:
            image = PILImage.open(path)
        except Exception as exc:
            raise FieldException(
                field="image_url",
                message="not a decodable image",
                type="image_load_error",
            ) from exc
        if image.format not in ALLOWED_FORMATS:
            raise FieldException(
                field="image_url",
                message=f"unsupported format {image.format}; allowed: "
                f"{', '.join(sorted(ALLOWED_FORMATS))}",
                type="unsupported_image_format",
            )
        gates.append("format_allowlist")

        # Header dims only - no pixels decoded until load().
        check_pixel_budget(image.width, image.height, request.max_pixels, "image_url")
        gates.append("pixel_budget")

        try:
            image.load()
        except Exception as exc:
            raise FieldException(
                field="image_url",
                message="image data is truncated or corrupt",
                type="image_load_error",
            ) from exc
        gates.append("decode")

        archive_stats = None
        if request.archive_url:
            check_url_policy(request.archive_url, "archive_url")
            archive_path = download_guarded(
                request.archive_url, request.max_download_mb, "archive_url"
            )
            archive_stats = inspect_zip(
                archive_path,
                request.max_archive_entries,
                request.max_uncompressed_mb,
                "archive_url",
            )
            gates.append("archive_gates")

        return Output(
            accepted=True,
            format=image.format,
            width=image.width,
            height=image.height,
            file_size=size_bytes,
            gates=gates,
            archive=archive_stats,
        )


def check_url_policy(url: str, field: str) -> None:
    """Scheme/port policy - the SDK's SSRF client ignores ports, and data: URIs bypass it."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FieldException(
            field=field,
            message=f"unsupported URL scheme {parsed.scheme or '(none)'!r}; "
            f"allowed: {', '.join(sorted(ALLOWED_SCHEMES))}",
            type="value_error",
        )
    if parsed.scheme != "data" and parsed.port is not None:
        if parsed.port not in ALLOWED_PORTS:
            raise FieldException(
                field=field,
                message=f"port {parsed.port} is not allowed; allowed: "
                f"{', '.join(str(p) for p in sorted(ALLOWED_PORTS))}",
                type="value_error",
            )


def check_pixel_budget(width: int, height: int, max_pixels: int, field: str) -> None:
    """Bomb gate on header dimensions, before any decode - Pillow only warns at this size."""
    area = width * height
    if area > max_pixels:
        raise FieldException(
            field=field,
            message=f"image is {width}x{height} ({area:,} pixels); "
            f"maximum is {max_pixels:,} pixels",
            type="image_too_large",
        )


def inspect_zip(
    path: Path, max_entries: int, max_uncompressed_mb: int, field: str
) -> ArchiveStats:
    """Archive gates, applied before extraction - fal's CompressedFile extracts with no limits."""
    max_uncompressed = max_uncompressed_mb * 1024 * 1024
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise FieldException(
            field=field, message="not a readable zip archive", type="invalid_archive"
        ) from exc

    if len(infos) > max_entries:
        raise FieldException(
            field=field,
            message=f"archive has {len(infos):,} entries; maximum is {max_entries:,}",
            type="archive_file_count_exceeds_maximum",
        )
    uncompressed = sum(info.file_size for info in infos)
    if uncompressed > max_uncompressed:
        raise FieldException(
            field=field,
            message=f"archive expands to {uncompressed / 1024 / 1024:.1f} MB; "
            f"maximum is {max_uncompressed_mb} MB",
            type="file_too_large",
        )
    compressed = sum(info.compress_size for info in infos) or 1
    ratio = uncompressed / compressed
    # Callers who extract must reject absolute/..-traversal member names first.
    unsafe = [
        info.filename
        for info in infos
        if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts
    ]
    if unsafe:
        raise FieldException(
            field=field,
            message=f"archive contains paths that escape the extraction directory: {unsafe[0]}",
            type="invalid_archive",
        )
    return ArchiveStats(
        entries=len(infos),
        uncompressed_mb=round(uncompressed / 1024 / 1024, 2),
        compression_ratio=round(ratio, 1),
    )


def read_data_uri(url: str, target_dir: Path, max_mb: int, field: str) -> Path:
    """Materialize a data: URI - download_file would log the full payload and cap only post-decode."""
    header, _, payload = url.partition(",")
    if not payload:
        raise FieldException(field=field, message="malformed data URI", type="value_error")
    is_base64 = ";base64" in header
    max_bytes = max_mb * 1024 * 1024
    estimated = len(payload) * 3 // 4 if is_base64 else len(payload)
    if estimated > max_bytes:
        raise FieldException(
            field=field,
            message=f"inline payload is ~{estimated / 1024 / 1024:.1f} MB; "
            f"maximum is {max_mb} MB",
            type="file_too_large",
        )
    try:
        data = base64.b64decode(payload, validate=True) if is_base64 else payload.encode()
    except (binascii.Error, ValueError) as exc:
        raise FieldException(
            field=field, message="data URI payload is not valid base64", type="value_error"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"inline-{hashlib.sha256(data).hexdigest()[:16]}"
    path.write_bytes(data)
    return path


def download_guarded(url: str, max_mb: int, field: str) -> Path:
    """Fetch a user-supplied URL, mapping every DownloadError onto a typed 422 (else it's a 500)."""
    if urlparse(url).scheme == "data":
        return read_data_uri(url, DOWNLOAD_DIR, max_mb, field)
    # No timeout kwarg in fal 1.78.2; the platform request timeout bounds this fetch.
    try:
        return download_file(url, DOWNLOAD_DIR, filesize_limit=max_mb)
    except DownloadError as exc:
        detail = str(exc)
        if "exceeded" in detail or "over the limit" in detail:
            raise FieldException(
                field=field,
                message=f"file exceeds the {max_mb} MB limit",
                type="file_too_large",
            ) from exc
        # SSRF refusals from the SDK arrive here verbatim.
        raise FieldException(
            field=field, message=f"could not download: {detail}", type="file_download_error"
        ) from exc
