"""Bulletproof inputs - typed 422s, bomb gates, and safe URL ingestion.

Catalog slot 00.4 (SERV-1309). Every public endpoint eventually meets a
hostile input: a URL pointing at your own metadata service, a 400 MB
"thumbnail", a 200x200 PNG that decodes to 144 megapixels, a 200 KB zip
that unpacks to 200 GB. This example is the gauntlet those inputs hit
before your model ever sees them - and the typed errors callers get back.

What fal already does for you (verified against fal 1.78.2, see README):
  - download_file() fetches http(s) through an SSRF-safe client: private,
    loopback, link-local (169.254.169.254), CGNAT and IPv6-transition
    addresses are refused; DNS results are pinned so the host can't rebind
    between check and connect; each redirect hop is re-validated.
  - pydantic Field constraints become documented-shape 422s for free.
What is yours to add - the subject of this example:
  - a size cap (download_file has none by default), a port/scheme policy,
    a pixel gate before decode, archive gates before extraction, and
    machine-readable error types on every rejection.
"""
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

# Defaults mirror what fal's own production ingestion enforces.
DEFAULT_MAX_DOWNLOAD_MB = 100
DEFAULT_MAX_PIXELS = 89_478_485  # PIL's own bomb threshold; ~9460x9460
ALLOWED_SCHEMES = {"http", "https", "data"}
ALLOWED_PORTS = {80, 443}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def reject(field: str, message: str, type: str) -> FieldException:
    """A 422 the caller can branch on.

    `type` is the machine-readable code; fal's documented error taxonomy
    (docs.fal.ai/errors) is the vocabulary to draw from, so clients that
    already handle fal model errors handle yours too.
    """
    return FieldException(field=field, message=message, type=type)


# --------------------------------------------------------------------------
# Gates - pure functions, unit-tested without deploying anything
# --------------------------------------------------------------------------

def check_url_policy(url: str, field: str) -> None:
    """Scheme and port policy. The SDK's SSRF client already restricts
    http(s) fetches to globally routable IPs, but it does not restrict
    ports, and `data:` URIs bypass it entirely - so state the policy here."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise reject(
            field,
            f"unsupported URL scheme {parsed.scheme or '(none)'!r}; "
            f"allowed: {', '.join(sorted(ALLOWED_SCHEMES))}",
            "value_error",
        )
    if parsed.scheme != "data" and parsed.port is not None:
        if parsed.port not in ALLOWED_PORTS:
            raise reject(
                field,
                f"port {parsed.port} is not allowed; allowed: "
                f"{', '.join(str(p) for p in sorted(ALLOWED_PORTS))}",
                "value_error",
            )


def check_pixel_budget(width: int, height: int, max_pixels: int, field: str) -> None:
    """The decompression-bomb gate. Runs on header dimensions BEFORE any
    decode: a 12000x12000 PNG is ~400 KB on the wire and ~430 MB decoded,
    which OOM-kills a worker (and every request multiplexed onto it).
    Pillow only warns at this size, so make it a deterministic 422."""
    area = width * height
    if area > max_pixels:
        raise reject(
            field,
            f"image is {width}x{height} ({area:,} pixels); "
            f"maximum is {max_pixels:,} pixels",
            "image_too_large",
        )


def inspect_zip(path: Path, max_entries: int, max_uncompressed_mb: int, field: str) -> dict:
    """Archive gates, all applied BEFORE extraction.

    fal.toolkit's CompressedFile extracts with no limits at all, so an
    archive is only safe if you inspect the central directory first: entry
    count (metadata bombs exhaust inodes even under a byte cap), declared
    uncompressed size, and compression ratio (the zip-bomb signature).
    """
    max_uncompressed = max_uncompressed_mb * 1024 * 1024
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError):
        # Corrupt, truncated, or not a zip at all - all the caller's problem,
        # none of them a 500.
        raise reject(field, "not a readable zip archive", "invalid_archive")

    if len(infos) > max_entries:
        raise reject(
            field,
            f"archive has {len(infos):,} entries; maximum is {max_entries:,}",
            "archive_file_count_exceeds_maximum",
        )
    uncompressed = sum(info.file_size for info in infos)
    if uncompressed > max_uncompressed:
        raise reject(
            field,
            f"archive expands to {uncompressed / 1024 / 1024:.1f} MB; "
            f"maximum is {max_uncompressed_mb} MB",
            "file_too_large",
        )
    compressed = sum(info.compress_size for info in infos) or 1
    ratio = uncompressed / compressed
    # Every member must also land inside the target dir; CPython's
    # extractall() sanitizes member names, but check before trusting it.
    unsafe = [i.filename for i in infos if Path(i.filename).is_absolute() or ".." in Path(i.filename).parts]
    if unsafe:
        raise reject(
            field,
            f"archive contains paths that escape the extraction directory: {unsafe[0]}",
            "invalid_archive",
        )
    return {
        "entries": len(infos),
        "uncompressed_mb": round(uncompressed / 1024 / 1024, 2),
        "compression_ratio": round(ratio, 1),
    }


def read_data_uri(url: str, target_dir: Path, max_mb: int, field: str) -> Path:
    """Materialize a data: URI, sized before it is decoded.

    Two reasons not to hand these to download_file: it logs the URL it
    fetches, so a data: URI dumps its entire base64 payload into your runner
    logs; and the size cap would only apply after decoding. Base64 inflates
    by 4/3, so the encoded length tells you the decoded size for free.
    """
    header, _, payload = url.partition(",")
    if not payload:
        raise reject(field, "malformed data URI", "value_error")
    max_bytes = max_mb * 1024 * 1024
    estimated = len(payload) * 3 // 4 if ";base64" in header else len(payload)
    if estimated > max_bytes:
        raise reject(
            field,
            f"inline payload is ~{estimated / 1024 / 1024:.1f} MB; "
            f"maximum is {max_mb} MB",
            "file_too_large",
        )
    try:
        data = base64.b64decode(payload, validate=True) if ";base64" in header else payload.encode()
    except (binascii.Error, ValueError):
        raise reject(field, "data URI payload is not valid base64", "value_error")

    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"inline-{hashlib.sha256(data).hexdigest()[:16]}"
    path.write_bytes(data)
    return path


def download_guarded(url: str, target: str, max_mb: int, field: str) -> Path:
    """Fetch a user-supplied URL with every failure mapped onto a typed error.

    Left uncaught, a DownloadError is a plain exception - a 500 that tells
    the caller nothing and looks like your bug, not their bad input.
    """
    if urlparse(url).scheme == "data":
        return read_data_uri(url, Path("/data") / target, max_mb, field)
    try:
        return download_file(url, target, filesize_limit=max_mb)
    except DownloadError as exc:
        detail = str(exc)
        if "exceeded" in detail or "over the limit" in detail:
            raise reject(field, f"file exceeds the {max_mb} MB limit", "file_too_large")
        # SSRF refusals arrive here verbatim ("Hostname resolves to a
        # non-routable address", "URL scheme 'ftp' is not allowed", ...).
        raise reject(field, f"could not download: {detail}", "file_download_error")


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

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
        description="Byte cap for downloads. download_file has no default cap.",
    )
    max_pixels: int = Field(
        default=DEFAULT_MAX_PIXELS,
        ge=1024,
        le=DEFAULT_MAX_PIXELS,
        description="Decode budget in pixels, checked before the image is decoded.",
    )
    max_archive_entries: int = Field(
        default=10_000, ge=1, description="Entry-count cap for archives."
    )
    max_uncompressed_mb: int = Field(
        default=512, ge=1, description="Declared-uncompressed-size cap for archives."
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        default="png", description="Unused by validation - here to show a native enum 422."
    )


class Output(BaseModel):
    accepted: bool = Field(description="True when every gate passed.")
    format: str = Field(description="Detected image format.")
    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    bytes: int = Field(description="Downloaded size in bytes.")
    gates: list[str] = Field(description="Gates this request passed, in order.")
    archive: dict | None = Field(default=None, description="Archive stats when one was supplied.")


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

        # 1. URL policy, before a single byte moves.
        check_url_policy(request.image_url, "image_url")
        gates.append("url_policy")

        # 2. Bounded download. SSRF protection comes from the SDK; the cap
        #    is ours (download_file's filesize_limit defaults to None).
        path = download_guarded(
            request.image_url, "cache/bulletproof", request.max_download_mb, "image_url"
        )
        size_bytes = path.stat().st_size
        gates.append("download_cap")

        # 3. Header-only open: dimensions and format without decoding.
        try:
            image = PILImage.open(path)
        except Exception:
            raise reject("image_url", "not a decodable image", "image_load_error")
        if image.format not in ALLOWED_FORMATS:
            raise reject(
                "image_url",
                f"unsupported format {image.format}; allowed: "
                f"{', '.join(sorted(ALLOWED_FORMATS))}",
                "unsupported_image_format",
            )
        gates.append("format_allowlist")

        # 4. The bomb gate - still no pixels allocated.
        check_pixel_budget(image.width, image.height, request.max_pixels, "image_url")
        gates.append("pixel_budget")

        # 5. Only now is it safe to decode.
        try:
            image.load()
        except Exception:
            raise reject("image_url", "image data is truncated or corrupt", "image_load_error")
        gates.append("decode")

        archive_stats = None
        if request.archive_url:
            check_url_policy(request.archive_url, "archive_url")
            archive_path = download_guarded(
                request.archive_url,
                "cache/bulletproof",
                request.max_download_mb,
                "archive_url",
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
            bytes=size_bytes,
            gates=gates,
            archive=archive_stats,
        )
