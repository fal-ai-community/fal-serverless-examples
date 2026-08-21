"""Post-production pipeline - transcode ladder, thumbnails, captions on CPU."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import File, Image, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

MAX_DOWNLOAD_MB = 300
MAX_CAPTIONS_MB = 5
MAX_DURATION_S = 600  # caps a request's encode time; raise when self-hosting
# kills hung ffmpeg before fal's longer request timeout; 2x cap leaves rung headroom
FFMPEG_TIMEOUT_S = 2 * MAX_DURATION_S
CACHE_ROOT = Path("/data/cache/post-production")
DOCKERFILE = Path(__file__).parent / "Dockerfile"
Rung = Literal[2160, 1440, 1080, 720, 480, 360]
X264_CRF = "23"
X264_PRESET = "veryfast"
AAC_BITRATE = "128k"
THUMB_MAX_WIDTH = 1280
# fal-hosted (CI must not hit third-party servers): 1920x1080 H.264, 5.04s, no audio
SAMPLE_VIDEO = (
    "https://storage.googleapis.com/falserverless/example_outputs/pika/"
    "pika_i2v_v22_output.mp4"
)
SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:02,400
Every grand entrance begins somewhere.

2
00:00:02,400 --> 00:00:04,900
Post-production is what makes it ship.
"""


# --- pure helpers (unit-tested without ffmpeg) ------------------------------


def normalize_ladder(
    rungs: Sequence[int], source_height: int
) -> tuple[list[int], list[str]]:
    """Dedupe + sort descending, and drop rungs that would upscale."""
    wanted = sorted(set(rungs), reverse=True)
    if not wanted:
        raise FieldException(
            field="ladder",
            message="ladder needs at least one rung",
            type="value_error",
        )
    encode = [h for h in wanted if h <= source_height]
    skipped = [
        f"{h}p (source is {source_height}p)" for h in wanted if h > source_height
    ]
    if not encode:
        raise FieldException(
            field="ladder",
            message=(
                f"every requested rung is above the {source_height}p source; "
                "this pipeline never upscales"
            ),
            type="nothing_to_encode",
        )
    return encode, skipped


def thumbnail_timestamps(duration_s: float, count: int) -> list[float]:
    """Evenly spaced, excluding t=0 (often a black or logo frame)."""
    return [duration_s * (i + 1) / (count + 1) for i in range(count)]


def select_captions(
    captions_srt: str | None, captions_url: str | None
) -> Literal["inline", "url"] | None:
    if captions_srt and captions_url:
        raise FieldException(
            field="captions_srt",
            message="set captions_srt or captions_url, not both",
            type="value_error",
        )
    if captions_srt:
        return "inline"
    if captions_url:
        return "url"
    return None


@dataclass
class ProbeInfo:
    width: int
    height: int
    duration_s: float
    has_audio: bool
    video_codec: str


def parse_probe(probe: dict) -> ProbeInfo:
    """Gate source: reject stills/audio-only and over-long inputs before encoding."""
    streams = probe.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    duration_raw = probe.get("format", {}).get("duration")
    duration_s = float(duration_raw) if duration_raw else 0.0
    if not video or duration_s <= 0:
        raise FieldException(
            field="video_url",
            message=(
                "no playable video stream (still images and audio files are rejected)"
            ),
            type="invalid_video",
        )
    if duration_s > MAX_DURATION_S:
        raise FieldException(
            field="video_url",
            message=(
                f"source is {duration_s:.0f}s long; this deployment caps sources "
                f"at {MAX_DURATION_S}s (raise MAX_DURATION_S when self-hosting)"
            ),
            type="video_too_long",
        )
    v = video[0]
    return ProbeInfo(
        width=int(v["width"]),
        height=int(v["height"]),
        duration_s=duration_s,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_codec=v.get("codec_name", "unknown"),
    )


def build_rendition_cmd(
    source: Path, srt_path: Path | None, height: int, has_audio: bool, out_path: Path
) -> list[str]:
    """One rung: +faststart for CDN streams; yuv420p so browsers decode 4:4:4/10-bit."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if srt_path is not None:
        cmd += ["-i", str(srt_path)]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    if srt_path is not None:
        cmd += ["-map", "1:s:0"]
    cmd += [
        "-vf",
        f"scale=-2:{height}",
        "-c:v",
        "libx264",
        "-crf",
        X264_CRF,
        "-preset",
        X264_PRESET,
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", AAC_BITRATE]
    if srt_path is not None:
        cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "language=und"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    return cmd


def build_thumbnail_cmd(
    source: Path, timestamp: float, width: int, out_path: Path
) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "2",
        str(out_path),
    ]


def build_vtt_cmd(srt_path: Path, vtt_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(srt_path),
        str(vtt_path),
    ]


# --- process wrappers -------------------------------------------------------


def run_ffmpeg(
    cmd: list[str],
    *,
    error_field: str | None = None,
    error_message: str = "ffmpeg could not process this input",
) -> None:
    """error_field set: user faults become typed 422s; unset: pipeline stays 500s."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S, check=False
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-400:]
        if error_field is not None:
            raise FieldException(
                field=error_field,
                message=f"{error_message}: {tail}",
                type="invalid_media",
            )
        raise RuntimeError(f"{cmd[0]} failed (exit {result.returncode}): {tail}")


def probe_media(path: Path, *, error_field: str | None = None) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-200:]
        if error_field is not None:
            raise FieldException(
                field=error_field,
                message=f"not decodable media: {tail}",
                type="invalid_video",
            )
        raise RuntimeError(f"ffprobe failed on a pipeline artifact: {tail}")
    return json.loads(result.stdout)


def stream_summary(probe: dict) -> dict:
    """Dims + audio flag of a pipeline ARTIFACT - no gating."""
    video = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": any(s.get("codec_type") == "audio" for s in probe["streams"]),
    }


# --- the app ----------------------------------------------------------------


class Input(BaseModel):
    video_url: str = Field(
        description="Source video: paste a URL or upload a file.",
        examples=[SAMPLE_VIDEO],
    )
    ladder: list[Rung] = Field(
        default=[1080, 720, 480],
        description=(
            "Rendition heights to encode. Rungs above the source height are "
            "skipped, never upscaled (the report lists skips)."
        ),
        examples=[[1080, 720, 480]],
    )
    captions_srt: str | None = Field(
        default=None,
        description=(
            "SRT captions as inline text - soft-muxed into every rendition "
            "as a selectable track and returned as a WebVTT sidecar."
        ),
        examples=[SAMPLE_SRT],
    )
    captions_url: str | None = Field(
        default=None,
        description=(
            "URL of an .srt file; alternative to captions_srt (set at most one)."
        ),
    )
    num_thumbnails: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Poster frames, evenly spaced through the clip.",
    )


class Rendition(BaseModel):
    label: str = Field(description='Rung name, e.g. "720p".')
    video: File = Field(
        description=(
            "H.264 MP4 on fal's CDN: faststart, AAC when the source has "
            "audio, captions muxed when provided."
        )
    )
    width: int = Field(description="Actual encoded width (aspect preserved).")
    height: int = Field(description="Actual encoded height.")
    size_bytes: int = Field(description="Rendition file size in bytes.")
    encode_ms: int = Field(
        description="Wall-clock encode time for this rung in milliseconds."
    )


class Output(BaseModel):
    # no sync_mode on purpose: renditions are too large for data-URI transport
    renditions: list[Rendition] = Field(description="Highest rung first.")
    thumbnails: list[Image] = Field(description="JPEG poster frames, evenly spaced.")
    captions_vtt: File | None = Field(
        default=None, description="Normalized WebVTT sidecar for web players."
    )
    timings: dict[str, float] = Field(
        description=(
            "Stage wall-clock seconds: download, encode_wall, thumbnails, upload."
        )
    )
    report: File = Field(
        description=(
            "JSON provenance: probe summary, timings, per-rendition stats, "
            "skipped rungs, toolchain."
        )
    )


class PostProductionPipeline(fal.App, keep_alive=60):
    machine_type = "L"  # multi-rendition x264 wants the vCPUs
    app_auth = "private"
    # image carries deps; requirements unset; requirements.txt mirrors the Dockerfile
    image = ContainerImage.from_dockerfile(str(DOCKERFILE))

    def setup(self) -> None:
        """No weights to load - record the toolchain every report will cite."""
        banner = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        self.ffmpeg_version = banner.stdout.splitlines()[0]

    @fal.endpoint("/")
    def process(self, request: Input) -> Output:
        # validate caption exclusivity before paying for the download
        caption_source = select_captions(request.captions_srt, request.captions_url)

        t0 = time.perf_counter()
        cache_dir = (
            CACHE_ROOT / hashlib.sha256(request.video_url.encode()).hexdigest()[:16]
        )
        cached = None
        if cache_dir.is_dir():
            # download_file stages ".fal_download.tmp.*" here; never a cache hit
            cached = next(
                (
                    p
                    for p in cache_dir.iterdir()
                    if not p.name.startswith(".fal_download.tmp")
                ),
                None,
            )
        if cached is not None:
            source = cached
        else:
            try:
                source = download_file(
                    request.video_url, cache_dir, filesize_limit=MAX_DOWNLOAD_MB
                )
            except DownloadError as exc:
                raise FieldException(
                    field="video_url",
                    message=f"could not download (limit {MAX_DOWNLOAD_MB} MB): {exc}",
                    type="file_download_error",
                ) from exc
        download_s = time.perf_counter() - t0

        info = parse_probe(probe_media(source, error_field="video_url"))
        encode_rungs, skipped = normalize_ladder(request.ladder, info.height)

        with tempfile.TemporaryDirectory(prefix="postprod-") as tmp:
            workdir = Path(tmp)

            # convert captions before encoding so a bad SRT is a fast typed 422
            srt_path = vtt_path = None
            if caption_source == "inline":
                assert request.captions_srt is not None
                srt_path = workdir / "captions.srt"
                srt_path.write_text(request.captions_srt, encoding="utf-8")
            elif caption_source == "url":
                assert request.captions_url is not None
                try:
                    srt_path = download_file(
                        request.captions_url,
                        workdir / "captions-dl",
                        filesize_limit=MAX_CAPTIONS_MB,
                    )
                except DownloadError as exc:
                    raise FieldException(
                        field="captions_url",
                        message=f"could not download captions: {exc}",
                        type="file_download_error",
                    ) from exc
            if srt_path is not None:
                caption_field = (
                    "captions_srt" if caption_source == "inline" else "captions_url"
                )
                vtt_path = workdir / "captions.vtt"
                run_ffmpeg(
                    build_vtt_cmd(srt_path, vtt_path),
                    error_field=caption_field,
                    error_message="not parseable as SRT captions",
                )

            # parallel rungs: threads only babysit ffmpeg pipes; the GIL is no limit
            stem = Path(source).stem
            t1 = time.perf_counter()

            def encode(height: int) -> tuple[int, Path, int]:
                out = workdir / f"{stem}_{height}p.mp4"
                started = time.perf_counter()
                run_ffmpeg(
                    build_rendition_cmd(source, srt_path, height, info.has_audio, out)
                )
                return height, out, int((time.perf_counter() - started) * 1000)

            with ThreadPoolExecutor(max_workers=len(encode_rungs)) as pool:
                encoded = list(pool.map(encode, encode_rungs))
            encode_wall_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            thumb_width = (
                min(THUMB_MAX_WIDTH, info.width) // 2 * 2
            )  # even width: 4:2:0 subsampling
            thumb_paths = []
            for i, ts in enumerate(
                thumbnail_timestamps(info.duration_s, request.num_thumbnails)
            ):
                out = workdir / f"{stem}_thumb_{i:02d}.jpg"
                run_ffmpeg(build_thumbnail_cmd(source, ts, thumb_width, out))
                thumb_paths.append(out)
            thumbs_s = time.perf_counter() - t2

            # from_path never guesses content_type; octet-stream won't play inline
            t3 = time.perf_counter()
            renditions = []
            rendition_report = []
            for height, path, encode_ms in encoded:
                summary = stream_summary(probe_media(path))
                size_bytes = path.stat().st_size
                renditions.append(
                    Rendition(
                        label=f"{height}p",
                        video=File.from_path(path, content_type="video/mp4"),
                        width=summary["width"],
                        height=summary["height"],
                        size_bytes=size_bytes,
                        encode_ms=encode_ms,
                    )
                )
                rendition_report.append(
                    {
                        "label": f"{height}p",
                        **summary,
                        "size_bytes": size_bytes,
                        "encode_ms": encode_ms,
                        "captions_muxed": srt_path is not None,
                    }
                )
            thumbnails = [
                Image.from_bytes(path.read_bytes(), "jpeg", file_name=path.name)
                for path in thumb_paths
            ]
            captions_vtt = (
                File.from_path(vtt_path, content_type="text/vtt") if vtt_path else None
            )
            upload_s = time.perf_counter() - t3

        timings = {
            "download": download_s,
            "encode_wall": encode_wall_s,  # parallel: max(rung), not sum
            "thumbnails": thumbs_s,
            "upload": upload_s,
        }
        report_data = {
            "source": {
                "url": request.video_url,
                "width": info.width,
                "height": info.height,
                "duration_s": round(info.duration_s, 3),
                "video_codec": info.video_codec,
                "has_audio": info.has_audio,
                "cache_hit": cached is not None,
            },
            "ladder": {
                "requested": list(request.ladder),
                "encoded": [f"{h}p" for h in encode_rungs],
                "skipped_no_upscale": skipped,
            },
            "captions": caption_source or "none",
            "renditions": rendition_report,
            "timings_ms": {
                stage: int(seconds * 1000) for stage, seconds in timings.items()
            },
            "toolchain": {"ffmpeg": self.ffmpeg_version, "vcpus": os.cpu_count()},
        }
        report = File.from_bytes(
            json.dumps(report_data, indent=2).encode(),
            content_type="application/json",
            file_name="report.json",
        )
        return Output(
            renditions=renditions,
            thumbnails=thumbnails,
            captions_vtt=captions_vtt,
            timings={stage: round(seconds, 3) for stage, seconds in timings.items()},
            report=report,
        )
