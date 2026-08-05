"""Post-production pipeline - transcode ladder, thumbnails, captions on CPU.

Catalog slot 05.1 (SERV-1282). Every genmedia product rebuilds this layer by
hand: the model's raw MP4 has to become delivery renditions, poster frames,
and captioned files before it can ship. No model here - just ffmpeg on a CPU
machine, in a pinned container, with fal's queue + webhooks as the async
layer (see client.py / webhook_receiver.py).
"""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fal
from fal.container import ContainerImage
from fal.exceptions import FieldException
from fal.toolkit import File, Image, Video, download_file
from fal.toolkit.utils.download_utils import DownloadError
from pydantic import BaseModel, Field

MAX_DOWNLOAD_MB = 300
MAX_CAPTIONS_MB = 5
MAX_DURATION_S = 600  # caps a request's encode time; raise when self-hosting
CACHE_ROOT = Path("/data/cache/post-production")
DOCKERFILE = Path(__file__).parent / "Dockerfile"
Rung = Literal[2160, 1440, 1080, 720, 480, 360]
X264_CRF = "23"
X264_PRESET = "veryfast"
AAC_BITRATE = "128k"
THUMB_MAX_WIDTH = 1280
# Verified 1920x1080 H.264, 5.04s, 2.4MB, no audio - fal-hosted so CI never
# depends on a third-party server (see the audio-path sample in test_app.py).
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


# --- pure helpers (unit-tested without ffmpeg; see test_app.py) ------------

def normalize_ladder(rungs: list[int], source_height: int) -> tuple[list[int], list[str]]:
    """Dedupe + sort descending, and drop rungs that would upscale."""
    wanted = sorted(set(rungs), reverse=True)
    if not wanted:
        raise FieldException(field="ladder", message="ladder needs at least one rung")
    encode = [h for h in wanted if h <= source_height]
    skipped = [f"{h}p (source is {source_height}p)" for h in wanted if h > source_height]
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


def select_captions(captions_srt: str | None, captions_url: str | None) -> str | None:
    if captions_srt and captions_url:
        raise FieldException(
            field="captions_srt",
            message="set captions_srt or captions_url, not both",
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


def parse_probe(probe: dict, field: str = "video_url") -> ProbeInfo:
    """Gate the SOURCE: reject stills/audio-only (no playable duration) and
    absurd durations before any encode time is spent."""
    streams = probe.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    duration_raw = probe.get("format", {}).get("duration")
    duration_s = float(duration_raw) if duration_raw else 0.0
    if not video or duration_s <= 0:
        raise FieldException(
            field=field,
            message="no playable video stream (still images and audio files are rejected)",
            type="invalid_video",
        )
    if duration_s > MAX_DURATION_S:
        raise FieldException(
            field=field,
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
    """One ladder rung: H.264 + optional AAC + optional mov_text captions.

    +faststart moves the moov atom to the front so the MP4 streams from a
    CDN without a full download; yuv420p keeps browser players happy when
    genmedia models emit 4:4:4 or 10-bit sources.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if srt_path is not None:
        cmd += ["-i", str(srt_path)]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    if srt_path is not None:
        cmd += ["-map", "1:s:0"]
    cmd += [
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-crf", X264_CRF, "-preset", X264_PRESET,
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", AAC_BITRATE]
    if srt_path is not None:
        cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "language=und"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    return cmd


def build_thumbnail_cmd(source: Path, timestamp: float, width: int, out_path: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{timestamp:.3f}", "-i", str(source),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "2",
        str(out_path),
    ]


def build_vtt_cmd(srt_path: Path, vtt_path: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(srt_path), str(vtt_path),
    ]


# --- process wrappers -------------------------------------------------------

def run_ffmpeg(cmd: list[str], *, error_field: str | None = None,
               error_message: str = "ffmpeg could not process this input") -> None:
    """User-input failures become typed 422s (error_field set); pipeline
    failures on already-validated media stay 500s, so the queue marks the
    job failed rather than blaming the caller."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-400:]
        if error_field is not None:
            raise FieldException(
                field=error_field, message=f"{error_message}: {tail}", type="invalid_media"
            )
        raise RuntimeError(f"{cmd[0]} failed (exit {result.returncode}): {tail}")


def probe_media(path: Path, *, error_field: str | None = None) -> dict:
    result = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-200:]
        if error_field is not None:
            raise FieldException(
                field=error_field, message=f"not decodable media: {tail}",
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
        default_factory=lambda: [1080, 720, 480],
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
        description="URL of an .srt file; alternative to captions_srt (set at most one).",
    )
    thumbnail_count: int = Field(
        default=3, ge=1, le=10,
        description="Poster frames, evenly spaced through the clip.",
    )


class Rendition(BaseModel):
    label: str = Field(description='Rung name, e.g. "720p".')
    video: Video = Field(
        description="H.264 MP4 on fal's CDN: faststart, AAC when the source has audio, captions muxed when provided."
    )
    width: int = Field(description="Actual encoded width (aspect preserved).")
    height: int = Field(description="Actual encoded height.")
    size_bytes: int
    encode_ms: int


class Output(BaseModel):
    renditions: list[Rendition] = Field(description="Highest rung first.")
    thumbnails: list[Image] = Field(description="JPEG poster frames, evenly spaced.")
    captions_vtt: File | None = Field(
        default=None, description="Normalized WebVTT sidecar for web players."
    )
    report: File = Field(
        description="JSON provenance: probe summary, timings, per-rendition stats, skipped rungs, toolchain."
    )


class PostProductionPipeline(fal.App, keep_alive=60):
    machine_type = "L"  # multi-rendition x264 wants the vCPUs; see README "Cost & tuning"
    app_auth = "private"
    # The image carries everything (ffmpeg + fal); requirements stays unset -
    # requirements.txt mirrors the Dockerfile's pip installs instead.
    image = ContainerImage.from_dockerfile(str(DOCKERFILE))

    def setup(self) -> None:
        """No weights to load - record the toolchain every report will cite."""
        banner = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        self.ffmpeg_version = banner.stdout.splitlines()[0]

    @fal.endpoint("/")
    def process(self, request: Input) -> Output:
        # SOURCE - bounded, SSRF-guarded download (both built into
        # download_file), cached on /data keyed by URL hash: a repeat request
        # skips the transfer across workers, deploys, and CI runs.
        t0 = time.monotonic()
        cache_dir = CACHE_ROOT / hashlib.sha256(request.video_url.encode()).hexdigest()[:16]
        cached = next(cache_dir.glob("*"), None) if cache_dir.is_dir() else None
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
                )
        download_ms = int((time.monotonic() - t0) * 1000)

        # GATE - probe once, reject stills/marathons, learn dims + audio.
        info = parse_probe(probe_media(source, error_field="video_url"))
        encode_rungs, skipped = normalize_ladder(request.ladder, info.height)

        with tempfile.TemporaryDirectory(prefix="postprod-") as tmp:
            workdir = Path(tmp)

            # CAPTIONS - validate cheaply (SRT -> WebVTT) BEFORE spending
            # encode time, so a bad caption file is a fast typed 422.
            caption_source = select_captions(request.captions_srt, request.captions_url)
            caption_field = "captions_srt" if caption_source == "inline" else "captions_url"
            srt_path = vtt_path = None
            if caption_source == "inline":
                srt_path = workdir / "captions.srt"
                srt_path.write_text(request.captions_srt, encoding="utf-8")
            elif caption_source == "url":
                try:
                    srt_path = download_file(
                        request.captions_url, workdir / "captions-dl",
                        filesize_limit=MAX_CAPTIONS_MB,
                    )
                except DownloadError as exc:
                    raise FieldException(
                        field="captions_url",
                        message=f"could not download captions: {exc}",
                        type="file_download_error",
                    )
            if srt_path is not None:
                vtt_path = workdir / "captions.vtt"
                run_ffmpeg(
                    build_vtt_cmd(srt_path, vtt_path),
                    error_field=caption_field,
                    error_message="not parseable as SRT captions",
                )

            # LADDER - one ffmpeg process per rung, in parallel. These are
            # subprocesses, so Python threads only babysit pipes; the real
            # question is rung count vs vCPUs - see README "Cost & tuning".
            stem = Path(source).stem
            t1 = time.monotonic()

            def encode(height: int) -> tuple[int, Path, int]:
                out = workdir / f"{stem}_{height}p.mp4"
                started = time.monotonic()
                run_ffmpeg(build_rendition_cmd(source, srt_path, height, info.has_audio, out))
                return height, out, int((time.monotonic() - started) * 1000)

            with ThreadPoolExecutor(max_workers=len(encode_rungs)) as pool:
                encoded = list(pool.map(encode, encode_rungs))
            encode_wall_ms = int((time.monotonic() - t1) * 1000)

            # THUMBNAILS - fast keyframe seeks on the local source.
            t2 = time.monotonic()
            thumb_width = min(THUMB_MAX_WIDTH, info.width) // 2 * 2
            thumb_paths = []
            for i, ts in enumerate(thumbnail_timestamps(info.duration_s, request.thumbnail_count)):
                out = workdir / f"{stem}_thumb_{i:02d}.jpg"
                run_ffmpeg(build_thumbnail_cmd(source, ts, thumb_width, out))
                thumb_paths.append(out)
            thumbs_ms = int((time.monotonic() - t2) * 1000)

            # UPLOADS - fal CDN via toolkit types. content_type is explicit
            # everywhere: from_path never guesses, and an octet-stream video
            # won't play inline in a browser or the playground.
            t3 = time.monotonic()
            renditions = []
            rendition_report = []
            for height, path, encode_ms in encoded:
                summary = stream_summary(probe_media(path))
                renditions.append(
                    Rendition(
                        label=f"{height}p",
                        video=Video.from_path(path, content_type="video/mp4"),
                        width=summary["width"],
                        height=summary["height"],
                        size_bytes=path.stat().st_size,
                        encode_ms=encode_ms,
                    )
                )
                rendition_report.append(
                    {
                        "label": f"{height}p",
                        **summary,
                        "size_bytes": path.stat().st_size,
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
            upload_ms = int((time.monotonic() - t3) * 1000)

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
                "download": download_ms,  # ~0 on a cache hit
                "encode_wall": encode_wall_ms,  # parallel: max(rung), not sum
                "thumbnails": thumbs_ms,
                "upload": upload_ms,
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
            report=report,
        )
