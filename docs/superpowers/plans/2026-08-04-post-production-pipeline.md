# 05.1 Post-Production Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build catalog slot 05.1 ([SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282)) — an ffmpeg transcode-ladder/thumbnails/captions pipeline on a fal CPU machine, with a verified queue + webhook client story.

**Architecture:** One `fal.App` (custom ffmpeg container, `machine_type="L"`) with a single `/` endpoint doing download → probe gate → parallel ffmpeg ladder → caption mux → thumbnails → CDN uploads per request; fal's platform queue (`queue.fal.run`) is the async layer. A raw-httpx `client.py` and an Ed25519-verifying `webhook_receiver.py` are the product-backend reference. Spec: `docs/superpowers/specs/2026-08-04-post-production-pipeline-design.md`.

**Tech Stack:** Python 3.11, fal==1.78.2, ffmpeg (Debian bookworm 5.1.x via digest-pinned `python:3.11-slim`), httpx, cryptography (client-side only), pytest + repo harness (`harness/check.py`).

## Global Constraints

- Example folder: `05-media-processing/post-production-pipeline/` — never anywhere else.
- Pin `fal==1.78.2` everywhere (Dockerfile, requirements.txt), matching `harness/requirements.txt`.
- Container app: the Dockerfile carries all deps; do NOT set `App.requirements`; `requirements.txt` mirrors the Dockerfile's pip installs (static_check skips the mirror check when `App.requirements` is empty, but the convention holds).
- `app_auth = "private"` must be set explicitly (static_check enforces).
- Every required Input field carries `examples=[...]` — the harness builds its smoke payload from them (`harness/check.py::sample_input`). Optional fields with examples ARE included in the smoke payload; optional fields without examples are NOT.
- The smoke request is synchronous with a 120 s timeout — the sample-clip pipeline must finish well inside it.
- No performance/cost claims in any docs that aren't backed by a reproducible run (2026-07-21 descope; the app's own `report` measures timings instead).
- Python for any `fal deploy`/harness run must be 3.11 (container base is python:3.11-slim; cloudpickle is CPython-minor-specific).
- README follows `templates/example/README.md` §4-A section order exactly.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Verified platform facts (research 2026-08-04; each task repeats what it needs)

- Sample clips (fal-controlled GCS, both verified reachable): primary `https://storage.googleapis.com/falserverless/example_outputs/pika/pika_i2v_v22_output.mp4` (1920×1080 H.264, 5.04 s, 2.4 MB, **no audio**); audio-path `https://storage.googleapis.com/falserverless/example_outputs/bytedance/seedance_2/output.mp4` (1280×720 H.264 + AAC, 12.04 s, 15.4 MB).
- `File.from_path(path, content_type=None, repository="fal_v3", multipart=None, ...)` — **no content-type guessing** (`application/octet-stream` unless passed) and **no file_name param** (name = `Path(path).name`). `Video` and `Image` inherit it. `Image.from_bytes(data, format, size=None, file_name=None, ...)` forces `content_type=f"image/{format}"`.
- `download_file(url, target_dir, *, force=False, request_headers=None, filesize_limit=None)` — `filesize_limit` is **MiB**; relative `target_dir` resolves under `/data`; raises `DownloadError`; SSRF-guarded (blocks private/link-local/metadata IPs).
- `FieldException(field, message, status_code=422, type="value_error", billable_units=0)` from `fal.exceptions` → `{"detail": [{"loc": ["body", field], "msg": message, "type": type}]}`.
- Queue (private apps use the SAME URLs as public models): submit `POST https://queue.fal.run/{owner}/{app}` (root endpoint → **no trailing slash, no path suffix**; webhook via URL-encoded `?fal_webhook=`); status `GET .../requests/{id}/status?logs=1`; result `GET .../requests/{id}`; auth `Authorization: Key <FAL_KEY>` where FAL_KEY is `key_id:key_secret`. Status values: only `IN_QUEUE`/`IN_PROGRESS`/`COMPLETED`; failures = `COMPLETED` + `error`/`error_type` in status, result URL then returns non-2xx with the error body.
- Webhooks: POST with headers `X-Fal-Webhook-Request-Id`, `X-Fal-Webhook-User-Id`, `X-Fal-Webhook-Timestamp` (unix seconds), `X-Fal-Webhook-Signature` (hex Ed25519). Signed message = UTF-8 of `"{request_id}\n{user_id}\n{timestamp}\n{sha256(raw_body).hexdigest()}"` — hash the **raw received bytes**, never re-serialize. Verify against any key from `https://rest.fal.ai/.well-known/jwks.json` (OKP/Ed25519, `x` is base64url **with possible padding**), ±300 s timestamp leeway, JWKS cacheable ≤24 h. Body: `{request_id, gateway_request_id, status: "OK"|"ERROR", error, payload, payload_error?}`. Deliveries retry until a 2xx → dedupe by `request_id`.

---

### Task 1: Scaffold — folder, Dockerfile, requirements.txt, local env

**Files:**
- Create: `05-media-processing/post-production-pipeline/Dockerfile`
- Create: `05-media-processing/post-production-pipeline/requirements.txt`

**Interfaces:**
- Produces: a `.venv/` (repo root, git-ignored) with Python 3.11 + `harness/requirements.txt` installed — later tasks run `.venv/bin/python -m pytest` and `.venv/bin/python harness/check.py`.
- Produces: `Dockerfile` consumed by Task 2's `ContainerImage.from_dockerfile(str(Path(__file__).parent / "Dockerfile"))`.

- [ ] **Step 1: Create the venv and verify the toolchain**

```bash
cd <repo-root>
which python3.11 || sudo apt-get install -y python3.11 python3.11-venv  # or: uv venv --python 3.11 .venv
python3.11 -m venv .venv
.venv/bin/pip install -r harness/requirements.txt
.venv/bin/python -c "import fal, httpx, pytest; print('deps ok')"
which ffmpeg ffprobe || sudo apt-get install -y ffmpeg
```

Expected: `deps ok` and both ffmpeg binaries on PATH (needed for local unit runs only; GitHub `ubuntu-latest` preinstalls ffmpeg). Confirm `.venv/` is git-ignored (`git check-ignore .venv` prints `.venv`); if not, add it to `.gitignore` in this commit.

- [ ] **Step 2: Create the example folder and Dockerfile**

Create `05-media-processing/post-production-pipeline/Dockerfile`:

```dockerfile
# The pipeline's environment. fal builds this server-side - no local Docker,
# no registry push. Digest-pinned for the same reason as 01-environments-
# containers/same-model-two-ways: fal's build cache keys on the CONTENT of
# this file, so a digest makes every base-image change a visible edit.
# (python:3.11-slim, 2026-07-25.)
FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

# python3.11 must match the Python that runs `fal deploy` (apps ship as
# cloudpickled bytecode, which is CPython-minor-specific). curl is fal's
# installer dependency.
#
# ffmpeg IS this example: apt pins it to Debian bookworm's 5.1.x line, so
# every worker encodes with the exact same toolchain. The app echoes
# `ffmpeg -version` into every report for provenance.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# fal last, so it resolves against what is already installed.
RUN pip install --no-cache-dir fal==1.78.2

# No ENTRYPOINT/CMD: fal supplies its own.
```

- [ ] **Step 3: Create requirements.txt**

Create `05-media-processing/post-production-pipeline/requirements.txt`:

```
# Mirror of the Dockerfile's pip installs - the image is authoritative for
# this container example (App.requirements stays unset; repo convention).
fal==1.78.2
```

- [ ] **Step 4: Commit**

```bash
git add 05-media-processing/post-production-pipeline/ .gitignore
git commit -m "feat(05-media-processing): scaffold post-production-pipeline (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(static_check will not pass for this folder until app.py lands in Task 2 — that is fine; CI runs on PR, and the PR ships with the full example.)

---

### Task 2: app.py + test_app.py — the pipeline, unit-test-first

**Files:**
- Create: `05-media-processing/post-production-pipeline/app.py`
- Create: `05-media-processing/post-production-pipeline/test_app.py`

**Interfaces:**
- Consumes: `Dockerfile` from Task 1 (same directory).
- Produces (used by Tasks 3/5/6/7): module constants `SAMPLE_VIDEO`, `SAMPLE_SRT`; pure helpers `normalize_ladder(rungs, source_height) -> tuple[list[int], list[str]]`, `thumbnail_timestamps(duration_s, count) -> list[float]`, `select_captions(captions_srt, captions_url) -> str | None` (returns `"inline"`/`"url"`/`None`), `parse_probe(probe: dict, field="video_url") -> ProbeInfo(width, height, duration_s, has_audio, video_codec)`, `build_rendition_cmd(source, srt_path, height, has_audio, out_path) -> list[str]`, `build_thumbnail_cmd(source, timestamp, width, out_path) -> list[str]`, `build_vtt_cmd(srt_path, vtt_path) -> list[str]`; app class `PostProductionPipeline` with endpoint `/` taking `Input(video_url, ladder, captions_srt, captions_url, thumbnail_count)` returning `Output(renditions: list[Rendition], thumbnails: list[Image], captions_vtt: File | None, report: File)` where `Rendition(label, video: Video, width, height, size_bytes, encode_ms)`.

- [ ] **Step 1: Write the unit-test section of test_app.py (failing first)**

Create `05-media-processing/post-production-pipeline/test_app.py`:

```python
"""Tests for the post-production pipeline.

Three kinds, all run by CI:
  * unit tests over the pure helpers - free, no deploy, no ffmpeg binary
    needed (command builders return argv lists; nothing is executed);
  * live tests against a real ephemeral deploy (harness/check.py);
  * webhook-receiver verification tests (skipped unless `cryptography`
    is installed - it is a client-side dep, not an app dep).
"""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fal.exceptions import FieldException

_spec = importlib.util.spec_from_file_location(
    "postprod_app", Path(__file__).parent / "app.py"
)
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)

# Verified fal-hosted samples (research 2026-08-04): pika = 1920x1080 h264,
# 5.04s, no audio; seedance = 1280x720 h264 + AAC, 12.04s.
SAMPLE = app_module.SAMPLE_VIDEO
AUDIO_SAMPLE = (
    "https://storage.googleapis.com/falserverless/example_outputs/"
    "bytedance/seedance_2/output.mp4"
)
STILL_IMAGE = "https://storage.googleapis.com/falserverless/UNO/crystal_ball.png"


# --- unit tests: pure helpers (no deploy, no network, no ffmpeg) -----------

def test_ladder_dedupes_sorts_and_skips_upscales():
    encode, skipped = app_module.normalize_ladder([480, 1080, 480, 2160], 1080)
    assert encode == [1080, 480]
    assert skipped == ["2160p (source is 1080p)"]


def test_ladder_all_upscale_is_a_typed_422():
    with pytest.raises(FieldException) as excinfo:
        app_module.normalize_ladder([2160, 1440], 720)
    assert excinfo.value.field == "ladder"
    assert excinfo.value.type == "nothing_to_encode"
    with pytest.raises(FieldException):
        app_module.normalize_ladder([], 1080)


def test_thumbnail_timestamps_dodge_the_edges():
    assert app_module.thumbnail_timestamps(10.0, 3) == [2.5, 5.0, 7.5]
    assert app_module.thumbnail_timestamps(10.0, 1) == [5.0]


def test_select_captions_accepts_at_most_one_source():
    assert app_module.select_captions(None, None) is None
    assert app_module.select_captions("1\n...", None) == "inline"
    assert app_module.select_captions(None, "https://x/y.srt") == "url"
    with pytest.raises(FieldException) as excinfo:
        app_module.select_captions("1\n...", "https://x/y.srt")
    assert excinfo.value.field == "captions_srt"


def _probe(streams, duration="5.0"):
    fmt = {"duration": duration} if duration is not None else {}
    return {"streams": streams, "format": fmt}


VIDEO_STREAM = {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080}


def test_parse_probe_reads_dims_duration_and_audio():
    info = app_module.parse_probe(
        _probe([VIDEO_STREAM, {"codec_type": "audio", "codec_name": "aac"}])
    )
    assert (info.width, info.height, info.has_audio) == (1920, 1080, True)
    assert app_module.parse_probe(_probe([VIDEO_STREAM])).has_audio is False


def test_parse_probe_rejects_stills_and_marathons():
    # A still image probes as a video stream with no duration.
    with pytest.raises(FieldException) as excinfo:
        app_module.parse_probe(_probe([VIDEO_STREAM], duration=None))
    assert excinfo.value.type == "invalid_video"
    with pytest.raises(FieldException) as excinfo:
        app_module.parse_probe(_probe([VIDEO_STREAM], duration="4000"))
    assert excinfo.value.type == "video_too_long"
    with pytest.raises(FieldException):
        app_module.parse_probe(_probe([{"codec_type": "audio"}]))


def test_rendition_cmd_toggles_audio_and_captions():
    full = app_module.build_rendition_cmd(
        Path("in.mp4"), Path("subs.srt"), 720, True, Path("out.mp4")
    )
    assert full[full.index("-vf") + 1] == "scale=-2:720"
    for flag in ("mov_text", "+faststart", "yuv420p", "aac"):
        assert any(flag in arg for arg in full), f"missing {flag}"

    bare = app_module.build_rendition_cmd(
        Path("in.mp4"), None, 480, False, Path("out.mp4")
    )
    joined = " ".join(bare)
    assert "aac" not in joined and "mov_text" not in joined and "subs" not in joined
    assert "+faststart" in joined  # streaming-readiness is unconditional
```

- [ ] **Step 2: Run the unit tests to verify they fail**

```bash
.venv/bin/python -m pytest 05-media-processing/post-production-pipeline/test_app.py -q
```

Expected: FAIL at import/collection (`FileNotFoundError` / attribute errors — app.py doesn't exist yet).

- [ ] **Step 3: Write app.py**

Create `05-media-processing/post-production-pipeline/app.py`:

```python
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
```

- [ ] **Step 4: Run unit tests to verify they pass**

```bash
.venv/bin/python -m pytest 05-media-processing/post-production-pipeline/test_app.py -q
```

Expected: unit tests PASS; no live tests exist yet.

- [ ] **Step 5: Append the live-test section to test_app.py**

Append to `05-media-processing/post-production-pipeline/test_app.py`:

```python
# --- live tests: against a real ephemeral deploy (harness/check.py) --------

def assert_typed_422(exc_info, field: str, type_: str):
    response = exc_info.value.response
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", field]
    assert detail["type"] == type_
    return detail


def fetch(url: str) -> httpx.Response:
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response


def test_full_pipeline_ladder_captions_thumbnails(app_client):
    response = app_client.run(
        "/",
        {
            "video_url": SAMPLE,
            "ladder": [1080, 720, 480],
            "captions_srt": app_module.SAMPLE_SRT,
            "thumbnail_count": 2,
        },
    )
    renditions = response["renditions"]
    assert [r["label"] for r in renditions] == ["1080p", "720p", "480p"]
    assert [r["height"] for r in renditions] == [1080, 720, 480]
    for r in renditions:
        assert r["size_bytes"] > 0 and r["encode_ms"] > 0
        assert r["video"]["content_type"] == "video/mp4"

    # +faststart is the point: moov must come before mdat, or the CDN URL
    # can't stream. Verify on the smallest rendition's actual bytes.
    data = fetch(renditions[-1]["video"]["url"]).content
    moov, mdat = data.find(b"moov"), data.find(b"mdat")
    assert moov != -1, "no moov atom - not a valid MP4?"
    assert mdat == -1 or moov < mdat, "moov after mdat: faststart didn't apply"

    assert len(response["thumbnails"]) == 2
    for thumb in response["thumbnails"]:
        assert thumb["content_type"] == "image/jpeg"

    assert response["captions_vtt"] is not None
    assert "WEBVTT" in fetch(response["captions_vtt"]["url"]).text

    report = fetch(response["report"]["url"]).json()
    assert report["source"]["has_audio"] is False
    assert report["ladder"]["skipped_no_upscale"] == []
    assert report["renditions"][0]["captions_muxed"] is True
    assert report["toolchain"]["ffmpeg"].startswith("ffmpeg version")


def test_audio_source_transcodes_and_never_upscales(app_client):
    """720p source with AAC audio: 2160 must be skipped (never upscale),
    and the audio must survive into the rendition."""
    response = app_client.run(
        "/",
        {"video_url": AUDIO_SAMPLE, "ladder": [2160, 360], "thumbnail_count": 1},
    )
    assert [r["label"] for r in response["renditions"]] == ["360p"]
    report = fetch(response["report"]["url"]).json()
    assert report["source"]["has_audio"] is True
    assert report["ladder"]["skipped_no_upscale"] == ["2160p (source is 720p)"]
    assert report["renditions"][0]["has_audio"] is True
    assert response["captions_vtt"] is None


def test_still_image_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run("/", {"video_url": STILL_IMAGE})
    assert_typed_422(excinfo, "video_url", "invalid_video")


def test_both_caption_fields_is_refused(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {
                "video_url": SAMPLE,
                "captions_srt": app_module.SAMPLE_SRT,
                "captions_url": "https://example.com/captions.srt",
            },
        )
    assert_typed_422(excinfo, "captions_srt", "value_error")


def test_garbage_captions_are_refused_before_encoding(app_client):
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        app_client.run(
            "/",
            {"video_url": SAMPLE, "captions_srt": "this is not a subtitle file"},
        )
    assert_typed_422(excinfo, "captions_srt", "invalid_media")
```

(Contingency, verify in Task 6: if ffmpeg's SRT demuxer turns out to accept that garbage string leniently, replace the payload with `"\x00\x01 not srt"` — binary junk it cannot parse — and keep the same assertion.)

- [ ] **Step 6: Verify static checks and full collection pass**

```bash
.venv/bin/python harness/static_check.py
.venv/bin/python -m pytest 05-media-processing/post-production-pipeline/test_app.py -q
```

Expected: `ok   05-media-processing/post-production-pipeline` in static output (other examples also `ok`); pytest shows unit tests passing and live tests skipping (`no live app under test`).

- [ ] **Step 7: Commit**

```bash
git add 05-media-processing/post-production-pipeline/app.py 05-media-processing/post-production-pipeline/test_app.py
git commit -m "feat(05-media-processing): post-production pipeline app + tests (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: client.py — the verified queue reference for private apps

**Files:**
- Create: `05-media-processing/post-production-pipeline/client.py`

**Interfaces:**
- Consumes: a deployed app id like `<username>/post-production-pipeline` (Task 7) and the app's `Input` schema from Task 2.
- Produces: CLI used in Task 7's validation and quoted in the README.

- [ ] **Step 1: Write client.py**

Create `05-media-processing/post-production-pipeline/client.py`:

```python
"""Queue client for the deployed pipeline - the raw HTTP a backend really sends.

Deliberately httpx-only (no fal_client) so every URL and header for calling a
PRIVATE serverless app through fal's queue is visible and verifiable:

    submit  POST https://queue.fal.run/{owner}/{app}[?fal_webhook=...]
    status  GET  https://queue.fal.run/{owner}/{app}/requests/{id}/status?logs=1
    result  GET  https://queue.fal.run/{owner}/{app}/requests/{id}

Statuses are IN_QUEUE -> IN_PROGRESS -> COMPLETED (there is no FAILED status:
failures arrive as COMPLETED with an `error`, and the result URL then returns
a non-2xx with the error body).

Usage:
    export FAL_KEY=<key_id>:<key_secret>
    python client.py --app <username>/post-production-pipeline \
        [--video-url URL] [--srt-file captions.srt] [--ladder 1080,720,480] \
        [--webhook https://your-host/fal-webhook]

Dependency: pip install httpx==0.28.1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

QUEUE = "https://queue.fal.run"
DEFAULT_VIDEO = (
    "https://storage.googleapis.com/falserverless/example_outputs/pika/"
    "pika_i2v_v22_output.mp4"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="owner/app-name of your deploy")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO)
    parser.add_argument("--ladder", default="1080,720,480", help="comma-separated rung heights")
    parser.add_argument("--srt-file", default=None, help="local .srt sent as inline captions")
    parser.add_argument("--webhook", default=None, help="public URL fal POSTs the result to")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        sys.exit("set FAL_KEY=<key_id>:<key_secret> (dashboard -> Keys)")
    headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}

    payload = {
        "video_url": args.video_url,
        "ladder": [int(h) for h in args.ladder.split(",")],
    }
    if args.srt_file:
        payload["captions_srt"] = Path(args.srt_file).read_text(encoding="utf-8")

    # SUBMIT - the queue answers immediately with request handles; the
    # optional fal_webhook parameter must be URL-encoded.
    submit_url = f"{QUEUE}/{args.app}"
    if args.webhook:
        submit_url += "?" + urlencode({"fal_webhook": args.webhook})
    submitted = httpx.post(submit_url, json=payload, headers=headers, timeout=30)
    submitted.raise_for_status()
    handles = submitted.json()
    request_id = handles["request_id"]
    print(f"queued: request_id={request_id}")
    if args.webhook:
        print(f"fal will POST the result to {args.webhook} (signed; see webhook_receiver.py)")
        print("polling too, as the fallback a real backend keeps:")

    # POLL - ?logs=1 streams the app's own log lines into the status body.
    status_url = f"{QUEUE}/{args.app}/requests/{request_id}/status"
    result_url = f"{QUEUE}/{args.app}/requests/{request_id}"
    seen_logs = 0
    last_status = ""
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        status = httpx.get(
            status_url, params={"logs": "1"}, headers=headers, timeout=30
        )
        status.raise_for_status()
        body = status.json()
        for line in (body.get("logs") or [])[seen_logs:]:
            print(f"  [app] {line.get('message', line)}")
        seen_logs = len(body.get("logs") or [])
        if body["status"] != last_status:
            last_status = body["status"]
            position = body.get("queue_position")
            print(f"  status: {last_status}"
                  + (f" (queue position {position})" if position is not None else ""))
        if last_status == "COMPLETED":
            if body.get("error"):
                print(f"  completed with error: {body['error']} ({body.get('error_type')})")
            break
        time.sleep(args.poll_interval)
    else:
        sys.exit(f"timed out after {args.timeout:.0f}s; check {status_url}")

    # RESULT - a failed request returns non-2xx here, with the error body.
    result = httpx.get(result_url, headers=headers, timeout=60)
    if not result.is_success:
        print(f"FAILED: HTTP {result.status_code}")
        print(result.text[:500])
        return 1
    print(json.dumps(result.json(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke the argument parsing (no network)**

```bash
.venv/bin/python 05-media-processing/post-production-pipeline/client.py --help
FAL_KEY= .venv/bin/python 05-media-processing/post-production-pipeline/client.py --app x/y 2>&1 | head -1
```

Expected: help text renders; second command exits with the `set FAL_KEY=...` message.

- [ ] **Step 3: Commit**

```bash
git add 05-media-processing/post-production-pipeline/client.py
git commit -m "feat(05-media-processing): queue client for post-production pipeline (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: webhook_receiver.py — Ed25519-verified completion webhooks

**Files:**
- Create: `05-media-processing/post-production-pipeline/webhook_receiver.py`
- Modify: `05-media-processing/post-production-pipeline/test_app.py` (append receiver unit tests)

**Interfaces:**
- Consumes: fal's verified signing scheme (below) — no repo dependencies.
- Produces: `verify(headers: dict[str, str], body: bytes, keys: list) -> tuple[bool, str]` (headers lowercase-keyed), used by its own `do_POST` and by the unit tests.

Verified signing scheme (fal production signer, 2026-08-04): headers `X-Fal-Webhook-Request-Id`, `X-Fal-Webhook-User-Id`, `X-Fal-Webhook-Timestamp` (unix seconds), `X-Fal-Webhook-Signature` (128 hex chars). Message = UTF-8 of `request_id + "\n" + user_id + "\n" + timestamp + "\n" + sha256(raw_body).hexdigest()`. Keys: JWKS at `https://rest.fal.ai/.well-known/jwks.json`, each key's `x` is base64url (possibly padded) of a 32-byte Ed25519 public key; accept if ANY key verifies; ±300 s timestamp leeway; deliveries repeat until a 2xx → dedupe by request_id.

- [ ] **Step 1: Append failing receiver unit tests to test_app.py**

Append to `05-media-processing/post-production-pipeline/test_app.py`:

```python
# --- webhook receiver: signature verification (client-side dep) ------------
# The receiver is product-backend reference code, not part of the app image;
# its `cryptography` dep is not in requirements.txt, so skip when absent.

crypto = pytest.importorskip("cryptography")


def _receiver_module():
    spec = importlib.util.spec_from_file_location(
        "postprod_receiver", Path(__file__).parent / "webhook_receiver.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signed_delivery(timestamp: int | None = None):
    """Forge a delivery exactly the way fal's dispatcher signs one."""
    import hashlib as _hashlib
    import time as _time

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    body = json.dumps(
        {"request_id": "req-1", "gateway_request_id": "req-1",
         "status": "OK", "error": None, "payload": {"renditions": []}},
        sort_keys=True,
    ).encode()
    ts = str(timestamp if timestamp is not None else int(_time.time()))
    message = "\n".join(
        ["req-1", "user-1", ts, _hashlib.sha256(body).hexdigest()]
    ).encode()
    headers = {
        "x-fal-webhook-request-id": "req-1",
        "x-fal-webhook-user-id": "user-1",
        "x-fal-webhook-timestamp": ts,
        "x-fal-webhook-signature": private.sign(message).hex(),
    }
    return headers, body, [private.public_key()]


def test_receiver_accepts_a_correctly_signed_delivery():
    receiver = _receiver_module()
    headers, body, keys = _signed_delivery()
    ok, reason = receiver.verify(headers, body, keys)
    assert ok, reason


def test_receiver_rejects_tampered_bodies_and_stale_timestamps():
    receiver = _receiver_module()

    headers, body, keys = _signed_delivery()
    ok, reason = receiver.verify(headers, body + b" ", keys)
    assert not ok and "signature" in reason

    stale_headers, stale_body, stale_keys = _signed_delivery(timestamp=1_000_000)
    ok, reason = receiver.verify(stale_headers, stale_body, stale_keys)
    assert not ok and "timestamp" in reason

    missing = dict(headers)
    del missing["x-fal-webhook-signature"]
    ok, reason = receiver.verify(missing, body, keys)
    assert not ok and "header" in reason
```

- [ ] **Step 2: Install cryptography locally and run to verify the new tests fail**

```bash
.venv/bin/pip install cryptography
.venv/bin/pip show cryptography | head -2   # note the exact version for the file header
.venv/bin/python -m pytest 05-media-processing/post-production-pipeline/test_app.py -q -k receiver
```

Expected: FAIL (`webhook_receiver.py` not found).

- [ ] **Step 3: Write webhook_receiver.py**

Create `05-media-processing/post-production-pipeline/webhook_receiver.py` (in the header pin cryptography to the exact version Step 2 printed — shown here as `46.0.4`, replace with the real one):

```python
"""Verified fal webhook receiver - the piece every product backend needs.

fal signs every webhook delivery with Ed25519. Verifying means nobody can
forge "your render is done" calls into your backend. The scheme (verified
against fal's production signer):

    message   = f"{request_id}\\n{user_id}\\n{timestamp}\\n{sha256(raw_body).hexdigest()}"
    signature = X-Fal-Webhook-Signature header, hex-encoded Ed25519 over message
    keys      = https://rest.fal.ai/.well-known/jwks.json (cache <= 24h)

Hash the RAW bytes you received - never re-parse and re-serialize the JSON.
Deliveries repeat until fal sees a 2xx, so handle duplicates (this receiver
tracks request_ids and marks replays).

Run:
    pip install httpx==0.28.1 cryptography==46.0.4
    python webhook_receiver.py --port 8321
    # expose it publicly, e.g.:  cloudflared tunnel --url http://localhost:8321

Then submit with:  python client.py --app <owner>/<app> --webhook https://<tunnel-host>/fal-webhook
"""
import argparse
import base64
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

JWKS_URL = "https://rest.fal.ai/.well-known/jwks.json"
JWKS_TTL_S = 24 * 3600  # keys rotate; do not cache longer
TIMESTAMP_LEEWAY_S = 300

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}


def public_keys() -> list[Ed25519PublicKey]:
    now = time.time()
    if _jwks_cache["keys"] is None or now - _jwks_cache["fetched_at"] > JWKS_TTL_S:
        jwks = httpx.get(JWKS_URL, timeout=10).json()
        keys = []
        for entry in jwks.get("keys", []):
            x = entry["x"]
            raw = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
            keys.append(Ed25519PublicKey.from_public_bytes(raw))
        _jwks_cache.update(keys=keys, fetched_at=now)
    return _jwks_cache["keys"]


def verify(headers: dict[str, str], body: bytes, keys: list[Ed25519PublicKey]) -> tuple[bool, str]:
    """(ok, reason) for one delivery. `headers` must be lowercase-keyed."""
    try:
        request_id = headers["x-fal-webhook-request-id"]
        user_id = headers["x-fal-webhook-user-id"]
        timestamp = headers["x-fal-webhook-timestamp"]
        signature = bytes.fromhex(headers["x-fal-webhook-signature"])
    except KeyError as missing:
        return False, f"missing header {missing}"
    except ValueError:
        return False, "signature is not valid hex"
    try:
        skew = abs(time.time() - int(timestamp))
    except ValueError:
        return False, "timestamp is not an integer"
    if skew > TIMESTAMP_LEEWAY_S:
        return False, f"timestamp outside +/-{TIMESTAMP_LEEWAY_S}s (skew {skew:.0f}s)"
    message = "\n".join(
        [request_id, user_id, timestamp, hashlib.sha256(body).hexdigest()]
    ).encode()
    for key in keys:
        try:
            key.verify(signature, message)
            return True, "signature valid"
        except InvalidSignature:
            continue
    return False, "signature does not match any current JWKS key"


class Receiver(BaseHTTPRequestHandler):
    seen: set[str] = set()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        headers = {k.lower(): v for k, v in self.headers.items()}
        ok, reason = verify(headers, body, public_keys())
        if not ok:
            print(f"REJECTED: {reason}")
            self._respond(401, {"ok": False, "reason": reason})
            return
        event = json.loads(body)
        request_id = event.get("request_id", "?")
        replay = " (duplicate delivery - fal retries until it sees a 2xx)" \
            if request_id in self.seen else ""
        self.seen.add(request_id)
        print(f"VERIFIED status={event.get('status')} request_id={request_id}{replay}")
        if event.get("status") == "OK":
            for rendition in (event.get("payload") or {}).get("renditions", []):
                print(f"  {rendition['label']}: {rendition['video']['url']}")
        else:
            print(f"  error: {event.get('error')}")
            print(f"  payload: {json.dumps(event.get('payload'))[:300]}")
        self._respond(200, {"ok": True})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass  # our own prints above are the log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()
    print(f"listening on http://localhost:{args.port} - POST path can be anything")
    print(f"expose it, e.g.: cloudflared tunnel --url http://localhost:{args.port}")
    ThreadingHTTPServer(("", args.port), Receiver).serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the receiver tests to verify they pass**

```bash
.venv/bin/python -m pytest 05-media-processing/post-production-pipeline/test_app.py -q -k receiver
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add 05-media-processing/post-production-pipeline/webhook_receiver.py 05-media-processing/post-production-pipeline/test_app.py
git commit -m "feat(05-media-processing): Ed25519-verified webhook receiver (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: README + tier table + llms.txt

**Files:**
- Create: `05-media-processing/post-production-pipeline/README.md`
- Modify: `05-media-processing/README.md` (05.1 row: add folder link)
- Modify: `llms.txt` (add Live examples entry)

**Interfaces:**
- Consumes: everything from Tasks 2–4 (schema field names, script flags, sample URLs).
- Produces: the §4-A page; Task 7 fills the marked transcript block in Deploy.

- [ ] **Step 1: Write README.md**

Create `05-media-processing/post-production-pipeline/README.md`:

````markdown
# Post-production pipeline

A generated video isn't a shippable video. Every genmedia product ends up building the same layer by hand: rendition ladders so playback doesn't buffer, poster thumbnails for the feed, captions muxed in for accessibility — then a queue and a webhook so the backend hears about it. This example is that layer as one deployable fal app: ffmpeg on a cheap CPU machine, with fal's queue as the queue.

> Catalog slot `05.1` ([SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282)).

## What you get

```json
{
  "renditions": [
    { "label": "1080p", "video": { "url": "https://v3.fal.media/files/…/….mp4", "content_type": "video/mp4" },
      "width": 1920, "height": 1080, "size_bytes": 2264861, "encode_ms": 3861 },
    { "label": "720p",  "video": { "url": "…" }, "width": 1280, "height": 720, "size_bytes": 1113713, "encode_ms": 2405 },
    { "label": "480p",  "video": { "url": "…" }, "width": 854,  "height": 480, "size_bytes": 561120,  "encode_ms": 1633 }
  ],
  "thumbnails": [ { "url": "https://v3.fal.media/files/…/…_thumb_00.jpg", "content_type": "image/jpeg" } ],
  "captions_vtt": { "url": "https://v3.fal.media/files/…/captions.vtt", "content_type": "text/vtt" },
  "report": { "url": "https://v3.fal.media/files/…/report.json", "content_type": "application/json" }
}
```

Every rendition is an H.264/AAC MP4 with the moov atom up front (`+faststart`), so the CDN URL streams immediately in a `<video>` tag; captions ride along as a selectable `mov_text` track plus a WebVTT sidecar for web players. The `report` is the pipeline's own measurement: probe summary, per-rendition encode times, download/upload timings, skipped rungs, and the exact ffmpeg version that did the work.

## Run it

```bash
pip install fal==1.78.2
fal auth login
fal run app.py        # prints playground + API URLs; Ctrl-C to stop
```

The playground gives `video_url` a drag-and-drop upload widget; the defaults encode a 1080p/720p/480p ladder with two demo caption cues.

## How it works

**One endpoint, five stages.** Download → probe gate → parallel ladder → captions → thumbnails + uploads. The app is a stateless CPU processor; all the async machinery (queueing, status, webhooks) comes from the platform, not from app code — see Deploy.

**The environment is a Dockerfile, and that's the point.** ffmpeg is a system package, so this app pins it the honest way: a digest-pinned `python:3.11-slim` base plus apt's ffmpeg, built server-side by fal (no local Docker, no registry). Same pattern, same base digest as [01.1 same-model-two-ways](../../01-environments-containers/same-model-two-ways/) — read that example for why digests beat tags. The report echoes `ffmpeg -version` from inside the container, so provenance is verifiable per request.

**Bounded download, cached on `/data`.** `download_file` is SSRF-guarded and size-capped (300 MB here); sources land in a URL-hashed cache directory on [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage), so re-processing the same source (new ladder, new captions) skips the transfer — `report.timings_ms.download` goes to ~0.

**Probe first, encode later.** One `ffprobe` gates the request: still images and audio-only files are a typed 422 (`invalid_video`), durations over `MAX_DURATION_S` (600 s) are refused before they can burn minutes of encode time, and the probe's height feeds the **never-upscale rule** — requested rungs above the source height are skipped and listed in `report.ladder.skipped_no_upscale`, not silently stretched.

**The ladder runs in parallel.** Each rung is its own ffmpeg subprocess (`ThreadPoolExecutor` just babysits pipes — no GIL story), so wall-clock is roughly the slowest rung, not the sum: `report.timings_ms.encode_wall` vs the per-rendition `encode_ms` makes the difference visible. Encoding is H.264 `-crf 23 -preset veryfast` with `yuv420p` normalization — genmedia models love emitting 4:4:4 or 10-bit sources that browsers then refuse to play.

**Captions are validated before you pay for encoding.** SRT comes inline (`captions_srt`) or by URL (`captions_url`, at most one); the app converts it to WebVTT first — a malformed file fails as a fast typed 422 (`invalid_media`) before any rendition is encoded — then soft-muxes it into every MP4 as a selectable track.

**Outputs are typed toolkit objects on the fal CDN.** `Video`/`Image`/`File` with explicit content types (`from_path` deliberately doesn't guess — an `application/octet-stream` MP4 won't stream inline). URLs are served from `v3.fal.media`; retention follows your account's media lifecycle, and per-request retention is available via the `x-fal-object-lifecycle-preference` header (see [00.3 media-in-media-out](../../00-getting-started/media-in-media-out/)).

## Deploy

```bash
fal deploy app.py
```

Now the part every product backend actually needs — submit through **fal's queue** and get the result **pushed** to you. The queue API for your private app is the same one the public Model APIs use, with your app id in the URL:

```bash
# submit (returns request handles immediately)
curl -X POST "https://queue.fal.run/<your-username>/post-production-pipeline" \
  -H "Authorization: Key $FAL_KEY" -H "Content-Type: application/json" \
  -d '{"video_url": "https://storage.googleapis.com/falserverless/example_outputs/pika/pika_i2v_v22_output.mp4"}'

# poll
curl "https://queue.fal.run/<your-username>/post-production-pipeline/requests/<request_id>/status?logs=1" \
  -H "Authorization: Key $FAL_KEY"
```

Or use the two reference scripts, which is the full product story:

```bash
# terminal 1 - a receiver that VERIFIES fal's Ed25519 webhook signatures
pip install httpx==0.28.1 cryptography==<pin>
python webhook_receiver.py --port 8321
cloudflared tunnel --url http://localhost:8321   # or any public HTTPS route

# terminal 2 - submit with a webhook; polling stays on as the fallback
export FAL_KEY=<key_id>:<key_secret>
python client.py --app <your-username>/post-production-pipeline \
  --srt-file captions.srt --webhook https://<tunnel-host>/fal-webhook
```

Webhook deliveries are signed (Ed25519 over the request id, user id, timestamp, and the SHA-256 of the raw body — keys at `rest.fal.ai/.well-known/jwks.json`); `webhook_receiver.py` rejects anything that doesn't verify, checks timestamp freshness, and dedupes retries by `request_id`. Statuses are `IN_QUEUE → IN_PROGRESS → COMPLETED` — there is no `FAILED` status; failures arrive as `COMPLETED` with `error`/`error_type`, and the result URL returns the error body with a non-2xx code.

<!-- VALIDATION TRANSCRIPT (Task 7): replace this comment with the real
     client.py + webhook_receiver.py output from the validated deployment. -->

## Cost & tuning

- **CPU machines bill per second, per machine.** This app runs on `L`; the whole cost story is throughput — how many seconds of runner time a job needs. The report measures exactly that (`encode_wall`, per-rung `encode_ms`, `download`, `upload`) on your actual workload; tune from those numbers, not from folklore. See [machine types & pricing](https://fal.ai/pricing).
- **Ladder width ↔ vCPUs.** Rungs encode in parallel, so wall-clock ≈ the slowest rung — until the rungs starve each other for cores. `report.toolchain.vcpus` tells you what the machine actually has; if `encode_wall` approaches the *sum* of `encode_ms` values, the ladder is core-bound: use a bigger machine or fewer rungs.
- **`keep_alive=60`** keeps the worker warm for a minute after each job — post-production traffic is bursty (a batch of generations finishes, a batch of transcodes follows), and a warm worker skips the container cold start for the rest of the batch. Raise it if your queue is steadily busy; it bills like any other runner second.
- **The queue is free capacity management.** Submitting via `queue.fal.run` instead of a synchronous call means bursts wait in line instead of timing out, and webhook delivery means no client holds a connection open for a minute-long job.

## Next steps

- **[06.2 — Async queue + webhooks into a product backend](../../06-pipelines-apps/)**: this example's client story grown into a real app (Next.js submits, webhook updates the DB).
- **[05.3 — Image variant factory](../)**: the same CPU-proxy archetype for images (AVIF/WebP ladders).
- **HLS is the natural extension**: per-rung segment playlists + a master playlist. The ladder here is the hard part; HLS is packaging (`-f hls`) plus playlist URL rewriting for the CDN.
- Concept docs: [queue](https://docs.fal.ai/model-apis/queue) · [webhooks](https://docs.fal.ai/model-apis/webhooks) · [container apps](https://fal.ai/docs/documentation/development/use-custom-container-image) · [persistent storage](https://fal.ai/docs/documentation/development/use-persistent-storage)
````

Replace `cryptography==<pin>` with the exact version pinned in webhook_receiver.py (Task 4 Step 2). The `What you get` JSON is illustrative until Task 7 — Task 7 Step 4 replaces it with real values from the validated deployment.

- [ ] **Step 2: Link the folder from the tier table**

In `05-media-processing/README.md`, change the 05.1 row's Example cell from:

```
| `05.1` | Post-production pipeline (transcode ladder, thumbnails, captions; ffmpeg on CPU machines + queue + webhooks) | `net-new` | [SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282) |
```

to:

```
| `05.1` | Post-production pipeline (transcode ladder, thumbnails, captions; ffmpeg on CPU machines + queue + webhooks) — [`post-production-pipeline/`](post-production-pipeline/) | `net-new` | [SERV-1282](https://linear.app/features-and-labels/issue/SERV-1282) |
```

- [ ] **Step 3: Add the llms.txt entry**

In `llms.txt`, append to the `## Live examples` list (after the 01.1 line):

```
- [05.1 post-production-pipeline](05-media-processing/post-production-pipeline/README.md): transcode ladder + thumbnails + captions on a CPU machine - parallel ffmpeg rungs in a pinned container, faststart MP4s, soft-muxed captions with VTT sidecar, and a queue + Ed25519-verified-webhook client pair
```

- [ ] **Step 4: Commit**

```bash
git add 05-media-processing/post-production-pipeline/README.md 05-media-processing/README.md llms.txt
git commit -m "docs(05-media-processing): post-production pipeline README + catalog links (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Ephemeral deploy-check — make the harness green

**Files:**
- Modify (only if failures demand it): `05-media-processing/post-production-pipeline/app.py`, `test_app.py`

**Interfaces:**
- Consumes: live fal credentials — `fal auth login` session or `FAL_KEY` env var. STOP and ask Rehan if neither is available.
- Produces: a green `harness/check.py` run — the CI Definition of Done.

- [ ] **Step 1: Run the harness against this example only**

```bash
.venv/bin/python harness/check.py --example 05-media-processing/post-production-pipeline
```

Expected happy path: `deploying ephemeral app (machine_type=L, custom image, endpoint /, smoke payload {...})` → container build logs (first build takes minutes; the 900 s URL timeout accommodates it) → `url: https://…` → smoke response with 3 renditions → `running test_app.py against the live app` → all live tests pass → `runner … stopped` → `checked 1/1 example(s) OK`.

- [ ] **Step 2: Triage failures (use superpowers:systematic-debugging; known risk areas)**

- Smoke payload includes `captions_srt` (it has an example) — if muxing fails, the smoke fails: read the `[app]` log lines the harness prints.
- Garbage-SRT test: if ffmpeg's SRT demuxer accepted `"this is not a subtitle file"` leniently (no 422), swap the payload to `"\x00\x01 not srt"` as noted in Task 2.
- `Video`/`Image` JSON shape: if `content_type` asserts fail, print the actual response and align assertions with the real serialized fields.
- Runtime `pydantic` on the deployed worker is whatever fal ships — if `list[Rung]` + `default_factory` misbehaves, check the boot logs for schema errors.
- Timeout: if the full-pipeline test exceeds 120 s, drop the smoke/test ladder to `[720, 480]` — but record the observed timing in the PR description.

- [ ] **Step 3: Re-run until green, then commit any fixes**

```bash
.venv/bin/python harness/check.py --example 05-media-processing/post-production-pipeline
git add -A 05-media-processing/post-production-pipeline/
git commit -m "fix(05-media-processing): post-production pipeline deploy-check fixes (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(Skip the commit if nothing changed.)

---

### Task 7: Registered deploy + live queue/webhook validation

**Files:**
- Modify: `05-media-processing/post-production-pipeline/README.md` (insert real transcript + real output JSON)

**Interfaces:**
- Consumes: Rehan's fal account (`fal auth login`), `cloudflared` (or any public HTTPS tunnel), `client.py`, `webhook_receiver.py`.
- Produces: the ticket's queue/webhook verification evidence; the README Deploy section becomes fact, not theory.

- [ ] **Step 1: Deploy the registered app**

```bash
cd 05-media-processing/post-production-pipeline
../../.venv/bin/fal deploy app.py
```

Expected: a registered app URL containing `<username>/post-production-pipeline`. Note the exact app id printed — every later command uses it.

- [ ] **Step 2: Start the verified receiver + tunnel**

```bash
# terminal 1
../../.venv/bin/pip install cryptography
../../.venv/bin/python webhook_receiver.py --port 8321
# terminal 2
cloudflared tunnel --url http://localhost:8321
```

Note the `https://<random>.trycloudflare.com` URL. If cloudflared is unavailable, any public HTTPS reverse proxy to localhost:8321 works.

- [ ] **Step 3: Submit through the queue with a webhook**

```bash
cat > /tmp/demo-captions.srt <<'EOF'
1
00:00:00,000 --> 00:00:02,400
Every grand entrance begins somewhere.

2
00:00:02,400 --> 00:00:04,900
Post-production is what makes it ship.
EOF
export FAL_KEY=<key_id>:<key_secret>
../../.venv/bin/python client.py --app <username>/post-production-pipeline \
  --srt-file /tmp/demo-captions.srt \
  --webhook https://<tunnel-host>/fal-webhook
```

Verify ALL of: (a) client prints `queued: request_id=…` then `IN_PROGRESS` then `COMPLETED` with the result JSON; (b) the receiver prints `VERIFIED status=OK request_id=…` with rendition URLs; (c) a second identical submit shows the cache (`report.timings_ms.download` ≈ 0); (d) tamper check — restart the receiver with `--port 8321` and POST a hand-forged body to it (`curl -X POST localhost:8321 -d '{}'`) → prints `REJECTED: missing header …` and returns 401.

- [ ] **Step 4: Put the evidence in the README**

Replace the `<!-- VALIDATION TRANSCRIPT … -->` comment in README.md's Deploy section with a fenced transcript of (a)+(b) — client output on top, receiver lines below, secrets/tunnel hosts redacted. Replace the illustrative JSON in "What you get" with the real response body (URLs may be truncated with `…`). Update the `curl`/`client.py` examples if the deployed app id differs from `post-production-pipeline`.

- [ ] **Step 5: Commit**

```bash
git add 05-media-processing/post-production-pipeline/README.md
git commit -m "docs(05-media-processing): validated queue+webhook transcript (SERV-1282)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification + PR

- [ ] **Step 1: Full local verification (superpowers:verification-before-completion)**

```bash
.venv/bin/python harness/static_check.py
.venv/bin/python -m pytest -q
.venv/bin/python harness/check.py --example 05-media-processing/post-production-pipeline
```

Expected: static all `ok`; pytest green (live tests skip, unit + receiver tests pass); harness `checked 1/1 example(s) OK`.

- [ ] **Step 2: Request code review (superpowers:requesting-code-review) and fix findings**

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin claude/post-production-pipeline-example-3bf0b8
gh pr create --title "feat(05-media-processing): post-production pipeline (SERV-1282)" --body "$(cat <<'EOF'
Catalog slot 05.1 — transcode ladder, thumbnails, captions: ffmpeg on a CPU machine (custom container), platform queue + verified webhooks as the async layer.

- app.py: single endpoint, five stages (bounded /data-cached download → probe gate → parallel x264 ladder w/ never-upscale → SRT→VTT validation + mov_text mux → JPEG posters), typed 422s, provenance report
- Dockerfile: digest-pinned python:3.11-slim + apt ffmpeg (01.1 pattern)
- client.py: verified queue.fal.run reference for PRIVATE apps (submit/status?logs=1/result, fal_webhook)
- webhook_receiver.py: Ed25519 signature verification against fal's JWKS (+ unit tests)
- Validated: ephemeral deploy-check green; registered deploy + live queue→webhook transcript in the README
- Spec: docs/superpowers/specs/2026-08-04-post-production-pipeline-design.md

Closes SERV-1282 (docs-page follow-up tracked on the ticket).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Update the Linear ticket** — comment on SERV-1282 with the PR link, validation evidence summary, and the note that the fal-docs page is follow-up scope (per the 2026-08-04 session decision).

---

## Self-review notes

- Spec coverage: schema/stages/error taxonomy (Task 2), container/cost posture (Tasks 1–2), client + receiver (Tasks 3–4), README §4-A + inlined CDN/cost guidance (Task 5), CI tests (Task 2 + 6), registered-deploy validation (Task 7), DoD mapping (Tasks 6–8). Never-upscale, faststart, audio-conditional AAC, both-captions 422, duration cap: all have explicit tests.
- Types consistent across tasks: `normalize_ladder -> tuple[list[int], list[str]]`, `select_captions -> str|None`, `verify(headers, body, keys) -> tuple[bool, str]`, `Rendition(label, video, width, height, size_bytes, encode_ms)` used identically in app, tests, README JSON.
- Known intentional deviations from earlier drafts: thumbnail width computed in Python (`min(1280, iw)` pre-computed) instead of an lavfi `min()` expression — avoids filtergraph comma-quoting; `Image.from_bytes(data, "jpeg", file_name=…)` instead of `ImageSize` plumbing (width/height of thumbnails aren't the lesson).
