"""Fast weight loading for 20+ GB models on fal."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL_REPO = "diffusers/LTX-2.3-Distilled-Diffusers"
MODEL_REV = "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee"  # pin: repo state 2026-05-21
MODEL_DIR = "/data/models/ltx-2.3-distilled-diffusers"

# madebyollin/taehv (MIT), pinned to the branch with the LTX-2.3 "wide" decoder weights
_TAEHV_REF = "32ac0146b11007cda5a57b60a3b35653361fb8a4"
TAEHV_WEIGHTS_URL = f"https://raw.githubusercontent.com/madebyollin/taehv/{_TAEHV_REF}/taeltx2_3_wide.pth"
TAEHV_WEIGHTS_SHA = (
    "007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339"  # spike-verified
)
_TAEHV_DIR = Path("/data/models/taehv")


def tune_hf_env() -> None:
    """Parallelize Hugging Face Xet downloads. Call before importing huggingface_hub."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault(
        "HF_XET_CHUNK_CACHE_SIZE_BYTES", "1000000000000"
    )  # 1 TB on /data
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # non-Xet repos


def fetch_model() -> str:
    """Download pinned snapshot to /data; warm boots skip ~45s of hub metadata calls."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    t0 = time.perf_counter()
    try:
        path = snapshot_download(
            MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, local_files_only=True
        )
        print(f"[loader] model cache hit in {time.perf_counter() - t0:.1f}s")
    except (LocalEntryNotFoundError, FileNotFoundError):
        print(f"[loader] cold download of {MODEL_REPO} (~88 GB) ...")
        path = snapshot_download(
            MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR, max_workers=32
        )
        print(f"[loader] downloaded in {time.perf_counter() - t0:.1f}s")
    return path


def fetch_taehv() -> Path:
    """Fetch the pinned tiny-VAE weights to /data with atomic rename + sha check."""
    dest = _TAEHV_DIR / Path(TAEHV_WEIGHTS_URL).name
    if dest.exists():
        return dest
    _TAEHV_DIR.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.NamedTemporaryFile(dir=_TAEHV_DIR, delete=False) as tmp,
        urllib.request.urlopen(TAEHV_WEIGHTS_URL, timeout=60) as resp,
    ):
        shutil.copyfileobj(resp, tmp)
    digest = hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest()
    if digest != TAEHV_WEIGHTS_SHA:
        os.unlink(tmp.name)
        raise RuntimeError(f"taehv weights sha mismatch: {digest}")
    os.rename(tmp.name, dest)  # atomic on /data: concurrent workers see all-or-nothing
    return dest


def warm_page_cache(directory: str, parallelism: int = 32) -> float:
    """Pre-read files; parallel beats sequential ~6-15x on /data's chunked store."""
    t0 = time.perf_counter()
    subprocess.run(
        f"find {shlex.quote(directory)} -type f"
        f" | xargs -P {parallelism} -I {{}} cat {{}} > /dev/null",
        shell=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[loader] page cache warm in {elapsed:.1f}s ({directory})")
    return elapsed
