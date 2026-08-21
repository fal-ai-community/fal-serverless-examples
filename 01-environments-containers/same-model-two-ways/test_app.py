"""One suite run against both build variants (HARNESS_VARIANT selects which)."""
import json
import os
import tempfile
from pathlib import Path

SAMPLE = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
EXPECTED_BUILDER = {"app": "fal-runtime", "app_container": "container"}
# Cross-variant transcript handoff; keyed by parent PID so a stale ledger
# from an earlier run can't poison the compare.
LEDGER = Path(tempfile.gettempdir()) / f"same-model-two-ways-{os.getppid()}.json"


def variant() -> str:
    return os.environ.get("HARNESS_VARIANT", "app")


def test_transcribes_real_speech(app_client):
    response = app_client.run("/", {"audio_url": SAMPLE})
    text = response["text"]
    assert text, "empty transcript"
    # No exact-words assert (breaks on model bumps); distinct-word count
    # catches whisper's repetition loop on non-speech audio.
    words = [w.strip(".,!?").lower() for w in text.split()]
    assert len(set(words)) >= 8, f"degenerate transcript: {text[:120]!r}"


def test_reports_its_own_build_path(app_client):
    response = app_client.run("/", {"audio_url": SAMPLE})
    assert response["built_with"] == EXPECTED_BUILDER[variant()]


def test_both_build_paths_agree(app_client):
    """Each variant records its transcript; whichever runs second compares."""
    text = app_client.run("/", {"audio_url": SAMPLE})["text"].strip()
    ledger = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    for other, other_text in ledger.items():
        if other != variant():
            assert text == other_text, (
                f"{variant()} and {other} produced different transcripts - "
                f"the two build paths are no longer equivalent:\n"
                f"  {variant()}: {text[:100]!r}\n  {other}: {other_text[:100]!r}"
            )
    ledger[variant()] = text
    LEDGER.write_text(json.dumps(ledger))
