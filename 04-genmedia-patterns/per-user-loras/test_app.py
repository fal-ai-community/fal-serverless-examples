"""Smoke test - run by the harness against the live ephemeral app.

Unit tests of the pure helpers run free (no GPU, no deploy); integration
tests take the harness `app_client` fixture and skip unless HARNESS_APP_URL
is set (harness/check.py sets it after the ephemeral deploy).
"""
import importlib.util
import os
import struct
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location(
    "per_user_loras_app", Path(__file__).parent / "app.py"
)
app_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_mod)


class TestHintFor:
    def test_prefix_and_value(self):
        assert app_mod.hint_for("ukiyo-e") == "lora:ukiyo-e"

    def test_latin1_comma_free_and_bounded(self):
        h = app_mod.hint_for("a" * 40)
        h.encode("latin-1")
        assert "," not in h and len(h) <= 45


class TestUserIdPattern:
    @pytest.mark.parametrize("ok", ["ukiyo-e", "yarn-art", "u_001", "a"])
    def test_valid(self, ok):
        assert app_mod.USER_ID_RE.match(ok)

    @pytest.mark.parametrize(
        "bad", ["", "-lead", "UPPER", "has space", "x" * 41, "a,b", "Ünïcode"]
    )
    def test_invalid(self, bad):
        assert not app_mod.USER_ID_RE.match(bad)

    def test_all_featured_users_match(self):
        for uid in app_mod.FEATURED_USERS:
            assert app_mod.USER_ID_RE.match(uid)


class TestValidateLoraSource:
    def test_https_url(self):
        assert (
            app_mod.validate_lora_source(
                "https://huggingface.co/a/b/resolve/main/w.safetensors"
            )
            == "url"
        )

    def test_hf_repo_id(self):
        assert (
            app_mod.validate_lora_source("fal/flux-2-klein-4b-spritesheet-lora")
            == "hf_repo"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "http://example.com/w.safetensors",  # https only
            "ftp://example.com/w.safetensors",
            "https://",  # no host
            "not a url at all",
            "a/b/c",  # not owner/repo
        ],
    )
    def test_rejects(self, bad):
        from fal.exceptions import FieldException

        with pytest.raises(FieldException):
            app_mod.validate_lora_source(bad)


class TestLooksLikeSafetensors:
    def _head(self, header_len: int, first: bytes = b"{") -> bytes:
        return struct.pack("<Q", header_len) + first

    def test_accepts_wellformed(self):
        assert app_mod.looks_like_safetensors(self._head(100), file_size=200)

    def test_rejects_header_longer_than_file(self):
        assert not app_mod.looks_like_safetensors(self._head(10_000), file_size=200)

    def test_rejects_non_json_start(self):
        assert not app_mod.looks_like_safetensors(
            self._head(100, first=b"<"), file_size=200
        )

    def test_rejects_short_head(self):
        assert not app_mod.looks_like_safetensors(b"\x00\x01", file_size=200)


class TestIsOomError:
    def test_plain_runtime_error_is_not_oom(self):
        assert not app_mod.is_oom_error(RuntimeError("size mismatch"))

    def test_message_marker(self):
        assert app_mod.is_oom_error(
            RuntimeError("CUDA out of memory. Tried to allocate...")
        )

    def test_class_name(self):
        class OutOfMemoryError(Exception):
            pass

        assert app_mod.is_oom_error(OutOfMemoryError("x"))

    def test_walks_cause_chain(self):
        inner = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        outer = ValueError("Failed to load LoRA")
        outer.__cause__ = inner
        assert app_mod.is_oom_error(outer)

    def test_cycle_protection(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert not app_mod.is_oom_error(a)


class TestSyntheticSeed:
    def test_deterministic_and_distinct(self):
        assert app_mod.synthetic_seed("u1") == app_mod.synthetic_seed("u1")
        assert app_mod.synthetic_seed("u1") != app_mod.synthetic_seed("u2")
        assert 0 <= app_mod.synthetic_seed("u1") < 2**31


# ---------------------------------------------------------------------------
# Integration tests - run by the harness against the live ephemeral deploy.
# The app_client fixture skips every test below when HARNESS_APP_URL is unset.
# ---------------------------------------------------------------------------


def _direct_post(path: str, payload: dict, hint: str | None = None) -> httpx.Response:
    """Raw POST so header assertions and routing hints are possible
    (app_client sends and returns json only)."""
    import fal.sdk

    url = os.environ["HARNESS_APP_URL"].rstrip("/") + path
    headers = fal.sdk.get_credentials().to_headers()
    if hint is not None:
        headers["X-Fal-Runner-Hint"] = hint
    return httpx.post(url, json=payload, headers=headers, timeout=120)


class TestGenerate:
    def test_featured_user_smoke(self, app_client):
        result = app_client.run("/", {
            "user_id": "ukiyo-e",
            "prompt": "ukiyo-e woodblock print of a red fox in a forest clearing",
            "seed": 42,
        })
        assert result["image"]["url"]
        assert result["seed"] == 42
        assert result["serving"]["hint"] == "lora:ukiyo-e"
        assert result["serving"]["adapter_source"] in ("gpu-cache", "disk", "downloaded")
        assert "lora" in result["timings"] and "inference" in result["timings"]

    def test_hint_echo_header(self, app_client):
        resp = _direct_post("/", {
            "user_id": "ukiyo-e",
            "prompt": "ukiyo-e woodblock print of a lighthouse at dusk",
            "seed": 7,
        })
        resp.raise_for_status()
        assert resp.headers.get("x-fal-runner-hint") == "lora:ukiyo-e"

    def test_unregistered_user_is_typed_422(self, app_client):
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            app_client.run("/", {"user_id": "nobody-here", "prompt": "x", "seed": 1})
        resp = excinfo.value.response
        assert resp.status_code == 422
        detail = resp.json()["detail"][0]
        assert detail["type"] == "user_not_registered"
        assert detail["loc"] == ["body", "user_id"]

    def test_bad_user_id_pattern_is_422(self, app_client):
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            app_client.run("/", {"user_id": "NOT VALID!", "prompt": "x", "seed": 1})
        assert excinfo.value.response.status_code == 422


class TestRegisterFlow:
    def test_synthetic_register_then_cache_transition(self, app_client):
        reg = app_client.run("/register", {"user_id": "it-synth-1"})
        assert reg["kind"] == "synthetic"
        assert reg["hint"] == "lora:it-synth-1"
        assert reg["adapter_bytes"] > 0

        first = app_client.run("/", {
            "user_id": "it-synth-1", "prompt": "a red bicycle", "seed": 5,
        })
        assert first["serving"]["adapter_source"] in ("disk", "downloaded")

        # Hint the repeat like a real client - unhinted, a second warm runner
        # could take it, reload from disk, and flake both assertions.
        resp = _direct_post("/", {
            "user_id": "it-synth-1", "prompt": "a blue bicycle", "seed": 6,
        }, hint=first["serving"]["hint"])
        resp.raise_for_status()
        second = resp.json()
        assert second["serving"]["adapter_source"] == "gpu-cache"
        assert second["serving"]["worker_id"] == first["serving"]["worker_id"]

    def test_register_real_hf_repo(self, app_client):
        reg = app_client.run("/register", {
            "user_id": "it-real-1",
            "lora_url": "fal/flux-2-klein-4b-spritesheet-lora",
            "scale": 1.0,
        })
        assert reg["kind"] == "registered"
        gen = app_client.run("/", {
            "user_id": "it-real-1",
            "prompt": "2x2 sprite sheet of a walking robot",
            "seed": 11,
        })
        assert gen["image"]["url"]

    def test_register_bad_url_is_typed_422(self, app_client):
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            app_client.run("/register", {
                "user_id": "it-bad-1",
                "lora_url": "http://insecure.example.com/w.safetensors",
            })
        resp = excinfo.value.response
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["type"] == "lora_download_failed"
