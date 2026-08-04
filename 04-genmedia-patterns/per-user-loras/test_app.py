"""Smoke test - run by the harness against the live ephemeral app.

Unit tests of the pure helpers run free (no GPU, no deploy); integration
tests take the harness `app_client` fixture and skip unless HARNESS_APP_URL
is set (harness/check.py sets it after the ephemeral deploy).
"""
import importlib.util
import struct
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "per_user_loras_app", Path(__file__).parent / "app.py"
)
app_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_mod)


class TestHintFor:
    def test_prefix_and_value(self):
        assert app_mod.hint_for("numzoo") == "lora:numzoo"

    def test_latin1_comma_free_and_bounded(self):
        h = app_mod.hint_for("a" * 40)
        h.encode("latin-1")
        assert "," not in h and len(h) <= 45


class TestUserIdPattern:
    @pytest.mark.parametrize("ok", ["numzoo", "yarn-art", "u_001", "a"])
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
