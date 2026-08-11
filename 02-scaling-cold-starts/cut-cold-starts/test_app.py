"""One suite for both variants plus free (no-GPU, no-deploy) unit tests.

Live tests use the harness `app_client` fixture (skips outside harness runs).
The harness deploys every app*.py in this folder and runs this file against
each; HARNESS_VARIANT says which one is live. Free tests exercise the shared
module logic that must hold before any GPU is spent.
"""
import os
import sys
import types
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wan_flashpack  # top level is stdlib-only by design

EXPECTED_VARIANT = {"app": "optimized", "app_baseline": "baseline"}
# The same small/seeded request the harness smoke uses (Field examples).
SMALL_REQUEST = {
    "prompt": "A calico cat surfing a small wave at golden hour, cinematic",
    "num_frames": 17,
    "num_inference_steps": 8,
    "seed": 42,
}


def variant() -> str:
    return os.environ.get("HARNESS_VARIANT", "app")


# --------------------------- free unit tests -------------------------------

def test_publish_classes_creates_missing_module_entry():
    """On a fal runner the module arrives as a cloudpickle by-value
    reconstruction with NO sys.modules entry; _publish_classes must create
    one so transformers introspection and diffusers importlib resolution work."""
    name = "wan_flashpack_test_ghost"
    sys.modules.pop(name, None)
    try:
        mod = wan_flashpack._publish_classes(name, {"Marker": dict})
        assert sys.modules[name] is mod
        assert mod.Marker is dict
        import importlib
        assert importlib.import_module(name) is mod
    finally:
        sys.modules.pop(name, None)


def test_publish_classes_is_idempotent_on_real_modules():
    """When the module is normally importable (local runs), the existing
    entry must be reused, not replaced."""
    before = sys.modules["wan_flashpack"]
    mod = wan_flashpack._publish_classes("wan_flashpack", {})
    assert mod is before
    assert sys.modules["wan_flashpack"] is before


def test_module_top_level_is_import_light():
    """Static CI loads app files with only fal+httpx+pytest installed; the
    heavy imports must all live inside get_classes()."""
    import ast
    tree = ast.parse((HERE / "wan_flashpack.py").read_text())
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.add((node.module or "").split(".")[0])
    assert top_level_imports <= {"sys", "types", "__future__"}, top_level_imports
