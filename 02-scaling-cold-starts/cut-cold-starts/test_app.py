"""One test suite for both app variants: free (no-GPU) unit tests plus live harness tests."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wan_flashpack  # top level is stdlib-only by design

EXPECTED_VARIANT = {"app": "optimized", "app_baseline": "baseline"}
# Keep in sync with the Field examples the harness smoke payload uses.
SMALL_REQUEST = {
    "prompt": "A calico cat surfing a small wave at golden hour, cinematic",
    "num_frames": 17,
    "num_inference_steps": 8,
    "seed": 42,
}


def variant() -> str:
    return os.environ.get("HARNESS_VARIANT", "app")


def test_publish_classes_creates_missing_module_entry():
    """Cloudpickle by-value modules arrive with no sys.modules entry; _publish_classes must create one."""
    name = "wan_flashpack_test_ghost"
    sys.modules.pop(name, None)
    try:
        mod = wan_flashpack._publish_classes(name, {"Marker": dict})
        assert sys.modules[name] is mod
        assert mod.Marker is dict
        assert importlib.import_module(name) is mod
    finally:
        sys.modules.pop(name, None)


def test_publish_classes_is_idempotent_on_real_modules():
    """A normally imported module's sys.modules entry must be reused, not replaced."""
    before = sys.modules["wan_flashpack"]
    mod = wan_flashpack._publish_classes("wan_flashpack", {})
    assert mod is before
    assert sys.modules["wan_flashpack"] is before


def test_module_top_level_is_import_light():
    """Static CI imports this module with only fal+httpx+pytest installed."""
    tree = ast.parse((HERE / "wan_flashpack.py").read_text())
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.add((node.module or "").split(".")[0])
    assert top_level_imports <= {"sys", "types", "__future__"}, top_level_imports


def test_wan_flashpack_registers_bare_class_names_for_diffusers_prefix_strip():
    """diffusers 0.39 strips the 'FlashPack' prefix when resolving class names, so each class must register under both names."""
    source = (HERE / "wan_flashpack.py").read_text()
    tree = ast.parse(source)

    found_dual_registration = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            has_full_name_setattr = False
            has_removeprefix_setattr = False

            for item in ast.walk(node):
                if isinstance(item, ast.Call):
                    if (
                        isinstance(item.func, ast.Name)
                        and item.func.id == "setattr"
                        and len(item.args) >= 3
                    ):
                        second_arg = item.args[1]
                        if isinstance(second_arg, ast.Attribute):
                            if second_arg.attr == "__name__":
                                has_full_name_setattr = True

                        if (
                            isinstance(second_arg, ast.Call)
                            and isinstance(second_arg.func, ast.Attribute)
                            and second_arg.func.attr == "removeprefix"
                            and isinstance(second_arg.func.value, ast.Attribute)
                            and second_arg.func.value.attr == "__name__"
                        ):
                            has_removeprefix_setattr = True

            if has_full_name_setattr and has_removeprefix_setattr:
                found_dual_registration = True
                break

    assert (
        found_dual_registration
    ), "Registration loop must use setattr twice for each class: once with __name__ and once with removeprefix('FlashPack')"


def test_pipeline_load_receives_explicit_text_encoder():
    """from_pretrained_flashpack must get an explicit re-tied text_encoder kwarg or the UMT5 weight tie silently breaks."""
    source = (HERE / "app.py").read_text()
    tree = ast.parse(source)

    def is_pipeline_target(value: ast.expr) -> bool:
        if isinstance(value, ast.Name) and value.id == "FlashPackWanPipeline":
            return True
        if isinstance(value, ast.Subscript):
            key = value.slice
            return isinstance(key, ast.Constant) and key.value == "FlashPackWanPipeline"
        return False

    found_text_encoder_kwarg = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained_flashpack"
            and is_pipeline_target(node.func.value)
            and any(kw.arg == "text_encoder" for kw in node.keywords)
        ):
            found_text_encoder_kwarg = True
            break

    assert found_text_encoder_kwarg, (
        "FlashPackWanPipeline.from_pretrained_flashpack must receive an "
        "explicit text_encoder= kwarg (loaded + re-tied beforehand) - the "
        "single-call form silently breaks the UMT5 embed_tokens/shared tie"
    )


# app_client skips the tests below unless the harness sets HARNESS_APP_URL.


def test_small_seeded_generation(app_client):
    response = app_client.run("/", SMALL_REQUEST)
    assert response["seed"] == 42, "seed must echo back for reproducibility"
    assert response["video"]["url"].startswith("https://"), response["video"]
    assert response["video"]["content_type"] == "video/mp4", response["video"]
    assert response["timings"]["generate_s"] > 0


def test_setup_report_matches_variant(app_client):
    report = app_client.run("/setup-report", {})
    assert report["variant"] == EXPECTED_VARIANT[variant()]
    assert report["total_seconds"] > 0
    stage_names = list(report["timings"])
    assert stage_names, "setup() must record its stages"
    if report["variant"] == "optimized":
        assert any("FlashPack" in name for name in stage_names), stage_names
        assert any("inductor" in name for name in stage_names), stage_names
    else:
        assert report["built_caches_this_boot"] is False


def test_rejects_non_4k_plus_1_frames(app_client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        app_client.run("/", {**SMALL_REQUEST, "num_frames": 18})
    assert exc_info.value.response.status_code == 422


def _load_bench():
    """Load bench.py under a unique name: plain `import bench` could collide in a whole-repo pytest run."""
    spec = importlib.util.spec_from_file_location(
        "cut_cold_starts_bench", HERE / "bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_table_formats_stages_in_order():
    bench = _load_bench()

    runs = [
        {
            "spawn_to_healthy_s": 100.0,
            "variant": "optimized",
            "gpu": "H100",
            "built_caches_this_boot": False,
            "timings": {
                "imports": 5.0,
                "load weights (FlashPack, /data -> GPU)": 9.5,
            },
            "total_seconds": 90.0,
        },
        {
            "spawn_to_healthy_s": 700.0,
            "variant": "optimized",
            "gpu": "H100",
            "built_caches_this_boot": True,
            "timings": {
                "imports": 5.2,
                "download + build FlashPack pack (first boot only)": 400.0,
                "load weights (FlashPack, /data -> GPU)": 12.0,
            },
            "total_seconds": 650.0,
        },
    ]
    table = bench.format_table(runs)
    lines = table.splitlines()
    assert lines[0].startswith("| stage |")
    assert "| imports | 5.0 | 5.2 |" in table
    assert "| download + build FlashPack pack (first boot only) | - | 400.0 |" in table
    assert "| setup() total | 90.0 | 650.0 |" in table
    assert "| spawn -> healthy | 100.0 | 700.0 |" in table
