#!/usr/bin/env python3
"""Deploy-check harness - see harness/README.md.

Ephemeral-deploys each example, waits for it to become healthy, fires one
smoke request against its endpoint, runs the example's test_app.py against
the live app, and tears everything down. Exits non-zero if any example fails
to deploy, respond, or pass its tests - CI doubles as example rot detection.
CI checks correctness only - deploy + health + smoke + tests; it does not
measure performance.

Auth: FAL_KEY env var (CI) or a stored `fal auth login` session (local); the
same code path serves both via fal.sdk.get_credentials().

Written against fal==1.78.2 (pinned in harness/requirements.txt) and verified
against live ephemeral deploys. Version-specific facts this file relies on:
  - fal 1.78.2 discards the spawn worker's Future, so a deploy that fails
    before emitting a URL would hang forever; we own the spawn thread instead.
  - Ephemeral runners are INVISIBLE to `fal runners list` / `fal apps runners`
    (gRPC lists cover registered apps only); they ARE visible to REST
    GET /runners/{runner_id}, with the runner id parsed from the platform's
    "Starting runner <uuid>" log line.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import runpy
import sys
import threading
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

URL_TIMEOUT_S = float(os.environ.get("HARNESS_URL_TIMEOUT", "900"))
STARTUP_TIMEOUT_S = float(os.environ.get("HARNESS_STARTUP_TIMEOUT", "600"))
REQUEST_TIMEOUT_S = float(os.environ.get("HARNESS_REQUEST_TIMEOUT", "120"))
TEST_TIMEOUT_S = float(os.environ.get("HARNESS_TEST_TIMEOUT", "600"))
POLL_INTERVAL_S = 0.3
TEARDOWN_WAIT_S = 120.0

FAILED_RUNNER_STATES = {"CRASH_BACKOFF", "FAILURE_DELAY", "FAILED", "DEAD"}
RUNNER_ID_RE = re.compile(r"Starting runner ([0-9a-f-]{36})")


class CheckError(RuntimeError):
    """A per-example check failure (deploy, health, smoke, or tests)."""


def log(msg: str) -> None:
    print(msg, flush=True)


def auth_headers() -> dict[str, str]:
    """One auth path for local (Bearer via `fal auth login`) and CI (FAL_KEY)."""
    from fal.sdk import get_credentials

    return get_credentials().to_headers()


def rest_url() -> str:
    from fal import flags

    return flags.REST_URL


# --------------------------------------------------------------------------
# Example discovery and app loading
# --------------------------------------------------------------------------

def discover_examples() -> list[Path]:
    dirs = []
    for tier_dir in sorted(REPO_ROOT.glob("[0-9][0-9]-*")):
        if not tier_dir.is_dir():
            continue
        for example in sorted(tier_dir.iterdir()):
            if example.is_dir() and (example / "app.py").is_file():
                dirs.append(example)
    return dirs


def discover_variants(example_dir: Path) -> list[Path]:
    """App files to check for one example.

    Most examples are a single `app.py`. Comparison examples (e.g. the same
    model served two ways) ship additional `app_<variant>.py` files; each is
    deployed and tested independently, against the same test_app.py - if two
    builds of one model don't pass identical tests, the comparison is a lie.
    """
    return sorted(example_dir.glob("app*.py"))


def load_app_class(app_file: Path):
    """Load the unique fal.App subclass from an example's app.py.

    Mirrors what `fal run <file>` does (fal/utils.py::load_function_from):
    runpy-execute the file, pick the single App subclass, register the
    enclosing package for serialization.
    """
    import fal
    from fal._serialization import include_package_from_path

    example_dir = str(app_file.parent)
    sys.path.insert(0, example_dir)
    try:
        module = runpy.run_path(str(app_file))
    finally:
        # Avoid one example's dir shadowing another's imports in --all runs.
        if example_dir in sys.path:
            sys.path.remove(example_dir)
    include_package_from_path(str(app_file))

    app_classes = list(
        {
            obj
            for obj in module.values()
            if isinstance(obj, type) and issubclass(obj, fal.App) and obj is not fal.App
        }
    )
    if len(app_classes) != 1:
        raise CheckError(
            f"{app_file}: expected exactly one fal.App subclass, found {len(app_classes)}"
        )
    return app_classes[0]


def pick_endpoint(app_cls) -> tuple[str, object]:
    """Return (path, function) for the endpoint to smoke; prefer '/'. """
    endpoints = []
    for _, fn in inspect.getmembers(app_cls, inspect.isfunction):
        sig = getattr(fn, "route_signature", None)
        if sig is None or getattr(sig, "is_websocket", False):
            continue
        endpoints.append((sig.path, fn))
    if not endpoints:
        raise CheckError(f"{app_cls.__name__}: no HTTP @fal.endpoint found")
    for path, fn in endpoints:
        if path == "/":
            return path, fn
    return sorted(endpoints, key=lambda e: e[0])[0]


def sample_input(endpoint_fn) -> dict:
    """Build the smoke payload from the Input model's Field examples.

    Convention (harness/README.md): every required Input field carries
    `examples=[...]` - the same sample the README shows. Optional fields fall
    back to their server-side defaults.
    """
    from pydantic import BaseModel

    params = [
        p for name, p in inspect.signature(endpoint_fn).parameters.items()
        if name != "self"
    ]
    if not params:
        return {}
    model = params[0].annotation
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise CheckError(
            f"endpoint '{endpoint_fn.__name__}': request parameter must be a pydantic model"
        )
    payload = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.examples:
            payload[field_name] = field_info.examples[0]
        elif field_info.is_required():
            raise CheckError(
                f"Input field '{field_name}' has no examples=[...]; the harness "
                "builds its smoke request from Field examples (see harness/README.md)"
            )
    return payload


# --------------------------------------------------------------------------
# Boot observation: log drain (runner id) + REST runner polling (crash/teardown)
# --------------------------------------------------------------------------

class BootWatcher(threading.Thread):
    """Drains the spawn log queue, printing every line and capturing the
    runner id from the platform's 'Starting runner <uuid>' line."""

    def __init__(self, spawn_info):
        super().__init__(daemon=True, name="boot-watcher")
        self._logs = spawn_info.logs
        self._stop_event = threading.Event()
        self.runner_id: str | None = None

    def run(self) -> None:
        import queue as queue_mod

        while True:
            try:
                entry = self._logs.get(timeout=0.5)
            except queue_mod.Empty:
                if self._stop_event.is_set():
                    return
                continue
            log(f"    [app] {entry.message}")
            match = RUNNER_ID_RE.search(entry.message)
            if match and self.runner_id is None:
                self.runner_id = match.group(1)

    def stop(self) -> None:
        self._stop_event.set()


class RestRunnerPoller(threading.Thread):
    """Polls REST GET /runners/{runner_id} - the only API that can see an
    ephemeral runner - for crash detection and teardown confirmation."""

    def __init__(self, headers: dict[str, str]):
        super().__init__(daemon=True, name="rest-runner-poller")
        self._headers = headers
        self._stop_event = threading.Event()
        self.runner_id: str | None = None  # set once known
        self.last_state: str = ""
        self._failed: str | None = None

    def run(self) -> None:
        base = rest_url()
        with httpx.Client(timeout=10) as client:
            while not self._stop_event.is_set():
                if self.runner_id:
                    try:
                        resp = client.get(
                            f"{base}/runners/{self.runner_id}", headers=self._headers
                        )
                        if resp.is_success:
                            state = str(resp.json().get("state", ""))
                            self.last_state = state
                            if state in FAILED_RUNNER_STATES and self._failed is None:
                                self._failed = state
                    except Exception:
                        pass  # transient REST errors
                self._stop_event.wait(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()

    def failed_state_seen(self) -> str | None:
        return self._failed


# --------------------------------------------------------------------------
# Spawn, health, smoke
# --------------------------------------------------------------------------

def spawn_owned(fn):
    """Ephemeral-deploy on a harness-owned daemon thread.

    fn.spawn() (fal 1.78.2) submits the deploy to a non-daemon pool and
    discards the Future: a pre-URL failure is silently swallowed, and a stuck
    stream would block interpreter shutdown. Driving the same public
    Host.run(...) call ourselves keeps the worker's exception observable and
    lets the process exit regardless. Auth defaults to PRIVATE (upstream
    default is public - anyone with the URL could invoke and bill during the
    check window).
    """
    from fal.api.api import SpawnInfo, SpawnResultHandler

    info = SpawnInfo()
    failure: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            fn.host.run(
                fn.func,
                fn.options,
                args=(),
                kwargs={},
                application_name=fn.app_name,
                application_auth_mode=fn.app_auth or "private",
                result_handler=SpawnResultHandler(info),
                entrypoint=fn.run_entrypoint,
            )
        except BaseException as exc:  # surfaced by wait_for_url
            failure["exc"] = exc

    threading.Thread(target=worker, daemon=True, name="spawn-worker").start()
    return info, failure


def wait_for_url(spawn_info, failure: dict) -> str:
    """Bounded wait for the app URL, surfacing the deploy worker's real
    exception immediately instead of burning the whole timeout."""
    deadline = time.monotonic() + URL_TIMEOUT_S
    ready = getattr(spawn_info, "_url_ready", None)
    while time.monotonic() < deadline:
        if "exc" in failure:
            exc = failure["exc"]
            raise CheckError(f"deploy failed: {type(exc).__name__}: {exc}")
        if ready is not None and ready.is_set():
            return spawn_info.url
        time.sleep(0.25)
    raise CheckError(
        f"deploy produced no URL within {URL_TIMEOUT_S:.0f}s - the env "
        "build or scheduling stalled; see app logs above"
    )


def wait_healthy(url: str, headers: dict, poller: RestRunnerPoller) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_status: object = None
    with httpx.Client(timeout=10) as client:
        while time.monotonic() < deadline:
            failed = poller.failed_state_seen()
            if failed:
                raise CheckError(
                    f"runner entered {failed} during startup - see app logs"
                )
            try:
                resp = client.get(f"{url.rstrip('/')}/health", headers=headers)
                if resp.is_success:
                    return
                last_status = resp.status_code
            except httpx.HTTPError as exc:
                last_status = type(exc).__name__
            time.sleep(0.5)
    raise CheckError(
        f"app not healthy within {STARTUP_TIMEOUT_S:.0f}s (last: {last_status})"
    )


def smoke_request(url: str, path: str, payload: dict, headers: dict) -> None:
    """One validated request against the live endpoint."""
    target = url.rstrip("/") + path
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        resp = client.post(target, json=payload, headers=headers)
        if resp.status_code in (401, 403):
            headers = auth_headers()  # token may have expired mid-run; retry once
            resp = client.post(target, json=payload, headers=headers)
        if not resp.is_success:
            raise CheckError(
                f"smoke request failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        log(f"    smoke response: {resp.text[:200]}")


def run_example_tests(
    example_dir: Path, url: str, endpoint_path: str, variant: str = ""
) -> None:
    """Run the example's test_app.py against the live ephemeral app."""
    test_file = example_dir / "test_app.py"
    if not test_file.is_file():
        return
    import subprocess

    env = os.environ.copy()
    env["HARNESS_APP_URL"] = url
    env["HARNESS_ENDPOINT_PATH"] = endpoint_path
    env["HARNESS_VARIANT"] = variant
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )
    log("    running test_app.py against the live app")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q"],
            cwd=REPO_ROOT,
            env=env,
            timeout=TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise CheckError(f"test_app.py timed out after {TEST_TIMEOUT_S:.0f}s")
    if result.returncode != 0:
        raise CheckError(f"test_app.py failed (pytest exit {result.returncode})")


# --------------------------------------------------------------------------
# Teardown
# --------------------------------------------------------------------------

def teardown(spawn_info, poller: RestRunnerPoller, headers: dict) -> str | None:
    """Cancel the run stream (that IS the ephemeral teardown), then confirm
    via REST that the runner actually stopped - it bills until it does.

    Returns the runner id if it could not be confirmed stopped (a money leak
    the caller must escalate), else None.
    """
    poller.stop()
    stream = getattr(spawn_info, "stream", None)
    if stream is not None:
        try:
            stream.cancel()
        except Exception as exc:
            log(f"    warning: stream cancel failed ({exc})")

    runner_id = poller.runner_id
    if runner_id is None:
        return None
    base = rest_url()
    deadline = time.monotonic() + TEARDOWN_WAIT_S
    errors = 0
    with httpx.Client(timeout=10) as client:
        while time.monotonic() < deadline:
            try:
                resp = client.get(f"{base}/runners/{runner_id}", headers=headers)
                if resp.is_success:
                    data = resp.json()
                    state = str(data.get("state", ""))
                    if data.get("finished_at") or state in {
                        "TERMINATED", "TERMINATING", "DEAD", "DRAINING",
                    }:
                        log(f"    runner {runner_id} stopped ({state or 'finished'})")
                        return None
                elif resp.status_code == 404:
                    return None  # unknown to the control plane - nothing billing
            except Exception as exc:
                errors += 1
                if errors == 1:
                    log(f"    warning: teardown confirmation errored ({exc}); retrying")
            time.sleep(2)
    log(
        f"    ERROR: runner {runner_id} still alive after {TEARDOWN_WAIT_S:.0f}s "
        "- it keeps billing until terminated; check `fal runners list`"
    )
    return runner_id


# --------------------------------------------------------------------------
# Per-example check
# --------------------------------------------------------------------------

def check(example_dir: Path, teardown_report: dict | None = None) -> None:
    """Deploy-check every variant of one example."""
    variants = discover_variants(example_dir)
    if not variants:
        raise CheckError(f"{example_dir}: no app.py")
    for app_file in variants:
        if len(variants) > 1:
            log(f"  variant {app_file.name}")
        check_variant(example_dir, app_file, teardown_report=teardown_report)


def check_variant(
    example_dir: Path, app_file: Path, teardown_report: dict | None = None
) -> None:
    from fal.app import wrap_app

    app_cls = load_app_class(app_file)
    endpoint_path, endpoint_fn = pick_endpoint(app_cls)
    payload = sample_input(endpoint_fn)
    headers = auth_headers()

    fn = wrap_app(app_cls)
    log(
        f"  deploying ephemeral app (machine_type={app_cls.machine_type}"
        f"{', custom image' if getattr(app_cls, 'image', None) else ''}, "
        f"endpoint {endpoint_path}, smoke payload {json.dumps(payload)})"
    )

    poller = RestRunnerPoller(headers)
    poller.start()
    spawn_info, spawn_failure = spawn_owned(fn)
    watcher = BootWatcher(spawn_info)
    watcher.start()
    try:
        url = wait_for_url(spawn_info, spawn_failure)
        log(f"    url: {url}")
        if watcher.runner_id:
            poller.runner_id = watcher.runner_id
        wait_healthy(url, headers, poller)
        smoke_request(url, endpoint_path, payload, headers)
        run_example_tests(example_dir, url, endpoint_path, variant=app_file.stem)
    finally:
        if poller.runner_id is None and watcher.runner_id:
            poller.runner_id = watcher.runner_id
        leaked = teardown(spawn_info, poller, headers)
        if leaked and teardown_report is not None:
            teardown_report["leaked_runner"] = leaked
        watcher.stop()


def purge_example_modules(example_dir: Path) -> None:
    """Drop modules imported from this example's dir so a later example's
    same-named helper (e.g. utils.py) can't silently receive stale code."""
    prefix = str(example_dir)
    for name in [
        n for n, mod in list(sys.modules.items())
        if (getattr(mod, "__file__", "") or "").startswith(prefix)
    ]:
        del sys.modules[name]


def preflight(headers: dict) -> None:
    """One cheap authenticated call so a bad/revoked credential fails the run
    in seconds with a real error, not one silent URL timeout per example."""
    resp = httpx.get(f"{rest_url()}/users/current", headers=headers, timeout=30)
    if not resp.is_success:
        raise CheckError(
            f"auth preflight failed: HTTP {resp.status_code}: {resp.text[:200]} "
            "- check FAL_KEY (CI) or `fal auth login` (local)"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="", help="one example dir; empty = all")
    args = parser.parse_args()

    if args.example:
        target = Path(args.example)
        if not target.is_absolute():
            target = REPO_ROOT / target
        if not (target / "app.py").is_file():
            log(f"error: {target} is not an example dir (no app.py)")
            return 2
        examples = [target]
    else:
        examples = discover_examples()
        if not examples:
            log("no examples found under the tier directories - nothing to check")
            return 0

    try:
        preflight(auth_headers())
    except CheckError as exc:
        log(f"error: {exc}")
        return 1
    except Exception as exc:
        log(
            f"error: credential setup failed: {type(exc).__name__}: {exc} "
            "- check FAL_KEY (CI) or `fal auth login` (local)"
        )
        return 1

    failures = []
    for example_dir in examples:
        rel = example_dir.relative_to(REPO_ROOT) if example_dir.is_relative_to(REPO_ROOT) else example_dir
        log(f"checking {rel}")
        teardown_report: dict = {}
        try:
            check(example_dir, teardown_report=teardown_report)
        except CheckError as exc:
            log(f"  FAILED: {exc}")
            failures.append((rel, str(exc)))
            continue
        except Exception as exc:  # unexpected - still fail the example, keep going
            log(f"  FAILED (unexpected {type(exc).__name__}): {exc}")
            failures.append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        finally:
            purge_example_modules(example_dir)
        log(f"  OK {rel}")
        if teardown_report.get("leaked_runner"):
            # The check passed, but a billing runner outlived teardown: the run
            # must go red so someone looks at it.
            failures.append(
                (rel, f"runner {teardown_report['leaked_runner']} not confirmed stopped")
            )

    log("")
    log(f"checked {len(examples) - len(failures)}/{len(examples)} example(s) OK")
    for rel, reason in failures:
        log(f"  FAILED {rel}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
