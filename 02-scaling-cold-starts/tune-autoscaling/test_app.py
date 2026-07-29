"""Two kinds of test, both run by CI.

* Unit tests over the pure model functions - free, no deploy, run on every
  PR by the static job. model.py is stdlib-only, so they need nothing
  beyond pytest.
* Live tests that fire real requests at an ephemeral deploy (added in a
  later task); the `app_client` fixture skips when no app is under test.

The unit tests side-load model.py by path rather than importing it, so the
example folder needs no package plumbing - same trick as
00-getting-started/bulletproof-inputs/test_app.py.
"""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "autoscale_model", Path(__file__).parent / "model.py"
)
model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = model
_spec.loader.exec_module(model)


# Every labelled bar from the three scaling diagrams published on
# docs.fal.ai/documentation/deployment/scale-your-application. fal states no
# formula in prose, so these charts are the only published ground truth for
# how the autoscaler picks a runner count - which makes them the oracle.
PUBLISHED_CHARTS = [
    # (config, expected runners for ongoing = 0, 1, 2, ...)
    (
        dict(min_concurrency=1, concurrency_buffer=1, max_multiplexing=1, max_concurrency=10),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10],
    ),
    (
        dict(min_concurrency=3, concurrency_buffer=2, max_multiplexing=1, max_concurrency=10),
        [3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
    ),
    (
        dict(min_concurrency=0, concurrency_buffer=2, max_multiplexing=4, max_concurrency=6),
        [2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
    ),
]


def test_law_matches_every_published_diagram():
    for config, expected in PUBLISHED_CHARTS:
        actual = [model.desired_runners(n, **config) for n in range(len(expected))]
        assert actual == expected, f"config {config}: {actual} != {expected}"


def test_buffer_percentage_is_a_floor_not_a_replacement():
    # Docs: "the actual buffer is the maximum of concurrency_buffer and
    # concurrency_buffer_perc / 100 * request volume."
    assert model.desired_runners(10, concurrency_buffer=5, concurrency_buffer_perc=20) == 15
    assert model.desired_runners(100, concurrency_buffer=5, concurrency_buffer_perc=20) == 120


def test_min_concurrency_wins_when_higher_than_load_plus_buffer():
    # Docs: "The system keeps whichever is higher."
    assert model.desired_runners(0, min_concurrency=4, concurrency_buffer=1) == 4
    assert model.desired_runners(6, min_concurrency=4, concurrency_buffer=1) == 7


def test_max_concurrency_is_a_hard_ceiling():
    assert model.desired_runners(50, max_concurrency=3) == 3
    assert model.desired_runners(50, min_concurrency=10, max_concurrency=3) == 3


def test_published_h100_list_rate():
    assert model.H100_USD_PER_HOUR == 3.99
    assert abs(model.USD_PER_SECOND - 3.99 / 3600) < 1e-12


def test_billed_window_spans_setup_through_keep_alive_expiry():
    # One runner: SETUP starts at t=0, last request finishes at t=40,
    # keep_alive=60 -> alive 0..100 -> 100 billed seconds.
    windows = [model.RunnerWindow("r1", setup_start_s=0.0, last_completion_s=40.0)]
    assert model.billable_seconds(windows, keep_alive=60) == 100.0


def test_billable_seconds_sums_across_runners():
    windows = [
        model.RunnerWindow("r1", setup_start_s=0.0, last_completion_s=40.0),
        model.RunnerWindow("r2", setup_start_s=30.0, last_completion_s=45.0),
    ]
    # r1: 0..100 = 100. r2: 30..105 = 75. Total 175.
    assert model.billable_seconds(windows, keep_alive=60) == 175.0


def test_keep_alive_zero_bills_only_setup_plus_work():
    windows = [model.RunnerWindow("r1", setup_start_s=10.0, last_completion_s=25.0)]
    assert model.billable_seconds(windows, keep_alive=0) == 15.0


def test_granularity_rounds_up_per_runner():
    # Billing granularity is published NOWHERE. The model rounds each
    # runner's alive time up to `granularity`, and states the assumption.
    windows = [
        model.RunnerWindow("r1", setup_start_s=0.0, last_completion_s=0.5),
        model.RunnerWindow("r2", setup_start_s=0.0, last_completion_s=0.5),
    ]
    assert model.billable_seconds(windows, keep_alive=0, granularity=1.0) == 2.0
    assert model.billable_seconds(windows, keep_alive=0, granularity=60.0) == 120.0


def test_cost_uses_the_published_rate_by_default():
    assert abs(model.cost_usd(3600.0) - 3.99) < 1e-9
    assert abs(model.cost_usd(3600.0, usd_per_second=1.89 / 3600) - 1.89) < 1e-9
