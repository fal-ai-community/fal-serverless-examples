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


def test_arrival_schedule_shapes():
    assert model.arrival_schedule("spike", requests=4, duration_s=10) == [0.0, 0.0, 0.0, 0.0]
    assert model.arrival_schedule("steady", requests=4, duration_s=10) == [0.0, 2.5, 5.0, 7.5]
    ramp = model.arrival_schedule("ramp", requests=4, duration_s=10)
    assert ramp[0] == 0.0
    # A ramp accelerates: later gaps are wider than earlier ones.
    gaps = [b - a for a, b in zip(ramp, ramp[1:])]
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]


def test_percentile_nearest_rank():
    assert model.percentile([1, 2, 3, 4, 5], 50) == 3
    assert model.percentile([1, 2, 3, 4, 5], 100) == 5
    assert model.percentile(list(range(1, 21)), 95) == 19
    assert model.percentile([7], 95) == 7


def test_cold_spike_every_request_waits_for_a_cold_start():
    # 4 requests at t=0, one runner each, 10s cold start, 2s service.
    result = model.simulate(
        model.arrival_schedule("spike", requests=4, duration_s=0),
        service_s=2.0, cold_start_s=10.0, keep_alive=0.0, max_concurrency=4,
    )
    assert len(result.waits) == 4, "requests are never dropped"
    assert all(abs(w - 10.0) < 0.2 for w in result.waits), result.waits
    assert len(result.windows) == 4
    # Each runner bills SETUP (10s) + its single 2s request, keep_alive=0.
    assert model.billable_seconds(result.windows, keep_alive=0.0) == 48.0


def test_max_concurrency_caps_runners_and_queues_the_rest():
    result = model.simulate(
        model.arrival_schedule("spike", requests=10, duration_s=0),
        service_s=1.0, cold_start_s=5.0, keep_alive=0.0, max_concurrency=2,
    )
    assert len(result.waits) == 10, "every request is eventually served"
    assert max(r["setup"] + r["idle"] + r["running"] for r in result.timeline) == 2
    # Ten requests through two runners at 1s each: the last waits ~4 rounds
    # on top of the 5s cold start.
    assert model.percentile(result.waits, 100) > 8.0


def test_min_concurrency_removes_the_cold_start_penalty():
    arrivals = model.arrival_schedule("spike", requests=2, duration_s=0)
    cold = model.simulate(arrivals, service_s=1.0, cold_start_s=20.0,
                          keep_alive=0.0, max_concurrency=2)
    warm = model.simulate(arrivals, service_s=1.0, cold_start_s=0.0,
                          keep_alive=0.0, min_concurrency=2, max_concurrency=2)
    assert model.percentile(cold.waits, 95) > 19.0
    assert model.percentile(warm.waits, 95) < 0.5


def test_scaling_delay_defers_the_scale_up():
    baseline = model.simulate(
        [0.0], service_s=1.0, cold_start_s=0.0, keep_alive=0.0,
        max_concurrency=1, scaling_delay=0.0,
    )
    result = model.simulate(
        [0.0], service_s=1.0, cold_start_s=0.0, keep_alive=0.0,
        max_concurrency=1, scaling_delay=5.0,
    )
    # The queued request must age scaling_delay seconds before it counts
    # toward the desired runner count.
    assert result.waits[0] >= 5.0
    # `waits[0] >= 5.0` alone would also pass if some unrelated bug
    # inflated every wait in the simulation, delayed or not. Pin the
    # delayed run's extra cost to roughly scaling_delay above a
    # scaling_delay=0.0 baseline, so only the delay mechanism itself can
    # satisfy it.
    assert abs((result.waits[0] - baseline.waits[0]) - 5.0) < 0.2


def test_keep_alive_keeps_the_second_burst_warm():
    # Two bursts 30s apart. keep_alive=60 spans the gap; keep_alive=5 does not.
    arrivals = [0.0, 30.0]
    warm = model.simulate(arrivals, service_s=1.0, cold_start_s=20.0,
                          keep_alive=60.0, max_concurrency=1)
    cold = model.simulate(arrivals, service_s=1.0, cold_start_s=20.0,
                          keep_alive=5.0, max_concurrency=1)
    assert warm.waits[1] < 1.0, "second request should hit a warm runner"
    assert cold.waits[1] > 15.0, "second request should pay another cold start"
    # And the warm run bills more idle time for the privilege.
    assert model.billable_seconds(warm.windows, keep_alive=60.0) > \
           model.billable_seconds(cold.windows, keep_alive=5.0)


def test_timeline_states_are_disjoint_and_bounded():
    result = model.simulate(
        model.arrival_schedule("ramp", requests=6, duration_s=20),
        service_s=2.0, cold_start_s=8.0, keep_alive=10.0, max_concurrency=3,
    )
    for row in result.timeline:
        assert row["setup"] >= 0 and row["idle"] >= 0 and row["running"] >= 0
        assert row["setup"] + row["idle"] + row["running"] <= 3
        assert row["queue_depth"] >= 0
    assert result.timeline[0]["t"] == 0.0
    assert result.timeline[-1]["queue_depth"] == 0, "the queue must drain"
    # The upper bound alone would pass for an all-zeros stub that never
    # reports a live runner at all; pin down that runners actually show up.
    assert any(row["setup"] + row["idle"] + row["running"] > 0 for row in result.timeline)


def test_derived_timestamps_stay_on_the_tick_grid_under_sequential_load():
    # Regression for un-rounded derived completion timestamps: a request's
    # completion time (t + service_s) is compared against the rounded tick
    # clock `t` without being rounded itself. Float error (0.1 + 0.2 != 0.3)
    # then makes a completion that should land exactly on a tick boundary
    # get recognised one tick late - and since the next completion is
    # computed from that already-late value, the error compounds rather
    # than cancelling.
    #
    # NOTE: this test only covers the *completion-time* _grid call in
    # simulate()'s dispatch loop. It cannot cover the other quantization
    # site, `ready_at = _grid(at + cold_start_s)` in start_runner, because
    # cold_start_s=0.0 here and every runner below is pre-started at
    # at=0.0: `0.0 + x == x` exactly for any float x, so `at + cold_start_s`
    # is an exact no-op regardless of quantization. See
    # test_ready_at_stays_on_the_tick_grid_when_a_runner_starts_mid_loop
    # for that site.
    #
    # service_s=0.2 is an exact multiple of the 0.1s tick, so a correctly
    # quantized simulation reproduces the exact-arithmetic wait for every
    # request (i * service_s, since all 20 requests arrive at t=0 and are
    # served strictly one after another by a single runner) within a
    # single tick, with no growth over the run. Integer-valued service_s
    # can't expose this: only non-integer ticks-per-service ratios collide
    # with float rounding.
    n = 20
    arrivals = [0.0] * n
    result = model.simulate(
        arrivals, service_s=0.2, cold_start_s=0.0, keep_alive=0.0,
        max_concurrency=1,
    )
    tolerance = model.TICK_S + 1e-6
    for i, wait in enumerate(result.waits):
        expected = i * 0.2
        assert abs(wait - expected) <= tolerance, (
            f"request {i}: wait={wait} expected~{expected} "
            f"(diff {wait - expected} exceeds one tick - drift is compounding)"
        )


def test_ready_at_stays_on_the_tick_grid_when_a_runner_starts_mid_loop():
    # Regression for the OTHER quantization site: `ready_at = _grid(at +
    # cold_start_s)` in start_runner. The sequential-load test above can't
    # exercise it: every runner it starts is pre-started at at=0.0, and
    # 0.0 + x == x exactly for any float x, so no float error can ever
    # arise there. Exposing this site requires a runner that starts
    # mid-loop at a non-zero tick, with a non-integer cold_start_s, so that
    # `at + cold_start_s` is a genuinely inexact sum.
    #
    # Two requests arrive at t=0.0 and t=0.1. service_s=5.0 is long enough
    # that the first runner is still mid-request when request 1 arrives,
    # so max_concurrency=2 forces a second runner to start at t=0.1 (a
    # non-zero tick) rather than reusing the first. cold_start_s=0.2 is
    # non-integer, and checked directly in the interpreter:
    #   >>> 0.1 + 0.2
    #   0.30000000000000004
    # which is a hair ABOVE the tick grid value at that point (exactly
    # 0.3, since t = round(step * tick_s, 6)). A correctly quantized
    # ready_at (_grid(0.1 + 0.2) == 0.3) satisfies `ready_at <= t` right
    # when t reaches 0.3, so request 1 - whose runner starts the instant
    # it arrives - waits exactly cold_start_s. An un-quantized ready_at
    # (0.30000000000000004) narrowly misses that same check at t=0.3, so
    # the runner isn't recognised as ready until t=0.4 - one whole tick
    # late, and request 1's wait grows to 0.3 instead of 0.2.
    #
    # Confirmed empirically against both variants of model.py: correctly
    # quantized -> waits == [0.2, ~0.2]; ready_at revert only -> waits ==
    # [0.2, ~0.3]; completion-time revert only -> waits == [0.2, ~0.2]
    # (unaffected - this scenario isolates the ready_at site).
    result = model.simulate(
        [0.0, 0.1], service_s=5.0, cold_start_s=0.2, keep_alive=0.0,
        max_concurrency=2,
    )
    assert len(result.waits) == 2
    # A one-tick tolerance would defeat the point of this test: the bug's
    # error IS exactly one tick (0.3 vs 0.2), so the check needs a tight
    # epsilon - not tick-scale slack - to tell "correct" from "broken".
    expected_wait = 0.2  # == cold_start_s; runner starts the instant the
    # request arrives, and 0.1 + 0.2 lands exactly on a tick boundary.
    assert abs(result.waits[1] - expected_wait) < 1e-6, (
        f"wait={result.waits[1]} expected~{expected_wait} "
        f"(diff {result.waits[1] - expected_wait} - ready_at is drifting "
        "off the tick grid)"
    )


def test_min_concurrency_prestart_never_exceeds_max_concurrency():
    # simulate() pre-starts min_concurrency runners before the loop, but
    # that pre-start must still respect the same max_concurrency ceiling
    # that desired_runners() enforces once the loop is running.
    result = model.simulate(
        [0.0], service_s=1.0, cold_start_s=1.0, keep_alive=0.0,
        min_concurrency=2, max_concurrency=1,
    )
    for row in result.timeline:
        assert row["setup"] + row["idle"] + row["running"] <= 1, row


def synthetic_trace(**overrides):
    """A hand-built trace with known-good arithmetic, for the model's readers.

    Two runners, both starting SETUP at t=0, service time exactly 4s:
      r-aaa: setup 30s -> ready at 30 -> serves req 0 (done 34), req 2 (done 38)
      r-bbb: setup 32s -> ready at 32 -> serves req 1 (done 36)
    All three requests arrive at t=0, so all three queue behind a cold start.
    """
    trace = {
        "schema": model.TRACE_SCHEMA,
        "recorded_at": "2026-07-28T14:03:11Z",
        "app": "someone/tune-autoscaling",
        "machine_type": "GPU-H100",
        "fal_version": "1.78.2",
        "rate_usd_per_second": model.USD_PER_SECOND,
        "scale_config": {
            "keep_alive": 300, "min_concurrency": 0, "max_concurrency": 4,
            "concurrency_buffer": 0, "concurrency_buffer_perc": 0,
            "scaling_delay": 0, "max_multiplexing": 1,
        },
        "burst": {"pattern": "spike", "requests": 3, "duration_s": 0.0, "concurrency": 3},
        "requests": [
            {"id": 0, "arrived_s": 0.0, "completed_s": 34.0, "status": 200,
             "served_by": "r-aaa", "was_cold": True, "setup_duration_s": 30.0},
            {"id": 1, "arrived_s": 0.0, "completed_s": 36.0, "status": 200,
             "served_by": "r-bbb", "was_cold": True, "setup_duration_s": 32.0},
            {"id": 2, "arrived_s": 0.0, "completed_s": 38.0, "status": 200,
             "served_by": "r-aaa", "was_cold": False, "setup_duration_s": 30.0},
        ],
    }
    trace.update(overrides)
    return trace


def test_calibration_reads_cold_start_from_the_app_self_report():
    calibration = model.calibrate(synthetic_trace())
    # nearest-rank p50 of setup_duration_s over the two cold requests,
    # i.e. percentile([30.0, 32.0], 50) -> 30.0, not an interpolated 31.0
    assert calibration.cold_start_s == 30.0
    # r-aaa completed at 34 then 38 back-to-back: one service time is 4s
    assert calibration.service_s == 4.0
    assert calibration.observed_runners == 2
    assert calibration.machine_type == "GPU-H100"
    assert calibration.arrivals == (0.0, 0.0, 0.0)
    assert calibration.scale_config["keep_alive"] == 300


def test_calibration_rejects_an_unknown_schema():
    import pytest

    with pytest.raises(ValueError, match="schema"):
        model.calibrate(synthetic_trace(schema="something/else@9"))


def test_calibration_rejects_a_trace_where_no_runner_served_twice():
    import pytest

    trace = synthetic_trace()
    # Drop r-aaa's second request: now every runner served exactly once, so
    # there is no back-to-back completion gap to read service time from.
    trace["requests"] = [r for r in trace["requests"] if r["id"] != 2]
    with pytest.raises(ValueError, match="two requests"):
        model.calibrate(trace)


def test_calibration_ignores_failed_requests():
    trace = synthetic_trace()
    trace["requests"].append(
        {"id": 3, "arrived_s": 0.0, "completed_s": 1.0, "status": 500,
         "served_by": "r-ccc", "was_cold": True, "setup_duration_s": 0.1}
    )
    calibration = model.calibrate(trace)
    assert calibration.observed_runners == 2, "a 5xx runner is not a data point"
    assert calibration.cold_start_s == 30.0


def test_measured_windows_derive_each_runner_billed_span():
    windows = {w.runner_id: w for w in model.measured_windows(synthetic_trace())}
    assert set(windows) == {"r-aaa", "r-bbb"}
    # r-aaa: first completion 34.0, its own setup 30.0, service 4.0
    # -> SETUP started at 34 - 30 - 4 = 0.0, which is when the burst began.
    assert windows["r-aaa"].setup_start_s == 0.0
    assert windows["r-aaa"].last_completion_s == 38.0
    # r-bbb: 36 - 32 - 4 = 0.0 as well; both booted on the same spike.
    assert windows["r-bbb"].setup_start_s == 0.0
    assert windows["r-bbb"].last_completion_s == 36.0


def test_measured_timeline_has_the_same_row_shape_as_the_simulation():
    timeline = model.measured_timeline(synthetic_trace(), tick_s=1.0)
    assert timeline, "a trace with requests must produce a timeline"
    assert set(timeline[0]) == {"t", "setup", "idle", "running", "queue_depth"}
    peak = max(r["setup"] + r["idle"] + r["running"] for r in timeline)
    assert peak == 2, "two distinct served_by values were observed"
    assert max(r["queue_depth"] for r in timeline) == 3, "all three queued at t=0"
    assert timeline[-1]["queue_depth"] == 0, "the queue drains"
