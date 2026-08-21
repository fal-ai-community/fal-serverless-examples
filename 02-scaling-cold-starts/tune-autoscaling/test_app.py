"""Unit tests for the autoscaling model, plus live tests against an ephemeral deploy."""

import importlib.util
import itertools
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "autoscale_model", Path(__file__).parent / "model.py"
)
assert _spec is not None and _spec.loader is not None
model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = model
_spec.loader.exec_module(model)


# Labelled bar values from the three diagrams on
# docs.fal.ai/documentation/deployment/scale-your-application - fal's only
# published ground truth for the autoscaler, hence the oracle.
PUBLISHED_CHARTS = [
    # (config, expected runners for ongoing = 0, 1, 2, ...)
    (
        {
            "min_concurrency": 1,
            "concurrency_buffer": 1,
            "max_multiplexing": 1,
            "max_concurrency": 10,
        },
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10],
    ),
    (
        {
            "min_concurrency": 3,
            "concurrency_buffer": 2,
            "max_multiplexing": 1,
            "max_concurrency": 10,
        },
        [3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10],
    ),
    (
        {
            "min_concurrency": 0,
            "concurrency_buffer": 2,
            "max_multiplexing": 4,
            "max_concurrency": 6,
        },
        [2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6],
    ),
]


def test_law_matches_every_published_diagram():
    for config, expected in PUBLISHED_CHARTS:
        actual = [model.desired_runners(n, **config) for n in range(len(expected))]
        assert actual == expected, f"config {config}: {actual} != {expected}"


def test_buffer_percentage_is_a_floor_not_a_replacement():
    # Docs: the buffer is max(concurrency_buffer, perc / 100 * volume).
    assert (
        model.desired_runners(10, concurrency_buffer=5, concurrency_buffer_perc=20)
        == 15
    )
    assert (
        model.desired_runners(100, concurrency_buffer=5, concurrency_buffer_perc=20)
        == 120
    )


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
    # SETUP at t=0, last done t=40, keep_alive=60 -> alive 0..100 -> 100 billed s.
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
    # Granularity is published nowhere; the per-runner round-up is an assumption.
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
    assert model.arrival_schedule("spike", requests=4, duration_s=10) == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert model.arrival_schedule("steady", requests=4, duration_s=10) == [
        0.0,
        2.5,
        5.0,
        7.5,
    ]
    ramp = model.arrival_schedule("ramp", requests=4, duration_s=10)
    assert ramp[0] == 0.0
    gaps = [b - a for a, b in itertools.pairwise(ramp)]
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]


def test_percentile_nearest_rank():
    assert model.percentile([1, 2, 3, 4, 5], 50) == 3
    assert model.percentile([1, 2, 3, 4, 5], 100) == 5
    assert model.percentile(list(range(1, 21)), 95) == 19
    assert model.percentile([7], 95) == 7


def test_percentile_nearest_rank_pins_even_length_p50():
    # Nearest-rank picks the lower middle (30.0), deliberately not the
    # interpolated median (31.0) - the one estimator used everywhere.
    assert model.percentile([30.0, 32.0], 50) == 30.0


def test_cold_spike_every_request_waits_for_a_cold_start():
    result = model.simulate(
        model.arrival_schedule("spike", requests=4, duration_s=0),
        service_s=2.0,
        cold_start_s=10.0,
        keep_alive=0.0,
        max_concurrency=4,
    )
    assert len(result.waits) == 4, "requests are never dropped"
    assert all(abs(w - 10.0) < 0.2 for w in result.waits), result.waits
    assert len(result.windows) == 4
    # Each runner bills SETUP (10s) + its single 2s request, keep_alive=0.
    assert model.billable_seconds(result.windows, keep_alive=0.0) == 48.0


def test_max_concurrency_caps_runners_and_queues_the_rest():
    result = model.simulate(
        model.arrival_schedule("spike", requests=10, duration_s=0),
        service_s=1.0,
        cold_start_s=5.0,
        keep_alive=0.0,
        max_concurrency=2,
    )
    assert len(result.waits) == 10, "every request is eventually served"
    assert max(r["setup"] + r["idle"] + r["running"] for r in result.timeline) == 2
    # Ten requests through two 1s runners: the last waits ~4s beyond the 5s cold start.
    assert model.percentile(result.waits, 100) > 8.0


def test_min_concurrency_removes_the_cold_start_penalty():
    arrivals = model.arrival_schedule("spike", requests=2, duration_s=0)
    cold = model.simulate(
        arrivals, service_s=1.0, cold_start_s=20.0, keep_alive=0.0, max_concurrency=2
    )
    # A real cold_start_s pins the claim: prewarmed runners are already READY.
    warm = model.simulate(
        arrivals,
        service_s=1.0,
        cold_start_s=20.0,
        keep_alive=0.0,
        min_concurrency=2,
        max_concurrency=2,
    )
    assert model.percentile(cold.waits, 95) > 19.0
    assert model.percentile(warm.waits, 95) < 0.5


def test_scaling_delay_defers_the_scale_up():
    baseline = model.simulate(
        [0.0],
        service_s=1.0,
        cold_start_s=0.0,
        keep_alive=0.0,
        max_concurrency=1,
        scaling_delay=0.0,
    )
    result = model.simulate(
        [0.0],
        service_s=1.0,
        cold_start_s=0.0,
        keep_alive=0.0,
        max_concurrency=1,
        scaling_delay=5.0,
    )
    # The queued request must age scaling_delay before it counts toward demand.
    assert result.waits[0] >= 5.0
    # Pin the delta against a zero-delay baseline so only the delay mechanism passes.
    assert abs((result.waits[0] - baseline.waits[0]) - 5.0) < 0.2


def test_keep_alive_keeps_the_second_burst_warm():
    # Two bursts 30s apart. keep_alive=60 spans the gap; keep_alive=5 does not.
    arrivals = [0.0, 30.0]
    warm = model.simulate(
        arrivals, service_s=1.0, cold_start_s=20.0, keep_alive=60.0, max_concurrency=1
    )
    cold = model.simulate(
        arrivals, service_s=1.0, cold_start_s=20.0, keep_alive=5.0, max_concurrency=1
    )
    assert warm.waits[1] < 1.0, "second request should hit a warm runner"
    assert cold.waits[1] > 15.0, "second request should pay another cold start"
    assert model.billable_seconds(
        warm.windows, keep_alive=60.0
    ) > model.billable_seconds(cold.windows, keep_alive=5.0)


def test_timeline_states_are_disjoint_and_bounded():
    result = model.simulate(
        model.arrival_schedule("ramp", requests=6, duration_s=20),
        service_s=2.0,
        cold_start_s=8.0,
        keep_alive=10.0,
        max_concurrency=3,
    )
    for row in result.timeline:
        assert row["setup"] >= 0 and row["idle"] >= 0 and row["running"] >= 0
        assert row["setup"] + row["idle"] + row["running"] <= 3
        assert row["queue_depth"] >= 0
    assert result.timeline[0]["t"] == 0.0
    assert result.timeline[-1]["queue_depth"] == 0, "the queue must drain"
    # Guard against an all-zeros stub: runners must actually appear.
    assert any(
        row["setup"] + row["idle"] + row["running"] > 0 for row in result.timeline
    )


def test_derived_timestamps_stay_on_the_tick_grid_under_sequential_load():
    # Regression: derived completion times (t + service_s) must be snapped to
    # the tick clock or float error compounds a tick late. Covers only the
    # completion-time snap site - at=0.0 and cold_start_s=0.0 make the
    # ready_at sum exact (the next test covers that site). service_s=0.2
    # matters: only non-integer ticks-per-service collides with float rounding.
    n = 20
    arrivals = [0.0] * n
    result = model.simulate(
        arrivals,
        service_s=0.2,
        cold_start_s=0.0,
        keep_alive=0.0,
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
    # Regression for the other snap site: ready_at = at + cold_start_s. A
    # second runner starts mid-loop at t=0.1 with cold_start_s=0.2, and
    # 0.1 + 0.2 = 0.30000000000000004 - a hair above the tick grid - so an
    # unquantized ready_at is recognised one whole tick late.
    result = model.simulate(
        [0.0, 0.1],
        service_s=5.0,
        cold_start_s=0.2,
        keep_alive=0.0,
        max_concurrency=2,
    )
    assert len(result.waits) == 2
    # The bug's error IS exactly one tick, so the check needs a tight epsilon.
    expected_wait = 0.2  # == cold_start_s; the runner starts the instant it arrives
    assert abs(result.waits[1] - expected_wait) < 1e-6, (
        f"wait={result.waits[1]} expected~{expected_wait} "
        f"(diff {result.waits[1] - expected_wait} - ready_at is drifting "
        "off the tick grid)"
    )


def test_min_concurrency_prestart_never_exceeds_max_concurrency():
    result = model.simulate(
        [0.0],
        service_s=1.0,
        cold_start_s=1.0,
        keep_alive=0.0,
        min_concurrency=2,
        max_concurrency=1,
    )
    for row in result.timeline:
        assert row["setup"] + row["idle"] + row["running"] <= 1, row


def test_prewarmed_runner_is_never_torn_down_even_when_it_goes_idle_first():
    # The prewarmed runner holds the floor and must never be the scale-down
    # victim. The late t=10 request breaks a coincidental last_completion_s
    # tie a two-request version of this scenario would have.
    keep_alive = 1.0
    result = model.simulate(
        [0.0, 0.0, 10.0],
        service_s=1.0,
        cold_start_s=5.0,
        keep_alive=keep_alive,
        min_concurrency=1,
        max_concurrency=2,
    )
    assert result.waits[2] == 0.0, (
        "the third request must be served warm by the survivor"
    )
    prewarmed = [w for w in result.windows if w.prewarmed]
    assert len(prewarmed) == 1, "min_concurrency=1 prestarts exactly one floor runner"

    # The window with the latest last_completion_s is the one still alive at the end.
    survivor = max(result.windows, key=lambda w: w.last_completion_s)
    assert survivor.prewarmed, (
        "the runner still alive at the end of the simulation must be the "
        "prewarmed one - it holds the min_concurrency floor - not the "
        "runner spun up to handle this burst"
    )


def synthetic_trace(**overrides):
    """Hand-built trace: two cold runners at t=0; req 2 warm on r-aaa back-to-back."""
    # req 2 arrives at t=34 - the first instant a runner is free: any earlier
    # forces a third runner, and no stagger at all would leave no back-to-back
    # pair to measure service_s from.
    trace = {
        "schema": model.TRACE_SCHEMA,
        "recorded_at": "2026-07-28T14:03:11Z",
        "app": "someone/tune-autoscaling",
        "machine_type": "GPU-H100",
        "fal_version": "1.78.2",
        "rate_usd_per_second": model.USD_PER_SECOND,
        "scale_config": {
            "keep_alive": 300,
            "min_concurrency": 0,
            "max_concurrency": 4,
            "concurrency_buffer": 0,
            "concurrency_buffer_perc": 0,
            "scaling_delay": 0,
            "max_multiplexing": 1,
        },
        "burst": {
            "pattern": "custom",
            "requests": 3,
            "duration_s": 34.0,
            "concurrency": 3,
        },
        "requests": [
            {
                "id": 0,
                "arrived_s": 0.0,
                "completed_s": 34.0,
                "status": 200,
                "served_by": "r-aaa",
                "was_cold": True,
                "setup_duration_s": 30.0,
            },
            {
                "id": 1,
                "arrived_s": 0.0,
                "completed_s": 36.0,
                "status": 200,
                "served_by": "r-bbb",
                "was_cold": True,
                "setup_duration_s": 32.0,
            },
            {
                "id": 2,
                "arrived_s": 34.0,
                "completed_s": 38.0,
                "status": 200,
                "served_by": "r-aaa",
                "was_cold": False,
                "setup_duration_s": 30.0,
            },
        ],
    }
    trace.update(overrides)
    return trace


def test_calibration_reads_cold_start_from_the_app_self_report():
    calibration = model.calibrate(synthetic_trace())
    # nearest-rank p50 of [30.0, 32.0] -> 30.0, not an interpolated 31.0
    assert calibration.cold_start_s == 30.0
    # r-aaa completed at 34 then 38 back-to-back: one service time is 4s
    assert calibration.service_s == 4.0
    assert calibration.observed_runners == 2
    assert calibration.machine_type == "GPU-H100"
    assert calibration.arrivals == (0.0, 0.0, 34.0)
    assert calibration.scale_config["keep_alive"] == 300


def test_calibration_rejects_an_unknown_schema():
    with pytest.raises(ValueError, match="schema"):
        model.calibrate(synthetic_trace(schema="something/else@9"))


def test_calibration_rejects_a_trace_where_no_runner_served_twice():
    trace = synthetic_trace()
    # Drop r-aaa's second request: no back-to-back gap remains.
    trace["requests"] = [r for r in trace["requests"] if r["id"] != 2]
    with pytest.raises(ValueError, match="two requests"):
        model.calibrate(trace)


def test_calibration_ignores_failed_requests():
    trace = synthetic_trace()
    trace["requests"].append(
        {
            "id": 3,
            "arrived_s": 0.0,
            "completed_s": 1.0,
            "status": 500,
            "served_by": "r-ccc",
            "was_cold": True,
            "setup_duration_s": 0.1,
        }
    )
    calibration = model.calibrate(trace)
    assert calibration.observed_runners == 2, "a 5xx runner is not a data point"
    assert calibration.cold_start_s == 30.0


def test_measured_windows_derive_each_runner_billed_span():
    windows = {w.runner_id: w for w in model.measured_windows(synthetic_trace())}
    assert set(windows) == {"r-aaa", "r-bbb"}
    # r-aaa: 34 - 30 - 4 = 0.0 -> SETUP began with the burst.
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
    # Queued spans [arrived_s, completed_s - service_s): req 0 -> [0, 30),
    # req 1 -> [0, 32), req 2 -> empty (served the instant it arrives).
    assert max(r["queue_depth"] for r in timeline) == 2
    assert timeline[-1]["queue_depth"] == 0, "the queue drains"


def test_measured_timeline_spans_the_full_keep_alive_tail():
    trace = synthetic_trace()
    keep_alive = trace["scale_config"]["keep_alive"]
    last_completion = max(r["completed_s"] for r in trace["requests"])  # 38.0

    timeline = model.measured_timeline(trace, tick_s=1.0)

    assert timeline[-1]["t"] >= last_completion + keep_alive, (
        "the timeline must keep emitting rows through the keep_alive idle "
        "tail instead of stopping at the last completion"
    )
    # Inside the keep_alive window both runners must still be alive (idle).
    well_after = next(row for row in timeline if row["t"] == last_completion + 100.0)
    assert well_after["idle"] >= 1


def test_calibration_excludes_requests_with_no_usable_runner_identity():
    # served_by == "" has no runner identity and must not become a phantom runner.
    baseline = model.calibrate(synthetic_trace())

    trace = synthetic_trace()
    trace["requests"] = trace["requests"] + [
        {
            "id": 3,
            "arrived_s": 0.0,
            "completed_s": 1.0,
            "status": 200,
            "served_by": "",
            "was_cold": False,
            "setup_duration_s": 0.0,
        },
    ]
    calibration = model.calibrate(trace)

    assert calibration.observed_runners == baseline.observed_runners
    assert calibration.service_s == baseline.service_s


def test_measured_windows_do_not_depend_on_a_runners_request_order():
    # _by_runner sorts only the outer dict; per-runner request order is arbitrary.
    baseline = {w.runner_id: w for w in model.measured_windows(synthetic_trace())}

    reordered = synthetic_trace()
    requests = reordered["requests"]
    r_aaa = [r for r in requests if r["served_by"] == "r-aaa"]
    others = [r for r in requests if r["served_by"] != "r-aaa"]
    assert len(r_aaa) == 2, "fixture must still have two requests for r-aaa"
    reordered["requests"] = others + list(reversed(r_aaa))

    reordered_windows = {w.runner_id: w for w in model.measured_windows(reordered)}
    assert reordered_windows["r-aaa"].setup_start_s == baseline["r-aaa"].setup_start_s
    assert reordered_windows["r-bbb"].setup_start_s == baseline["r-bbb"].setup_start_s


def test_sweep_covers_the_default_grid_and_flags_the_measured_point():
    points = model.sweep(model.calibrate(synthetic_trace()))
    expected = (
        len(model.DEFAULT_GRID["keep_alive"])
        * len(model.DEFAULT_GRID["min_concurrency"])
        * len(model.DEFAULT_GRID["concurrency_buffer"])
    )
    assert len(points) == expected
    measured = [p for p in points if p["measured"]]
    assert len(measured) == 1, "exactly one grid point matches the trace's own config"
    assert measured[0]["keep_alive"] == 300
    assert measured[0]["min_concurrency"] == 0
    assert measured[0]["concurrency_buffer"] == 0


def test_sweep_points_carry_every_documented_key():
    point = model.sweep(model.calibrate(synthetic_trace()))[0]
    assert set(point) == {
        "keep_alive",
        "min_concurrency",
        "concurrency_buffer",
        "concurrency_buffer_perc",
        "max_concurrency",
        "max_multiplexing",
        "p50_wait_s",
        "p95_wait_s",
        "billable_s",
        "cost_usd",
        "peak_runners",
        "validated",
        "measured",
    }
    assert len(point) == 13


def test_warm_configs_cost_more_and_wait_less():
    points = {
        (p["keep_alive"], p["min_concurrency"], p["concurrency_buffer"]): p
        for p in model.sweep(model.calibrate(synthetic_trace()))
    }
    scale_to_zero = points[(0, 0, 0)]
    always_warm = points[(300, 2, 0)]
    assert always_warm["p95_wait_s"] < scale_to_zero["p95_wait_s"]
    assert always_warm["cost_usd"] > scale_to_zero["cost_usd"]


def test_min_concurrency_floor_is_billed_across_the_default_horizon():
    # Even the cheapest combo at min_concurrency=2 must clear the 2 * 3600s floor.
    points = {
        (p["keep_alive"], p["min_concurrency"], p["concurrency_buffer"]): p
        for p in model.sweep(model.calibrate(synthetic_trace()))
    }
    point = points[(0, 2, 0)]
    assert point["billable_s"] >= 2 * 3600.0


def test_scale_to_zero_is_the_cheapest_point_on_the_fixture():
    # The central regression: burst-window floor costing made scale-to-zero
    # look dominated and the frontier advised keeping runners warm.
    calibration = model.calibrate(synthetic_trace())
    points = model.sweep(calibration)
    cheapest = min(points, key=lambda p: p["cost_usd"])
    assert cheapest["min_concurrency"] == 0
    assert cheapest["keep_alive"] == 0
    cheapest_index = next(i for i, p in enumerate(points) if p is cheapest)
    assert cheapest_index in model.pareto_front(points)


def test_min_concurrency_one_does_not_double_count_the_prewarmed_window():
    # Prewarmed runners bill via the floor only - assert the exact
    # floor + demand total, not just an inequality.
    calibration = model.calibrate(synthetic_trace())
    base = calibration.scale_config
    keep_alive = 0.0
    min_concurrency = 1
    concurrency_buffer = 0
    horizon_s = 3600.0
    bursts_per_hour = 1.0

    result = model.simulate(
        list(calibration.arrivals),
        service_s=calibration.service_s,
        cold_start_s=calibration.cold_start_s,
        keep_alive=keep_alive,
        min_concurrency=min_concurrency,
        max_concurrency=base.get("max_concurrency"),
        concurrency_buffer=concurrency_buffer,
        concurrency_buffer_perc=int(base.get("concurrency_buffer_perc", 0)),
        max_multiplexing=int(base.get("max_multiplexing", 1)),
        scaling_delay=float(base.get("scaling_delay", 0)),
    )
    demand_windows = [w for w in result.windows if not w.prewarmed]
    assert len(demand_windows) < len(result.windows), (
        "fixture must actually produce a prewarmed window, or this test "
        "isn't exercising the double-counting guard at all"
    )
    demand_billable = model.billable_seconds(
        demand_windows, keep_alive=keep_alive, granularity=1.0
    )
    floor_s = min_concurrency * horizon_s
    bursts_in_horizon = bursts_per_hour * horizon_s / 3600.0
    expected_total = floor_s + demand_billable * bursts_in_horizon

    points = {
        (p["keep_alive"], p["min_concurrency"], p["concurrency_buffer"]): p
        for p in model.sweep(
            calibration, horizon_s=horizon_s, bursts_per_hour=bursts_per_hour
        )
    }
    point = points[(int(keep_alive), min_concurrency, concurrency_buffer)]
    assert point["billable_s"] == round(expected_total, 3)

    # Double-counting (prewarmed window billed atop the floor) must cost more.
    double_counted = (
        floor_s
        + model.billable_seconds(result.windows, keep_alive=keep_alive, granularity=1.0)
        * bursts_in_horizon
    )
    assert point["billable_s"] < double_counted


def test_horizon_scales_the_always_on_floor():
    calibration = model.calibrate(synthetic_trace())
    short = {
        (p["keep_alive"], p["min_concurrency"], p["concurrency_buffer"]): p
        for p in model.sweep(calibration, horizon_s=3600.0)
    }
    long = {
        (p["keep_alive"], p["min_concurrency"], p["concurrency_buffer"]): p
        for p in model.sweep(calibration, horizon_s=7200.0)
    }
    # min_concurrency=2 pays a floor: doubling the horizon must cost more.
    assert long[(0, 2, 0)]["cost_usd"] > short[(0, 2, 0)]["cost_usd"]
    # No floor at min_concurrency=0: billable_s must double exactly with the horizon.
    assert long[(0, 0, 0)]["billable_s"] == pytest.approx(
        2 * short[(0, 0, 0)]["billable_s"]
    )


def test_cost_usd_is_recomputable_from_the_reported_billable_s():
    # cost_usd must derive from the rounded billable_s a reader sees, or hand-
    # multiplying the printed numbers mismatches in the last decimal. A
    # fractional bursts_per_hour makes demand_s fractional enough to exercise
    # the rounding order.
    calibration = model.calibrate(synthetic_trace())
    rate = 0.00095
    for point in model.sweep(calibration, usd_per_second=rate, bursts_per_hour=2 / 3):
        assert point["cost_usd"] == round(point["billable_s"] * rate, 6)


def test_points_beyond_the_measured_envelope_are_flagged_unvalidated():
    # 2 runners were observed; a config needing more is extrapolation.
    points = model.sweep(
        model.calibrate(synthetic_trace()),
        grid={
            "keep_alive": (0,),
            "min_concurrency": (0, 4),
            "concurrency_buffer": (0,),
        },
    )
    by_min = {p["min_concurrency"]: p for p in points}
    assert by_min[0]["validated"] is True
    assert by_min[4]["validated"] is False


def test_pareto_front_keeps_only_non_dominated_points():
    points = [
        {"cost_usd": 1.0, "p95_wait_s": 10.0},  # 0 cheap and slow  -> on front
        {"cost_usd": 2.0, "p95_wait_s": 5.0},  # 1 mid/mid         -> on front
        {"cost_usd": 3.0, "p95_wait_s": 6.0},  # 2 dominated by 1
        {"cost_usd": 4.0, "p95_wait_s": 1.0},  # 3 dear and fast   -> on front
    ]
    assert model.pareto_front(points) == [0, 1, 3]


def test_build_frontier_is_deterministic_and_carries_no_wall_clock():
    trace = synthetic_trace()
    first = model.build_frontier(trace)
    second = model.build_frontier(trace)
    assert first == second, "regenerating must be byte-identical"
    assert first["schema"] == model.FRONTIER_SCHEMA
    assert first["trace_recorded_at"] == trace["recorded_at"]
    assert "generated_at" not in first, (
        "a wall-clock stamp would break CI byte-identity"
    )
    assert first["calibration"]["cold_start_s"] == 30.0
    assert first["rate_usd_per_second"] == model.USD_PER_SECOND
    assert first["pareto_front"] == model.pareto_front(first["points"])
    assert first["horizon_s"] == 3600.0
    assert first["bursts_per_hour"] == 1.0


def test_cli_writes_a_frontier_file(tmp_path):
    trace_path = tmp_path / "trace.json"
    out_path = tmp_path / "frontier.json"
    trace_path.write_text(json.dumps(synthetic_trace()))
    exit_code = model.main(
        [
            "--calibrate",
            str(trace_path),
            "--rate",
            "0.001",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    written = json.loads(out_path.read_text())
    assert written["rate_usd_per_second"] == 0.001
    assert out_path.read_text().endswith("\n")


def test_sweep_flag_no_longer_exists():
    with pytest.raises(SystemExit):
        model.main(["--calibrate", "irrelevant.json", "--sweep"])


def test_main_reports_a_missing_trace_file_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.json"
    exit_code = model.main(["--calibrate", str(missing)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "no partial output once loading has failed"
    lines = captured.err.strip("\n").splitlines()
    assert len(lines) == 1, f"expected one clean error line, got: {captured.err!r}"
    assert str(missing) in lines[0]


def test_main_reports_malformed_json_without_a_traceback(tmp_path, capsys):
    bad = tmp_path / "trace.json"
    bad.write_text("{not valid json")
    exit_code = model.main(["--calibrate", str(bad)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip("\n").splitlines()
    assert len(lines) == 1, f"expected one clean error line, got: {captured.err!r}"
    assert str(bad) in lines[0]


def test_main_reports_a_wrong_schema_trace_without_a_traceback(tmp_path, capsys):
    wrong = tmp_path / "trace.json"
    wrong.write_text(json.dumps(synthetic_trace(schema="something/else@9")))
    exit_code = model.main(["--calibrate", str(wrong)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.strip("\n").splitlines()
    assert len(lines) == 1, f"expected one clean error line, got: {captured.err!r}"
    assert str(wrong) in lines[0]


# --- Live tests: real deploy; app_client skips when HARNESS_APP_URL is unset ---

PROMPT = "a calico cat riding a bicycle through a sunlit wheat field, film photo"


def test_generates_an_image_and_echoes_the_seed(app_client):
    response = app_client.run("/", {"prompt": PROMPT, "seed": 42})
    assert response["image"]["url"].startswith("http")
    assert response["seed"] == 42


def test_reports_which_runner_served_it(app_client):
    response = app_client.run("/", {"prompt": PROMPT})
    # burst.py counts distinct runners by this field alone - its shape is load-bearing.
    assert isinstance(response["served_by"], str)
    assert len(response["served_by"]) == 12
    assert response["setup_duration_s"] > 0, "setup() must time itself"


def test_sequential_requests_land_on_the_same_warm_runner(app_client):
    # min_concurrency=0 and concurrency_buffer=0 pin exactly one runner, which
    # is what lets distinct served_by values be read as a runner count.
    first = app_client.run("/", {"prompt": PROMPT})
    second = app_client.run("/", {"prompt": PROMPT})
    assert first["served_by"] == second["served_by"]
    assert first["setup_duration_s"] == second["setup_duration_s"]


def test_warm_requests_are_not_flagged_cold(app_client):
    # Warm via the harness's smoke request AND the tests above; keep at least
    # one of the two ahead of this test, or it silently tests nothing.
    response = app_client.run("/", {"prompt": PROMPT})
    assert response["was_cold"] is False


_burst_spec = importlib.util.spec_from_file_location(
    "autoscale_burst", Path(__file__).parent / "burst.py"
)
assert _burst_spec is not None and _burst_spec.loader is not None
burst = importlib.util.module_from_spec(_burst_spec)
_burst_spec.loader.exec_module(burst)


APPS_DOCUMENT = {
    "apps": [
        {
            "alias": "something-else",
            "machine_types": ["GPU-A100"],
            "keep_alive": 60,
            "min_concurrency": 0,
            "max_concurrency": 2,
            "concurrency_buffer": 0,
            "concurrency_buffer_perc": 0,
            "scaling_delay": 0,
            "max_multiplexing": 1,
        },
        {
            "alias": "tune-autoscaling",
            "machine_types": ["GPU-H100"],
            "keep_alive": 300,
            "min_concurrency": 0,
            "max_concurrency": 4,
            "concurrency_buffer": 0,
            "concurrency_buffer_perc": 0,
            "scaling_delay": 0,
            "max_multiplexing": 1,
        },
    ]
}


def test_select_app_config_finds_the_alias():
    config = burst.select_app_config(APPS_DOCUMENT, "someone/tune-autoscaling")
    assert config["keep_alive"] == 300
    assert config["max_concurrency"] == 4


def test_select_app_config_rejects_an_unknown_app():
    with pytest.raises(ValueError, match="not deployed"):
        burst.select_app_config(APPS_DOCUMENT, "someone/nope")


def test_assert_machine_accepts_the_pinned_machine():
    burst.assert_machine({"machine_types": ["GPU-H100"]})


def test_assert_machine_rejects_a_fallback_or_wrong_machine():
    with pytest.raises(RuntimeError, match="GPU-H100"):
        burst.assert_machine({"machine_types": ["GPU-L40"]})
    with pytest.raises(RuntimeError, match="GPU-H100"):
        burst.assert_machine({"machine_types": ["GPU-L40", "GPU-H100"]})


def test_build_trace_round_trips_through_the_model():
    """Producer/consumer agreement: build_trace writes what model.calibrate reads."""
    records = [
        {
            "id": 0,
            "arrived_s": 0.0,
            "completed_s": 32.0,
            "status": 200,
            "served_by": "r-aaa",
            "was_cold": True,
            "setup_duration_s": 30.0,
        },
        {
            "id": 1,
            "arrived_s": 0.0,
            "completed_s": 36.0,
            "status": 200,
            "served_by": "r-aaa",
            "was_cold": False,
            "setup_duration_s": 30.0,
        },
    ]
    trace = burst.build_trace(
        app="someone/tune-autoscaling",
        config=APPS_DOCUMENT["apps"][1],
        pattern="spike",
        requests=2,
        duration_s=0.0,
        concurrency=2,
        records=records,
        usd_per_second=model.USD_PER_SECOND,
        recorded_at="2026-07-28T14:03:11Z",
    )
    assert trace["schema"] == model.TRACE_SCHEMA
    assert trace["machine_type"] == "GPU-H100"
    assert trace["scale_config"]["keep_alive"] == 300
    calibration = model.calibrate(trace)  # must not raise
    assert calibration.cold_start_s == 30.0
    # r-aaa completed at 32.0 then 36.0 back-to-back: service time is 4s
    assert calibration.service_s == 4.0
    assert calibration.observed_runners == 1


# --- burst.py failure paths ---

GPU_H100_CONFIG = {
    "machine_types": ["GPU-H100"],
    "keep_alive": 300,
    "min_concurrency": 0,
    "max_concurrency": 4,
    "concurrency_buffer": 0,
    "concurrency_buffer_perc": 0,
    "scaling_delay": 0,
    "max_multiplexing": 1,
}


def calibratable_records():
    """Two runners; r-aaa serves twice back-to-back so service_s is derivable."""
    return [
        {
            "id": 0,
            "arrived_s": 0.0,
            "completed_s": 34.0,
            "status": 200,
            "served_by": "r-aaa",
            "was_cold": True,
            "setup_duration_s": 30.0,
        },
        {
            "id": 1,
            "arrived_s": 0.0,
            "completed_s": 36.0,
            "status": 200,
            "served_by": "r-bbb",
            "was_cold": True,
            "setup_duration_s": 32.0,
        },
        {
            "id": 2,
            "arrived_s": 34.0,
            "completed_s": 38.0,
            "status": 200,
            "served_by": "r-aaa",
            "was_cold": False,
            "setup_duration_s": 30.0,
        },
    ]


def uncalibratable_records():
    """Each runner serves exactly once: no back-to-back gap, so calibrate must raise."""
    return [
        {
            "id": 0,
            "arrived_s": 0.0,
            "completed_s": 34.0,
            "status": 200,
            "served_by": "r-aaa",
            "was_cold": True,
            "setup_duration_s": 30.0,
        },
        {
            "id": 1,
            "arrived_s": 0.0,
            "completed_s": 36.0,
            "status": 200,
            "served_by": "r-bbb",
            "was_cold": True,
            "setup_duration_s": 32.0,
        },
    ]


def test_cross_check_analytics_cannot_raise_on_a_transport_error(monkeypatch):
    # A bare URLError (not an HTTPError) must come back as a reported failure.
    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("dns boom")

    monkeypatch.setattr(burst.urllib.request, "urlopen", raise_url_error)
    result = burst.cross_check_analytics("someone/tune-autoscaling", {})
    assert result["ok"] is False
    assert "URLError" in result.get("error", "")
    assert "dns boom" in result.get("error", "")


def test_cross_check_analytics_still_reports_403_with_its_hint(monkeypatch):
    # The existing HTTPError/403 branch must survive the broadened handler.
    def raise_403(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://api.fal.ai/v1/serverless/analytics",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]  # stub wants Message
            fp=None,
        )

    monkeypatch.setattr(burst.urllib.request, "urlopen", raise_403)
    result = burst.cross_check_analytics("someone/tune-autoscaling", {})
    assert result["ok"] is False
    assert result["status"] == 403
    assert "Key-format" in result["hint"]


def test_main_saves_the_trace_even_when_the_cross_check_is_broken(
    tmp_path, monkeypatch
):
    # The trace must reach disk even when the cross-check's transport fails.
    # No network or CLI may be touched here.
    monkeypatch.setattr(burst, "read_live_config", lambda app: dict(GPU_H100_CONFIG))
    monkeypatch.setattr(burst, "run_burst", lambda *a, **k: calibratable_records())
    monkeypatch.setattr(burst, "_auth_headers", lambda: {})

    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("dns boom")

    monkeypatch.setattr(burst.urllib.request, "urlopen", raise_url_error)

    out = tmp_path / "trace.json"
    exit_code = burst.main(
        [
            "--app",
            "someone/tune-autoscaling",
            "--out",
            str(out),
            "--from-analytics",
        ]
    )

    assert exit_code == 0
    assert out.exists(), "trace.json must be written even though the cross-check failed"
    trace = json.loads(out.read_text())
    assert trace["schema"] == model.TRACE_SCHEMA
    assert len(trace["requests"]) == 3
    assert trace["analytics_cross_check"]["ok"] is False


def test_main_reports_an_uncalibratable_trace_without_a_traceback(
    tmp_path, monkeypatch, capsys
):
    # Easy to hit on a real burst: --concurrency >= --requests on a spike.
    monkeypatch.setattr(burst, "read_live_config", lambda app: dict(GPU_H100_CONFIG))
    monkeypatch.setattr(burst, "run_burst", lambda *a, **k: uncalibratable_records())
    monkeypatch.setattr(burst, "_auth_headers", lambda: {})

    out = tmp_path / "trace.json"
    exit_code = burst.main(
        [
            "--app",
            "someone/tune-autoscaling",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 1
    assert out.exists(), "the trace must remain saved even though calibration failed"
    trace = json.loads(out.read_text())
    assert trace["schema"] == model.TRACE_SCHEMA

    captured = capsys.readouterr()
    assert str(out) in captured.err, "the diagnostic must name the file that WAS saved"
    assert "two requests" in captured.err, "must quote calibrate()'s underlying message"
    assert "Traceback" not in captured.err


def test_read_live_config_reports_the_captured_stderr_on_a_non_zero_exit(monkeypatch):
    # CalledProcessError's default string hides the stderr holding the real reason.
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "Error: not logged in. Run `fal auth login` first.\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    with pytest.raises(RuntimeError, match="not logged in"):
        burst.read_live_config("someone/tune-autoscaling")


_chart_spec = importlib.util.spec_from_file_location(
    "autoscale_chart", Path(__file__).parent / "chart.py"
)
assert _chart_spec is not None and _chart_spec.loader is not None
chart = importlib.util.module_from_spec(_chart_spec)
_chart_spec.loader.exec_module(chart)

HERE = Path(__file__).parent


def _artifacts():
    if not (HERE / "trace.json").exists() or not (HERE / "frontier.json").exists():
        pytest.skip("trace.json / frontier.json not recorded yet")
    return (
        json.loads((HERE / "trace.json").read_text()),
        json.loads((HERE / "frontier.json").read_text()),
    )


def test_render_writes_both_charts(tmp_path):
    trace, frontier = _artifacts()
    written = chart.render(trace, frontier, tmp_path)
    assert [p.name for p in written] == [
        "burst-timeline.svg",
        "cost-latency-frontier.svg",
    ]
    for path in written:
        assert path.read_text().startswith("<svg")


def test_render_is_deterministic(tmp_path):
    trace, frontier = _artifacts()
    first = {
        p.name: p.read_bytes() for p in chart.render(trace, frontier, tmp_path / "a")
    }
    second = {
        p.name: p.read_bytes() for p in chart.render(trace, frontier, tmp_path / "b")
    }
    assert first == second, "rendering twice must produce identical bytes"


def test_committed_charts_are_current(tmp_path):
    """The free rot check: a stale committed chart fails CI for zero dollars."""
    trace, frontier = _artifacts()
    for path in chart.render(trace, frontier, tmp_path):
        committed = HERE / "charts" / path.name
        assert committed.exists(), f"{path.name} is not committed"
        assert path.read_bytes() == committed.read_bytes(), (
            f"{path.name} is stale - regenerate with:\n"
            f"  python chart.py --trace trace.json --frontier frontier.json --out charts/"  # noqa: E501  # copy-paste command
        )


def test_no_foreignobject_anywhere():
    # Mintlify's image CDN strips foreignObject, silently truncating its text.
    for path in sorted((HERE / "charts").glob("*.svg")):
        assert "foreignObject" not in path.read_text(), path.name


def test_render_refuses_an_empty_series(tmp_path):
    trace, frontier = _artifacts()
    empty = dict(frontier, points=[], pareto_front=[])
    with pytest.raises(ValueError, match="no points"):
        chart.render(trace, empty, tmp_path)
