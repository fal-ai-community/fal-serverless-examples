#!/usr/bin/env python3
"""Render the committed SVG charts for slot 02.2.

Stdlib only - requirements.txt mirrors App.requirements exactly, so a
plotting library here would ship to the GPU. Hand-written SVG also means
byte-for-byte determinism, which is what lets CI regenerate these files and
fail on a mismatch without spending a cent.

Two constraints that shape every line below:
  * GitHub's markdown sanitizer strips inline <svg>, so these are committed
    FILES referenced as images, never markup pasted into the README.
  * Docs image CDNs commonly strip <foreignObject>, so every label is a real
    <text> element.

The visual language here is deliberately minimal: two flat-colour series,
small labels, no gridlines or gradients. Keep render()'s signature stable
when restyling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402

WIDTH, HEIGHT = 960, 420
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 64, 24, 40, 48

MEASURED = "#8553ff"  # fal purple: measured series
MODELLED = "#6366f1"  # indigo: model output
QUEUE = "#94a3b8"
AXIS = "#64748b"
INK = "#0f172a"


def _n(value: float) -> str:
    """Fixed-precision numbers so output is byte-stable across platforms."""
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def _text(
    x: float,
    y: float,
    content: str,
    *,
    anchor: str = "start",
    size: int = 12,
    fill: str = INK,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" text-anchor="{anchor}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{content}</text>'
    )


def _polyline(
    points: list[tuple[float, float]],
    stroke: str,
    *,
    width: float = 2,
    dash: str | None = None,
) -> str:
    coords = " ".join(f"{_n(x)},{_n(y)}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round"{dash_attr}/>'
    )


def _frame(title: str, subtitle: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'width="{WIDTH}" height="{HEIGHT}" role="img">',
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="none"/>',
            _text(PAD_LEFT, 22, title, size=15, weight="600"),
            _text(PAD_LEFT, 36, subtitle, size=11, fill=AXIS),
            *body,
            "</svg>",
            "",
        ]
    )


def _axes(x_label: str, y_label: str, x_max: float, y_max: float) -> list[str]:
    x0, x1 = PAD_LEFT, WIDTH - PAD_RIGHT
    y0, y1 = HEIGHT - PAD_BOTTOM, PAD_TOP
    return [
        _polyline([(x0, y1), (x0, y0), (x1, y0)], AXIS, width=1),
        _text((x0 + x1) / 2, HEIGHT - 14, x_label, anchor="middle", size=11, fill=AXIS),
        _text(14, (y0 + y1) / 2, y_label, anchor="middle", size=11, fill=AXIS),
        _text(x1, y0 + 16, _n(x_max), anchor="end", size=10, fill=AXIS),
        _text(x0 - 8, y1 + 4, _n(y_max), anchor="end", size=10, fill=AXIS),
    ]


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def burst_timeline(trace: dict) -> str:
    """Measured runner count and queue depth over the recorded burst."""
    timeline = model.measured_timeline(trace, tick_s=1.0)
    if not timeline:
        raise ValueError("trace produced no timeline rows")
    # The axis origin is the burst's actual start (t=0), not the first row
    # measured_timeline() happens to emit. That first row starts at the
    # earliest runner's setup_start_s, which is well after t=0 whenever
    # capacity acquisition (PENDING, unbilled) takes a while - anchoring the
    # axis there instead of at 0 would silently hide that free gap.
    t_lo = 0.0
    t_hi = max(row["t"] for row in timeline)
    y_max = max(
        max(model.live_runners(r) for r in timeline),
        max(r["queue_depth"] for r in timeline),
        1,
    )
    x0, x1 = PAD_LEFT, WIDTH - PAD_RIGHT
    y0, y1 = HEIGHT - PAD_BOTTOM, PAD_TOP

    def to_xy(row, key):
        value = model.live_runners(row) if key == "runners" else row[key]
        return (
            _scale(row["t"], t_lo, t_hi, x0, x1),
            _scale(value, 0, y_max, y0, y1),
        )

    calibration = model.calibrate(trace)
    body = [
        *_axes("seconds from burst start", "count", t_hi - t_lo, y_max),
        _polyline([to_xy(r, "runners") for r in timeline], MEASURED, width=2),
        _polyline(
            [to_xy(r, "queue_depth") for r in timeline], QUEUE, width=2, dash="4 3"
        ),
        _text(x1, y1 + 14, "runners (measured)", anchor="end", size=11, fill=MEASURED),
        _text(x1, y1 + 30, "queue depth (measured)", anchor="end", size=11, fill=QUEUE),
    ]
    subtitle = (
        f"{trace['burst']['requests']} requests, {trace['burst']['pattern']} - "
        f"{calibration.observed_runners} runners, cold start "
        f"{calibration.cold_start_s:.0f}s - measured {trace['recorded_at']} "
        f"on {trace['machine_type']}"
    )
    return _frame("Runners and queue depth through one real burst", subtitle, body)


def cost_latency_frontier(frontier: dict) -> str:
    """Modelled cost against modelled p95 queue wait, one point per config."""
    points = frontier.get("points") or []
    if not points:
        raise ValueError("frontier has no points to plot")
    cost_hi = max(p["cost_usd"] for p in points)
    wait_hi = max(p["p95_wait_s"] for p in points)
    x0, x1 = PAD_LEFT, WIDTH - PAD_RIGHT
    y0, y1 = HEIGHT - PAD_BOTTOM, PAD_TOP

    def to_xy(point):
        return (
            _scale(point["cost_usd"], 0, cost_hi, x0, x1),
            _scale(point["p95_wait_s"], 0, wait_hi, y0, y1),
        )

    marks = []
    for point in points:
        x, y = to_xy(point)
        if point["measured"]:
            marks.append(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="6" fill="{MEASURED}"/>')
            marks.append(_text(x + 10, y + 4, "measured", size=11, fill=MEASURED))
        elif point["validated"]:
            marks.append(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="4" fill="{MODELLED}"/>')
        else:
            # Outside the measured envelope: hollow, so it never reads as an
            # equal to a validated point.
            marks.append(
                f'<circle cx="{_n(x)}" cy="{_n(y)}" r="4" fill="none" '
                f'stroke="{MODELLED}" stroke-width="1.5" stroke-dasharray="2 2"/>'
            )

    front = [to_xy(points[i]) for i in frontier.get("pareto_front", [])]
    body = [
        *_axes(
            "cost over the 1-hour horizon (USD)", "p95 queue wait (s)", cost_hi, wait_hi
        ),
        *([_polyline(front, MODELLED, width=1, dash="5 4")] if len(front) > 1 else []),
        *marks,
        _text(
            x1,
            y1 + 14,
            "modelled - dashed line = Pareto front",
            anchor="end",
            size=11,
            fill=MODELLED,
        ),
    ]
    subtitle = (
        f"{len(points)} configurations simulated on the measured traffic shape - "
        f"${frontier['rate_usd_per_second'] * 3600:.2f}/h, "
        f"granularity {frontier['billing_granularity_s']:.0f}s"
    )
    return _frame("What each configuration costs, and what it waits", subtitle, body)


def render(trace: dict, frontier: dict, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "burst-timeline.svg": burst_timeline(trace),
        "cost-latency-frontier.svg": cost_latency_frontier(frontier),
    }
    written = []
    for name in sorted(documents):
        path = out_dir / name
        path.write_text(documents[name])
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="trace.json")
    parser.add_argument("--frontier", default="frontier.json")
    parser.add_argument("--out", default="charts")
    args = parser.parse_args(argv)

    # Readers run this directly, so a missing or unparseable input should
    # print one clear line, not a raw traceback - same contract as model.py.
    try:
        path = args.trace
        with open(path) as handle:
            trace = json.load(handle)
        path = args.frontier
        with open(path) as handle:
            frontier = json.load(handle)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON ({exc})", file=sys.stderr)
        return 1

    for path in render(trace, frontier, args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
