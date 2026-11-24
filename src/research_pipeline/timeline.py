"""Render an execution timeline (a Gantt chart of DAG nodes) as text or SVG.

A timeline makes the scheduler's behaviour visible:

* Bars that overlap ran concurrently; with ``max_concurrency=1`` they form a staircase.
* The pale segment before a bar is *queue time*: the node was ready (its dependencies
  had finished) but waited for a concurrency slot. Long pale segments mean the
  concurrency cap, not the API, is the bottleneck.
* The ``write`` bar starts only after the last summary ends. It can't overlap anything,
  so its length bounds the possible speed-up (Amdahl's law).

Spans come from ``ReportResult.timeline``: ``{node, stage, queued, start, end, status}``,
seconds from the start of the run.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

Span = dict[str, Any]

STAGE_COLORS = {
    "plan": "#7c8591",
    "search": "#3b7dd8",
    "summarize": "#2e9e6a",
    "write": "#e07b28",
}
STAGE_LABELS = {
    "plan": "plan",
    "search": "search",
    "summarize": "summarize",
    "write": "write report",
}


def _order(span: Span) -> tuple[int, int, int]:
    stage, _, task = span["node"].partition(":")
    if stage == "plan":
        return (0, 0, 0)
    if stage == "write":
        return (2, 0, 0)
    num = int(task[1:]) if task[1:].isdigit() else 999
    return (1, num, 0 if stage == "search" else 1)


def sort_spans(spans: list[Span]) -> list[Span]:
    """plan, then search/summarize per sub-task (t1, t2, ...), then write."""
    return sorted((s for s in spans if s.get("end") is not None), key=_order)


def total_seconds(spans: list[Span]) -> float:
    return max((s["end"] for s in spans if s.get("end") is not None), default=0.0)


def render_ascii(spans: list[Span], width: int = 48) -> str:
    """A text Gantt chart: '█' = running, '░' = ready but waiting for a slot."""
    rows = sort_spans(spans)
    if not rows:
        return "(no timeline)"
    total = total_seconds(rows) or 1.0
    scale = width / total
    lines = [f"Execution timeline ({total:.1f}s; █ running, ░ waiting for a slot)"]
    for s in rows:
        q, a, b = (round(s[k] * scale) for k in ("queued", "start", "end"))
        b = max(b, a + 1)  # every node gets at least one cell
        bar = " " * q + "░" * (a - q) + "█" * (b - a)
        lines.append(f"  {s['node']:<14} {bar:<{width + 1}} {s['end'] - s['start']:5.1f}s")
    return "\n".join(lines)


def _nice_step(total: float, max_ticks: int = 7) -> float:
    for step in (0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300):
        if total / step <= max_ticks:
            return step
    return math.ceil(total / max_ticks)


def render_svg(panels: list[tuple[str, list[Span]]], title: str = "", note: str = "") -> str:
    """One or more timelines stacked on a shared time axis, as a standalone SVG.

    ``panels`` = [(heading, spans), ...]. A shared axis is the point: when sequential
    and parallel runs of the same topic are stacked, the difference in length *is* the
    speed-up.
    """
    width, left, right = 880, 132, 64
    plot = width - left - right
    pitch, bar_h = 22, 14
    panels = [(h, sort_spans(sp)) for h, sp in panels]
    total = max((total_seconds(sp) for _, sp in panels), default=1.0) or 1.0
    step = _nice_step(total)
    axis_max = math.ceil(total / step) * step

    def x(t: float) -> float:
        return left + t / axis_max * plot

    font = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
    out: list[str] = []
    y = 20
    if title:
        out.append(
            f"<text x='24' y='{y + 8}' font-size='17' font-weight='600' fill='#1f2328'>"
            f"{escape(title)}</text>"
        )
        y += 30
    if note:
        out.append(f"<text x='24' y='{y}' font-size='12.5' fill='#59636e'>{escape(note)}</text>")
        y += 18

    for heading, spans in panels:
        y += 18
        out.append(
            f"<text x='24' y='{y}' font-size='14' font-weight='600' fill='#1f2328'>"
            f"{escape(heading)}</text>"
        )
        y += 12
        top = y
        bottom = top + max(len(spans), 1) * pitch
        tick = 0.0
        while tick <= axis_max + 1e-9:
            tx = x(tick)
            out.append(
                f"<line x1='{tx:.1f}' y1='{top}' x2='{tx:.1f}' y2='{bottom}' stroke='#d8dee4'/>"
            )
            out.append(
                f"<text x='{tx:.1f}' y='{bottom + 14}' font-size='11' fill='#59636e' "
                f"text-anchor='middle'>{tick:g}s</text>"
            )
            tick += step
        for i, s in enumerate(spans):
            ry = top + i * pitch + (pitch - bar_h) / 2
            color = STAGE_COLORS.get(s["stage"], "#888888")
            out.append(
                f"<text x='{left - 10}' y='{ry + bar_h - 3:.1f}' font-size='12' "
                f"fill='#1f2328' text-anchor='end'>{escape(s['node'])}</text>"
            )
            if s["start"] - s["queued"] > 0.02:
                out.append(
                    f"<rect x='{x(s['queued']):.1f}' y='{ry:.1f}' "
                    f"width='{x(s['start']) - x(s['queued']):.1f}' height='{bar_h}' "
                    f"rx='3' fill='{color}' fill-opacity='0.18'/>"
                )
            w = max(x(s["end"]) - x(s["start"]), 1.5)
            fill_opacity = "1" if s.get("status", "done") == "done" else "0.45"
            out.append(
                f"<rect x='{x(s['start']):.1f}' y='{ry:.1f}' width='{w:.1f}' "
                f"height='{bar_h}' rx='3' fill='{color}' fill-opacity='{fill_opacity}'/>"
            )
            out.append(
                f"<text x='{x(s['start']) + w + 5:.1f}' y='{ry + bar_h - 3:.1f}' "
                f"font-size='11' fill='#59636e'>{s['end'] - s['start']:.1f}s</text>"
            )
        y = bottom + 30

    y += 8  # clear the last axis' tick labels
    lx = 24
    for stage, color in STAGE_COLORS.items():
        out.append(f"<rect x='{lx}' y='{y - 10}' width='12' height='12' rx='2' fill='{color}'/>")
        out.append(
            f"<text x='{lx + 18}' y='{y}' font-size='12' fill='#1f2328'>"
            f"{STAGE_LABELS[stage]}</text>"
        )
        lx += 30 + 7 * len(STAGE_LABELS[stage])
    out.append(
        f"<rect x='{lx}' y='{y - 10}' width='12' height='12' rx='2' "
        f"fill='#3b7dd8' fill-opacity='0.18'/>"
    )
    out.append(
        f"<text x='{lx + 18}' y='{y}' font-size='12' fill='#1f2328'>"
        f"ready, waiting for a concurrency slot</text>"
    )
    height = y + 18
    body = "\n  ".join(out)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' {font}>\n"
        f"  <rect width='100%' height='100%' fill='#ffffff'/>\n  {body}\n</svg>\n"
    )
