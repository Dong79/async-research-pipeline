"""Draw the README timeline: the same topic run sequentially and as a parallel DAG.

    python -m eval.timeline "How well do heat pumps work in cold climates?"
    python -m eval.timeline "..." --concurrency 4 --out docs/timeline.svg

Costs two report runs of API tokens. Both runs plan the topic themselves and use live
search (no cache), so neither gets a head start. The SVG stacks both runs on one time
axis: the difference in length is the speed-up, and the write bar shows why it is
bounded (it can only start after the last summary).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from pathlib import Path

from research_pipeline.config import Settings
from research_pipeline.llm import build_llm
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.search import build_search
from research_pipeline.timeline import render_ascii, render_svg


async def run(args: argparse.Namespace) -> None:
    base = Settings.from_env()
    llm, search = build_llm(base), build_search(base)
    results = {}
    try:
        for label, conc in (("Sequential", 1), ("Parallel DAG", args.concurrency)):
            s = dataclasses.replace(base, max_concurrency=conc)
            print(f"Running {label.lower()} (max_concurrency={conc})...")
            report = await ResearchPipeline(s, llm, search).run(args.topic)
            results[label] = report
            print(render_ascii(report.timeline) + "\n")
    finally:
        await llm.aclose()
        await search.aclose()

    seq = results["Sequential"].metrics["elapsed_seconds"]
    par = results["Parallel DAG"].metrics["elapsed_seconds"]
    simulated = base.llm_provider == "fake"
    note = f"Topic: {args.topic[:90]}" + (
        " | offline simulation, not a measurement" if simulated else f" | {base.llm_model}"
    )
    svg = render_svg(
        [
            (f"Sequential (max_concurrency=1): {seq:.1f}s", results["Sequential"].timeline),
            (
                f"Parallel DAG (max_concurrency={args.concurrency}): {par:.1f}s "
                f"({1 - par / seq:.0%} faster)",
                results["Parallel DAG"].timeline,
            ),
        ],
        title="Same topic, sequential vs parallel DAG execution",
        note=note,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"sequential {seq:.1f}s -> parallel {par:.1f}s ({1 - par / seq:.0%} faster). Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("topic")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="docs/timeline.svg")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
