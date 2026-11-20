"""End-to-end latency benchmark with an ablation.

    python -m eval.benchmark --limit 10 --concurrency 4

Variants (same queries, run order shuffled per query to spread provider-side drift):
  sequential       max_concurrency=1, no pruning     -> one-task-at-a-time baseline
  dag              max_concurrency=K, no pruning     -> effect of parallel DAG execution alone
  dag+pruning      max_concurrency=K, token budgets  -> + smaller prompts (faster prefill, shorter writer input)

Reported: mean / p50 / p95 wall-clock seconds per report and the reduction vs sequential.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import statistics
import time
from pathlib import Path

from research_pipeline.config import Settings
from research_pipeline.llm import build_llm
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.search import build_search

from .run_eval import load_queries


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run(args: argparse.Namespace) -> None:
    base = Settings.from_env()
    variants = {
        "sequential": {"max_concurrency": 1, "prune_context": False},
        "dag": {"max_concurrency": args.concurrency, "prune_context": False},
        "dag+pruning": {"max_concurrency": args.concurrency, "prune_context": True},
    }
    llm, search = build_llm(base), build_search(base)
    queries = load_queries(Path(args.queries), args.limit)
    rng = random.Random(0)
    times: dict[str, list[float]] = {v: [] for v in variants}
    print(
        f"Benchmarking {len(queries)} queries (llm={base.llm_provider}, search={base.search_provider})"
    )
    for q in queries:
        order = list(variants)
        rng.shuffle(order)
        for name in order:
            s = dataclasses.replace(base, **variants[name])
            t0 = time.perf_counter()
            try:
                await ResearchPipeline(s, llm, search).run(q["topic"])
            except Exception as exc:
                print(f"  ! {q['id']} {name}: {exc!r}")
                continue
            times[name].append(time.perf_counter() - t0)
            print(f"  {q['id']} {name:12s} {times[name][-1]:6.2f}s")
    await llm.aclose()
    await search.aclose()

    ref = statistics.mean(times["sequential"])
    lines = [
        "| variant | n | mean s | p50 s | p95 s | mean reduction vs sequential |",
        "|---|---|---|---|---|---|",
    ]
    result = {}
    for name, ts in times.items():
        if not ts:
            continue
        m = statistics.mean(ts)
        result[name] = {
            "n": len(ts),
            "mean": m,
            "p50": pct(ts, 0.5),
            "p95": pct(ts, 0.95),
            "reduction": 1 - m / ref,
        }
        lines.append(
            f"| {name} | {len(ts)} | {m:.2f} | {pct(ts, 0.5):.2f} | {pct(ts, 0.95):.2f} | {1 - m / ref:.1%} |"
        )
    text = "\n".join(lines)
    print("\n" + text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.md").write_text(text + "\n", encoding="utf-8")
    (out / "latency.json").write_text(
        json.dumps(
            {
                "settings": {
                    "llm": base.llm_model,
                    "search": base.search_provider,
                    "concurrency": args.concurrency,
                },
                "results": result,
                "raw": times,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="eval/results")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
