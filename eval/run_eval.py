"""Citation-grounding evaluation: baseline vs grounded pipeline over the benchmark set.

    python -m eval.run_eval --limit 100 --concurrency 3
    python -m eval.run_eval --summary-only            # re-print the summary from saved results
    python -m eval.run_eval --export-sample 60        # CSV for human labelling -> eval/agreement.py

Controls for a fair comparison:
* Both modes share the same research plan and the same cached search results per query,
  so the only differences are the generation prompts, context pruning and citation validation.
* The judge model and prompt are identical for both modes (temperature 0).
* Results are appended per run to JSONL, so an interrupted eval resumes where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import random
import statistics
import time
from pathlib import Path

from research_pipeline.agents import PlannerAgent
from research_pipeline.checkpoint import Checkpoint
from research_pipeline.config import Settings
from research_pipeline.judge import GroundingJudge
from research_pipeline.llm import build_judge_llm, build_llm
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.search import CachedSearch, build_search

MODES = {
    # Tutorial-style: loose prompts, every search result kept (length-capped), no validation.
    "baseline": {"grounded": False, "prune_context": False},
    # This project: strict citation prompts, relevance-ranked token-budgeted context, validation.
    "grounded": {"grounded": True, "prune_context": True},
}


def load_queries(path: Path, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return rows[:limit] if limit else rows


def load_done(out: Path) -> dict[tuple[str, str], dict]:
    done = {}
    for f in (out / f"{mode}.jsonl" for mode in MODES):  # not search_cache.jsonl
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["query_id"], r["mode"])] = r
    return done


async def evaluate(args: argparse.Namespace) -> None:
    base = Settings.from_env()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(Path(args.queries), args.limit)
    done = load_done(out)

    llm = build_llm(base)
    search = CachedSearch(build_search(base), path=str(out / "search_cache.jsonl"))
    judge_llm = build_judge_llm(base)  # None -> judge shares the main client
    judge = GroundingJudge(judge_llm or llm, model=base.effective_judge_model)
    planner = PlannerAgent(llm, base)
    plan_cache_path = out / "plans.json"
    plan_cache = json.loads(plan_cache_path.read_text()) if plan_cache_path.exists() else {}
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def one(q: dict) -> None:
        async with sem:
            if q["id"] not in plan_cache:
                plan_cache[q["id"]] = [t.model_dump() for t in await planner.plan(q["topic"])]
                async with write_lock:
                    plan_cache_path.write_text(json.dumps(plan_cache, indent=1))
            for mode, overrides in MODES.items():
                if mode not in args.modes or (q["id"], mode) in done:
                    continue
                s = dataclasses.replace(base, **overrides)
                ckpt = Checkpoint(None)
                ckpt.state["plan"] = plan_cache[q["id"]]  # same plan for both modes
                t0 = time.perf_counter()
                try:
                    report = await ResearchPipeline(s, llm, search, checkpoint=ckpt).run(q["topic"])
                    score = await judge.score(report)
                except Exception as exc:  # keep going; record the failure
                    print(f"  ! {q['id']} {mode}: {exc!r}")
                    continue
                rec = {
                    "query_id": q["id"],
                    "category": q["category"],
                    "topic": q["topic"],
                    "mode": mode,
                    "seconds": round(time.perf_counter() - t0, 2),
                    "judge": score.summary(),
                    "citation_stats": report.citation_stats,
                    "verdicts": [dataclasses.asdict(v) for v in score.verdicts],
                    "sources": {s.id: s.content[:1500] for s in report.sources},
                    "report": report.markdown,
                }
                async with write_lock:
                    with (out / f"{mode}.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    f"  {q['id']} {mode:9s} claims={score.n_claims:3d} "
                    f"unsupported={score.unsupported_rate:.1%} ({rec['seconds']}s)"
                )

    print(
        f"Evaluating {len(queries)} queries x {len(args.modes)} modes "
        f"(llm={base.llm_provider}:{base.llm_model}, judge={base.effective_judge_model}, search={base.search_provider})"
    )
    await asyncio.gather(*(one(q) for q in queries))
    await llm.aclose()
    if judge_llm is not None:
        await judge_llm.aclose()
    await search.aclose()
    print(f"LLM usage: {llm.usage.as_dict()}")
    summarize(out)


def paired_bootstrap(
    base: list[float], new: list[float], iters: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95% CI for the relative reduction in mean unsupported rate, resampling queries."""
    rng = random.Random(seed)
    n = len(base)
    stats = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base[i] for i in idx) / n
        g = sum(new[i] for i in idx) / n
        if b > 0:
            stats.append((b - g) / b)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def summarize(out: Path) -> None:
    done = load_done(out)
    by_mode: dict[str, dict[str, dict]] = {}
    for (qid, mode), r in done.items():
        by_mode.setdefault(mode, {})[qid] = r
    if not by_mode:
        print("no results yet")
        return

    lines = ["| metric | " + " | ".join(by_mode) + " |", "|---|" + "---|" * len(by_mode)]
    table: dict[str, dict[str, float]] = {}
    for mode, rows in by_mode.items():
        js = [r["judge"] for r in rows.values()]
        claims = sum(j["n_claims"] for j in js)
        table[mode] = {
            "queries": len(rows),
            "claims judged": claims,
            "unsupported rate (micro)": sum(j["unsupported"] for j in js) / claims if claims else 0,
            "unsupported rate (macro)": statistics.mean(j["unsupported_rate"] for j in js),
            "partially supported rate": sum(j["partially_supported"] for j in js) / claims
            if claims
            else 0,
            "citation precision": statistics.mean(j["citation_precision"] for j in js),
            "uncited claims / report": statistics.mean(j["uncited_claims"] for j in js),
            "judge errors": sum(j.get("judge_error", 0) for j in js),
            "mean seconds / report": statistics.mean(r["seconds"] for r in rows.values()),
        }
    for metric in next(iter(table.values())):
        vals = []
        for mode in by_mode:
            v = table[mode][metric]
            vals.append(
                f"{v:.1%}"
                if "rate" in metric or "precision" in metric
                else f"{v:,.2f}".rstrip("0").rstrip(".")
            )
        lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    summary = {"table": table}
    if {"baseline", "grounded"} <= by_mode.keys():
        common = sorted(by_mode["baseline"].keys() & by_mode["grounded"].keys())
        b = [by_mode["baseline"][q]["judge"]["unsupported_rate"] for q in common]
        g = [by_mode["grounded"][q]["judge"]["unsupported_rate"] for q in common]
        if common and statistics.mean(b) > 0:
            rel = (statistics.mean(b) - statistics.mean(g)) / statistics.mean(b)
            lo, hi = paired_bootstrap(b, g)
            wins = sum(gi < bi for bi, gi in zip(b, g, strict=True))
            lines += [
                "",
                f"Relative reduction in unsupported-claim rate (paired, n={len(common)}): "
                f"**{rel:.1%}** (95% bootstrap CI {lo:.1%} – {hi:.1%}); grounded better on {wins}/{len(common)} queries.",
            ]
            summary["relative_reduction"] = {
                "value": rel,
                "ci95": [lo, hi],
                "n": len(common),
                "wins": wins,
            }
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + text)


def export_sample(out: Path, n: int, seed: int = 0) -> None:
    """Blind sample (mode hidden, shuffled) of judged claims for human labelling."""
    rng = random.Random(seed)
    pool = []
    for (_, mode), r in load_done(out).items():
        for v in r["verdicts"]:
            if v["label"] in ("not_a_claim", "judge_error"):
                continue
            excerpts = "\n---\n".join(
                f"[{i}] {r['sources'].get(str(i), '(missing)')}" for i in v["citations"]
            )
            pool.append(
                {
                    "claim": v["text"],
                    "cited_excerpts": excerpts or "(no citation)",
                    "judge_label": v["label"],
                    "query_id": r["query_id"],
                    "mode": mode,
                }
            )
    rng.shuffle(pool)
    path = out / "human_label_sample.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "row",
                "claim",
                "cited_excerpts",
                "human_label",
                "judge_label",
                "query_id",
                "mode",
            ],
        )
        w.writeheader()
        for i, row in enumerate(pool[:n], start=1):
            w.writerow({"row": i, **row, "human_label": ""})
    print(
        f"wrote {min(n, len(pool))} rows to {path}. Fill 'human_label' (hide the judge_label column while labelling), "
        f"then run: python -m eval.agreement {path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--concurrency", type=int, default=3, help="queries evaluated in parallel")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--export-sample", type=int, default=0, metavar="N")
    args = ap.parse_args()
    if args.export_sample:
        export_sample(Path(args.out), args.export_sample)
    elif args.summary_only:
        summarize(Path(args.out))
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
