"""Command-line entry points (installed as ``research-run``).

    research-run --check                       verify .env: LLM key/model and web search
    research-run "How do heat pumps work in cold climates?"
    research-run "..." --baseline --out baseline.md --concurrency 1

Reads settings from ``.env`` in the current folder (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .dag import NodeExecutionError
from .llm import build_llm
from .pipeline import ResearchPipeline
from .retry import PermanentError, RetryableError
from .search import build_search

HINTS = {
    "http 401": "the API key was rejected: check LLM_API_KEY (no quotes or spaces around it).",
    "http 402": "the account has no credit: add billing/credit at your provider.",
    "http 403": "the key is not allowed to use this model or endpoint.",
    "http 404": "wrong endpoint or model: check LLM_BASE_URL (usually ends in /v1) and LLM_MODEL.",
    "model_not_found": "LLM_MODEL does not exist at this endpoint: check the provider's model list.",
    "insufficient": "the account has no credit or quota left.",
    "finish_reason=length": "use a non-reasoning chat model for LLM_MODEL.",
}


def _hint(error: str) -> str:
    low = error.lower()
    return next((f"\n       hint: {h}" for k, h in HINTS.items() if k in low), "")


def _mask(key: str) -> str:
    if not key:
        return "(missing)"
    return f"{key[:5]}…{key[-4:]}" if len(key) > 12 else "(set)"


async def check(s: Settings) -> bool:
    print("Settings from .env / environment:")
    print(f"  LLM_PROVIDER    {s.llm_provider}")
    if s.llm_provider != "fake":
        print(f"  LLM_BASE_URL    {s.llm_base_url}")
        print(f"  LLM_MODEL       {s.llm_model}")
        print(f"  LLM_API_KEY     {_mask(s.llm_api_key)}")
        print(f"  JUDGE_MODEL     {s.effective_judge_model}")
    print(f"  SEARCH_PROVIDER {s.search_provider}\n")
    ok = True

    if s.llm_provider == "fake":
        print(
            "[warn] LLM is the offline fake. For real calls set LLM_PROVIDER=openai and LLM_API_KEY."
        )
    else:
        try:
            llm = build_llm(s)
        except ValueError as exc:
            print(f"[FAIL] LLM: {exc}")
            ok = False
        else:
            t0 = time.perf_counter()
            try:
                reply = await llm.complete(
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    purpose="check",
                    temperature=0,
                    max_tokens=20,
                )
                print(
                    f"[ok]   LLM {s.llm_model} replied {reply.strip()[:40]!r} "
                    f"in {time.perf_counter() - t0:.1f}s"
                )
            except (PermanentError, RetryableError) as exc:
                print(f"[FAIL] LLM: {exc}{_hint(str(exc))}")
                ok = False
            finally:
                await llm.aclose()

    if s.search_provider == "fake":
        print("[warn] Search is the offline fake. For real search set SEARCH_PROVIDER=duckduckgo.")
    else:
        search = build_search(s)
        t0 = time.perf_counter()
        try:
            results = await search.search("solid-state battery electrolyte safety", 3)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] search ({s.search_provider}): {exc}")
            ok = False
        else:
            print(
                f"[ok]   search ({s.search_provider}): {len(results)} results "
                f"in {time.perf_counter() - t0:.1f}s"
            )
            for r in results:
                print(f"         {len(r.content):6,} chars  {r.title[:60]}")
            if results and all(len(r.content) < 400 for r in results):
                print(
                    "[warn] only short snippets came back: page fetching may be blocked "
                    "(firewall/VPN?). Reports will still work, with thinner evidence."
                )
        finally:
            await search.aclose()
    print("\nAll good." if ok else "\nFix the [FAIL] lines above, then run --check again.")
    return ok


class _Printer:
    """Turns pipeline events into a readable terminal log; streams the report text."""

    def __init__(self) -> None:
        self.writing = False

    async def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "plan":
            print(f"Plan ({len(data['tasks'])} sub-questions):")
            for t in data["tasks"]:
                print(f"  {t['id']}. {t['title']}   [search: {t['query']}]")
            print()
        elif kind == "task_status" and data["status"] in ("searched", "done", "failed"):
            extra = f" ({data['seconds']}s)" if "seconds" in data else ""
            err = f" - {data['error']}" if data.get("error") else ""
            print(f"  {data['task_id']}: {data['status']}{extra}{err}")
        elif kind == "pages_fetched":
            print(
                f"      fetched {data['fetched']}/{data['total']} pages for: {data['query'][:50]}"
            )
        elif kind == "retry":
            print(
                f"      retry {data['where']} #{data['attempt']} in {data['delay']}s ({data['error'][:80]})"
            )
        elif kind == "stage" and data.get("stage") == "writing" and not self.writing:
            self.writing = True
            print(
                "\nWriting report (draft streams below; the saved file is the validated version)\n"
            )
        elif kind == "report_delta":
            sys.stdout.write(data["text"])
            sys.stdout.flush()


async def run(topic: str, s: Settings, out: Path) -> int:
    llm, search = build_llm(s), build_search(s)
    t0 = time.perf_counter()
    try:
        report = await ResearchPipeline(s, llm, search, sink=_Printer()).run(topic)
    except (PermanentError, RetryableError, NodeExecutionError) as exc:
        cause = exc.cause if isinstance(exc, NodeExecutionError) else exc
        print(f"\n[FAIL] {cause}{_hint(str(cause))}")
        return 1
    finally:
        await llm.aclose()
        await search.aclose()
    await asyncio.to_thread(out.write_text, report.markdown, encoding="utf-8")
    c = report.citation_stats
    print(
        f"\n\nSaved {out}  ·  {time.perf_counter() - t0:.1f}s  ·  {len(report.sources)} sources cited"
        f"  ·  {c.get('cited_claims', 0)} cited / {c.get('uncited_claims', 0)} uncited sentences"
        f"  ·  LLM usage {llm.usage.as_dict()}"
    )
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="research-run", description="Run one research report.")
    ap.add_argument("topic", nargs="?", help="research question")
    ap.add_argument("--check", action="store_true", help="verify LLM key/model and search")
    ap.add_argument("--out", default="report.md", help="where to save the report")
    ap.add_argument("--baseline", action="store_true", help="loose prompts, no pruning")
    ap.add_argument("--concurrency", type=int, help="DAG nodes in flight (1 = sequential)")
    args = ap.parse_args()

    s = Settings.from_env()
    if args.check:
        sys.exit(0 if asyncio.run(check(s)) else 1)
    if not args.topic:
        ap.error('give a topic, e.g. research-run "How do heat pumps work in cold climates?"')
    overrides: dict[str, Any] = {}
    if args.baseline:
        overrides |= {"grounded": False, "prune_context": False}
    if args.concurrency:
        overrides["max_concurrency"] = args.concurrency
    s = dataclasses.replace(s, **overrides)
    try:
        sys.exit(asyncio.run(run(args.topic, s, Path(args.out))))
    except ValueError as exc:  # e.g. missing API key
        print(f"[FAIL] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
