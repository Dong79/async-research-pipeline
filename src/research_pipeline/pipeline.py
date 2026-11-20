"""The research pipeline: plan -> DAG(search_i -> summarize_i)* -> write.

        ┌─ search:t1 ─ summarize:t1 ─┐
 plan ──┼─ search:t2 ─ summarize:t2 ─┼── write ── report
        └─ search:t3 ─ summarize:t3 ─┘

Planning runs first because the shape of the graph (how many branches) depends on
its output; the rest is a static DAG executed by :class:`DAGExecutor`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .agents import PlannerAgent, ResearcherAgent, WriterAgent
from .budget import count_tokens, fit_sections, normalize_url, prune_sources, truncate_to_tokens
from .checkpoint import Checkpoint
from .citations import (
    extract_citations,
    remap_citations,
    strip_references_section,
    validate_citations,
)
from .config import Settings
from .dag import DAG, DAGExecutor, Node, NodeFailure
from .events import EventSink, emit, reset_sink, set_sink
from .llm import LLMClient
from .models import ReportResult, Source, SubTask, TaskResult
from .retry import PermanentError
from .search import SearchClient


class ResearchPipeline:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        search: SearchClient,
        *,
        sink: EventSink | None = None,
        checkpoint: Checkpoint | None = None,
    ):
        self.s = settings
        self.search = search
        self.sink = sink
        self.ckpt = checkpoint or Checkpoint(None)
        self.planner = PlannerAgent(llm, settings)
        self.researcher = ResearcherAgent(llm, settings)
        self.writer = WriterAgent(llm, settings)
        self.timings: dict[str, Any] = {}

    async def run(self, topic: str) -> ReportResult:
        token = set_sink(self.sink)
        try:
            return await self._run(topic)
        finally:
            reset_sink(token)

    async def _run(self, topic: str) -> ReportResult:
        t0 = time.perf_counter()
        await emit(
            "run_started",
            {
                "topic": topic,
                "grounded": self.s.grounded,
                "max_concurrency": self.s.max_concurrency,
            },
        )
        tasks = self.ckpt.plan
        if tasks is None:
            await emit("stage", {"stage": "planning"})
            tasks = await self.planner.plan(topic)
            await self.ckpt.save_plan(tasks)
        await emit("plan", {"tasks": [t.model_dump() for t in tasks]})

        executor = DAGExecutor(self.s.max_concurrency, on_event=self._on_node_event)
        results = await executor.run(
            self.build_dag(topic, tasks), precomputed=self.ckpt.precomputed()
        )
        self.timings = executor.timings

        report: ReportResult = results["write"]
        report.metrics["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        await emit(
            "report",
            {
                "markdown": report.markdown,
                "sources": [s.model_dump(exclude={"content"}) for s in report.sources],
                "citation_stats": report.citation_stats,
                "metrics": report.metrics,
            },
        )
        return report

    # ------------------------------------------------------------------------- graph

    def build_dag(self, topic: str, tasks: list[SubTask]) -> DAG:
        dag = DAG()
        for t in tasks:
            dag.add(Node(f"search:{t.id}", self._make_search(t), optional=True))
            dag.add(
                Node(
                    f"summarize:{t.id}",
                    self._make_summarize(t),
                    deps=(f"search:{t.id}",),
                    optional=True,
                )
            )
        dag.add(
            Node(
                "write",
                self._make_write(topic, tasks),
                deps=tuple(f"summarize:{t.id}" for t in tasks),
            )
        )
        return dag

    def _make_search(self, task: SubTask):
        async def run(_: Mapping[str, Any]) -> list[Source]:
            return await self.search.search(task.query, self.s.results_per_query)

        return run

    def _make_summarize(self, task: SubTask):
        async def run(inputs: Mapping[str, Any]) -> TaskResult:
            raw = inputs[f"search:{task.id}"]
            if isinstance(raw, NodeFailure):
                return TaskResult(task=task, degraded=True, error=raw.error)
            if self.s.prune_context:
                sources, stats = prune_sources(
                    raw, task.query, self.s.summarizer_token_budget, self.s.per_source_token_cap
                )
                await emit("context_pruned", {"task_id": task.id, **stats.as_dict()})
            else:  # naive: keep every result, only cap length
                sources = [
                    s.model_copy(
                        update={
                            "id": i,
                            "content": truncate_to_tokens(s.content, self.s.per_source_token_cap),
                        }
                    )
                    for i, s in enumerate(raw, start=1)
                ]
            summary = await self.researcher.summarize(task, sources)
            return TaskResult(task=task, sources=sources, summary=summary)

        return run

    def _make_write(self, topic: str, tasks: list[SubTask]):
        async def run(inputs: Mapping[str, Any]) -> ReportResult:
            results: list[TaskResult] = []
            for t in tasks:
                r = inputs[f"summarize:{t.id}"]
                results.append(
                    TaskResult(task=t, degraded=True, error=r.error)
                    if isinstance(r, NodeFailure)
                    else r
                )
            usable = [r for r in results if not r.degraded and r.summary.strip()]
            if not usable:
                raise PermanentError("every sub-task failed; nothing to write")
            await emit(
                "stage",
                {
                    "stage": "writing",
                    "usable_tasks": len(usable),
                    "degraded_tasks": len(results) - len(usable),
                },
            )
            return await self._write(topic, results, usable)

        return run

    async def _write(
        self, topic: str, all_results: list[TaskResult], usable: list[TaskResult]
    ) -> ReportResult:
        # 1. Global source numbering: dedupe across sub-tasks, remap each summary's local [n].
        global_by_url: dict[str, Source] = {}
        sections: list[tuple[str, str]] = []
        for r in usable:
            mapping: dict[int, int] = {}
            for s in r.sources:
                key = normalize_url(s.url)
                if key not in global_by_url:
                    global_by_url[key] = s.model_copy(update={"id": len(global_by_url) + 1})
                mapping[s.id] = global_by_url[key].id
            sections.append((r.task.title, remap_citations(r.summary, mapping)))
        all_sources = list(global_by_url.values())
        valid_ids = {s.id for s in all_sources}

        # 2. Token budget for the writer, shared fairly across sections.
        before = sum(count_tokens(s) for _, s in sections)
        fitted = fit_sections([s for _, s in sections], self.s.writer_token_budget)
        sections = [(t, f) for (t, _), f in zip(sections, fitted, strict=True)]
        await emit(
            "writer_context",
            {"tokens_before": before, "tokens_after": sum(count_tokens(s) for _, s in sections)},
        )

        # 3. Stream the report, forwarding deltas in small batches.
        chunks: list[str] = []
        buf = ""
        async for delta in self.writer.stream(topic, sections, all_sources):
            chunks.append(delta)
            buf += delta
            if len(buf) >= 160:
                await emit("report_delta", {"text": buf})
                buf = ""
        if buf:
            await emit("report_delta", {"text": buf})
        draft = strip_references_section("".join(chunks))

        # 4. Post-hoc citation validation (grounded mode) — deterministic, no extra LLM call.
        if self.s.grounded:
            body, vstats = validate_citations(draft, valid_ids, drop_uncited=self.s.drop_uncited)
        else:
            body = draft
            _, vstats = validate_citations(draft, valid_ids)  # measure only

        # 5. Renumber cited sources densely by first appearance; append canonical references.
        order = [i for i in extract_citations(body) if i in valid_ids]
        renumber = {old: new for new, old in enumerate(order, start=1)}
        body = remap_citations(body, renumber) if self.s.grounded else body
        by_id = {s.id: s for s in all_sources}
        if self.s.grounded:
            cited = [by_id[old].model_copy(update={"id": new}) for old, new in renumber.items()]
        else:
            cited = all_sources  # baseline keeps original global numbering
        refs = "\n".join(f"[{s.id}] [{s.title}]({s.url})" for s in cited)
        markdown = f"{body.rstrip()}\n\n## References\n\n{refs}\n"

        return ReportResult(
            topic=topic,
            markdown=markdown,
            sources=cited,
            tasks=all_results,
            citation_stats=vstats.as_dict(),
            metrics={"degraded_tasks": float(sum(r.degraded for r in all_results))},
        )

    # ------------------------------------------------------------------------- events

    async def _on_node_event(self, kind: str, data: dict[str, Any]) -> None:
        node: str = data["node"]
        stage, _, task_id = node.partition(":")
        if kind == "node_completed":
            await self.ckpt.save_node(node, data["result"])
        if node == "write":
            if kind == "node_started":
                await emit("stage", {"stage": "writing"})
            return
        status = {
            ("search", "node_started"): "searching",
            ("search", "node_completed"): "searched",
            ("summarize", "node_started"): "summarizing",
            ("summarize", "node_completed"): "done",
            ("search", "node_skipped"): "searched",
            ("summarize", "node_skipped"): "done",
        }.get((stage, kind), "failed" if kind == "node_failed" else kind)
        payload: dict[str, Any] = {"task_id": task_id, "status": status}
        if "seconds" in data:
            payload["seconds"] = data["seconds"]
        if "error" in data:
            payload["error"] = data["error"]
        await emit("task_status", payload)
        if stage == "summarize" and kind in ("node_completed", "node_skipped"):
            result: TaskResult = data.get("result") or self.ckpt.precomputed().get(node)
            if result is not None:
                await emit(
                    "task_summary",
                    {
                        "task_id": task_id,
                        "summary": result.summary,
                        "degraded": result.degraded,
                        "sources": [s.model_dump(exclude={"content"}) for s in result.sources],
                    },
                )
