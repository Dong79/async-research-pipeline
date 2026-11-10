"""The three pipeline agents: Planner, Researcher (summarizer) and Writer.

Each agent is a thin, stateless wrapper around one prompt + one LLM call, which
keeps them independently testable and lets the DAG decide *when* they run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator

from . import prompts
from .config import Settings
from .jsonutil import parse_json_loose
from .llm import LLMClient
from .models import Source, SubTask
from .retry import PermanentError


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.id}] {s.title}\nURL: {s.url}\nContent: {s.content}" for s in sources)


class PlannerAgent:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm, self.s = llm, settings

    async def plan(self, topic: str, max_repairs: int = 2) -> list[SubTask]:
        """Ask for a JSON plan; on malformed output, feed the error back and retry (self-repair)."""
        messages = [
            {"role": "system", "content": prompts.PLANNER_SYSTEM},
            {
                "role": "user",
                "content": prompts.PLANNER_USER.format(
                    date=dt.date.today().isoformat(),
                    topic=topic,
                    min_tasks=self.s.min_tasks,
                    max_tasks=self.s.max_tasks,
                ),
            },
        ]
        last_error = ""
        for _ in range(max_repairs + 1):
            raw = await self.llm.complete(
                messages,
                purpose="plan",
                temperature=0.0,
                meta={"topic": topic, "max_tasks": self.s.max_tasks},
            )
            try:
                return self._validate(raw)
            except ValueError as exc:
                last_error = str(exc)
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": prompts.PLANNER_REPAIR.format(
                            error=last_error, min_tasks=self.s.min_tasks, max_tasks=self.s.max_tasks
                        ),
                    },
                ]
        raise PermanentError(f"planner failed after repairs: {last_error}")

    def _validate(self, raw: str) -> list[SubTask]:
        data = parse_json_loose(raw)
        items = data.get("tasks") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("expected a 'tasks' list")
        tasks: list[SubTask] = []
        seen_queries: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or not all(
                str(item.get(k, "")).strip() for k in ("title", "intent", "query")
            ):
                continue  # drop malformed entries rather than failing the whole plan
            q = " ".join(str(item["query"]).lower().split())
            if q in seen_queries:
                continue
            seen_queries.add(q)
            tasks.append(
                SubTask(
                    id=f"t{len(tasks) + 1}",
                    title=str(item["title"]).strip(),
                    intent=str(item["intent"]).strip(),
                    query=str(item["query"]).strip(),
                )
            )
        if len(tasks) < self.s.min_tasks:
            raise ValueError(f"only {len(tasks)} valid tasks, need at least {self.s.min_tasks}")
        return tasks[: self.s.max_tasks]


class ResearcherAgent:
    """Summarizes the (already pruned and numbered) sources for one sub-task."""

    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm, self.s = llm, settings

    async def summarize(self, task: SubTask, sources: list[Source]) -> str:
        if not sources:
            return "The sources do not address this sub-question."
        template = (
            prompts.GROUNDED_SUMMARIZER_USER
            if self.s.grounded
            else prompts.BASELINE_SUMMARIZER_USER
        )
        user = template.format(
            title=task.title,
            intent=task.intent,
            sources=format_sources(sources),
            valid_ids=", ".join(f"[{s.id}]" for s in sources),
        )
        return await self.llm.complete(
            [
                {"role": "system", "content": prompts.SUMMARIZER_SYSTEM},
                {"role": "user", "content": user},
            ],
            purpose="summarize",
            temperature=0.1,
            max_tokens=600,
            meta={"sources": [s.model_dump() for s in sources]},
        )


class WriterAgent:
    def __init__(self, llm: LLMClient, settings: Settings):
        self.llm, self.s = llm, settings

    def stream(
        self, topic: str, sections: list[tuple[str, str]], sources: list[Source]
    ) -> AsyncIterator[str]:
        sec_text = "\n\n".join(f"### {title}\n{summary}" for title, summary in sections)
        valid = ", ".join(f"[{s.id}]" for s in sources)
        if self.s.grounded:
            user = prompts.GROUNDED_WRITER_USER.format(
                topic=topic, sections=sec_text, valid_ids=valid
            )
        else:
            src_list = "\n".join(f"[{s.id}] {s.title} — {s.url}" for s in sources)
            user = prompts.BASELINE_WRITER_USER.format(
                topic=topic, sections=sec_text, source_list=src_list
            )
        return self.llm.stream(
            [
                {"role": "system", "content": prompts.WRITER_SYSTEM},
                {"role": "user", "content": user},
            ],
            purpose="write",
            temperature=0.3,
            max_tokens=2000,
            meta={"topic": topic, "sections": [{"title": t, "summary": s} for t, s in sections]},
        )
