"""Typed data contracts passed between agents, the DAG, the API and the eval harness."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    title: str
    intent: str
    query: str


class Source(BaseModel):
    """A retrieved document. ``id`` is the citation number the LLM sees ([1], [2], ...)."""

    id: int = 0
    title: str
    url: str
    content: str
    score: float = 0.0


class TaskResult(BaseModel):
    task: SubTask
    sources: list[Source] = Field(default_factory=list)
    summary: str = ""
    degraded: bool = False  # True when search/summary failed and we continued without it
    error: str | None = None


class ReportResult(BaseModel):
    topic: str
    markdown: str
    sources: list[Source]  # globally numbered sources the report's [n] markers point to
    tasks: list[TaskResult]
    citation_stats: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # server stopped mid-run; resumable from checkpoint


class Event(BaseModel):
    """One SSE event. ``id`` is monotonically increasing per run (used for Last-Event-ID)."""

    id: int
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    grounded: bool | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=32)
