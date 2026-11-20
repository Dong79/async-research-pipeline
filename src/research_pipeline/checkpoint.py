"""Per-run state checkpointing.

After every DAG node completes, its result is persisted to ``<run_dir>/state.json``
(write to a temp file, then atomic ``os.replace`` — a crash mid-write can never leave a
half-written checkpoint). On resume the stored results are fed back to the executor
as ``precomputed`` nodes, so a run interrupted after 3 of 4 summaries only pays for the
fourth summary and the writer. Failed optional nodes are *not* stored, so resume
retries them.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .models import ReportResult, Source, SubTask, TaskResult


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def serialize(node_id: str, value: Any) -> Any:
    if node_id.startswith("search:"):
        return [s.model_dump() for s in value]
    return value.model_dump()


def deserialize(node_id: str, raw: Any) -> Any:
    if node_id.startswith("search:"):
        return [Source.model_validate(s) for s in raw]
    if node_id.startswith("summarize:"):
        return TaskResult.model_validate(raw)
    if node_id == "write":
        return ReportResult.model_validate(raw)
    raise KeyError(node_id)


class Checkpoint:
    """No-op when ``run_dir`` is None (benchmarks, tests)."""

    def __init__(self, run_dir: Path | None = None):
        self.path = run_dir / "state.json" if run_dir else None
        self._lock = asyncio.Lock()
        self.state: dict[str, Any] = {"plan": None, "nodes": {}}
        if self.path and self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def plan(self) -> list[SubTask] | None:
        raw = self.state.get("plan")
        return [SubTask.model_validate(t) for t in raw] if raw else None

    def precomputed(self) -> dict[str, Any]:
        return {nid: deserialize(nid, raw) for nid, raw in self.state["nodes"].items()}

    async def _flush(self) -> None:
        if self.path is None:
            return
        text = json.dumps(self.state, ensure_ascii=False)
        await asyncio.to_thread(_atomic_write, self.path, text)  # file I/O off the event loop

    async def save_plan(self, tasks: list[SubTask]) -> None:
        async with self._lock:
            self.state["plan"] = [t.model_dump() for t in tasks]
            await self._flush()

    async def save_node(self, node_id: str, value: Any) -> None:
        async with self._lock:
            self.state["nodes"][node_id] = serialize(node_id, value)
            await self._flush()
