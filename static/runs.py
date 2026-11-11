"""Run lifecycle + the append-only event log that powers SSE.

Why an event *log* and not a queue?

* A queue has one consumer; a log can be read by any number of SSE clients.
* A reconnecting client sends ``Last-Event-ID`` and we replay from that offset —
  no events lost during a network blip.
* The pipeline (producer) never waits on a slow client: publishing is an append +
  ``notify_all``. Each subscriber reads at its own pace. The research job is a
  background task, so closing the browser tab does not cancel the research.
* The log is also written to ``events.jsonl``, so a finished (or interrupted) run can
  be replayed after a server restart.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .checkpoint import Checkpoint
from .config import Settings
from .llm import LLMClient
from .models import Event, ReportResult, ResearchRequest, RunStatus
from .pipeline import ResearchPipeline
from .search import SearchClient

log = logging.getLogger(__name__)

TERMINAL_EVENTS = {"done", "error"}


class EventLog:
    def __init__(self, path: Path | None = None):
        self.events: list[Event] = []
        self.closed = False
        self._cond = asyncio.Condition()
        self._path = path
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.events.append(Event.model_validate_json(line))
        self._fh = path.open("a", encoding="utf-8") if path else None

    async def publish(self, event_type: str, data: dict[str, Any]) -> Event:
        async with self._cond:
            event = Event(id=len(self.events) + 1, type=event_type, data=data)
            self.events.append(event)
            if self._fh:  # small append; OS-buffered, cheap enough to do inline
                self._fh.write(event.model_dump_json() + "\n")
                self._fh.flush()
            self._cond.notify_all()
        return event

    async def close(self) -> None:
        async with self._cond:
            self.closed = True
            self._cond.notify_all()
        if self._fh:
            self._fh.close()
            self._fh = None

    def reopen(self) -> None:
        self.closed = False
        if self._path and self._fh is None:
            self._fh = self._path.open("a", encoding="utf-8")

    async def subscribe(
        self, after_id: int = 0, heartbeat: float = 15.0
    ) -> AsyncIterator[Event | None]:
        """Yield events with id > ``after_id``; yield ``None`` as a heartbeat tick when idle."""
        cursor = max(0, after_id)  # ids are 1..n, so the next index to read == cursor
        while True:
            if cursor < len(self.events):
                batch = self.events[cursor:]
                cursor += len(batch)
                for ev in batch:
                    yield ev
                continue
            if self.closed:
                return
            timed_out = False
            async with self._cond:
                if cursor >= len(self.events) and not self.closed:
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=heartbeat)
                    except TimeoutError:
                        timed_out = True
            if timed_out:
                yield None


@dataclasses.dataclass
class RunHandle:
    id: str
    topic: str
    settings: Settings
    dir: Path
    log: EventLog
    status: RunStatus = RunStatus.PENDING
    task: asyncio.Task[None] | None = None
    result: ReportResult | None = None
    error: str | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "status": self.status.value,
            "error": self.error,
            "grounded": self.settings.grounded,
            "max_concurrency": self.settings.max_concurrency,
        }


class RunManager:
    def __init__(self, settings: Settings, llm: LLMClient, search: SearchClient):
        self.settings = settings
        self.llm = llm  # shared: one connection pool + one rate limit for all runs
        self.search = search
        self.root = Path(settings.checkpoint_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, RunHandle] = {}

    # ---- discovery after restart ---------------------------------------------------
    def load_existing(self) -> None:
        for meta_path in sorted(self.root.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            run_dir = meta_path.parent
            status = RunStatus(meta["status"])
            if status in (RunStatus.RUNNING, RunStatus.PENDING):
                status = RunStatus.INTERRUPTED  # the process died mid-run
            s = dataclasses.replace(
                self.settings,
                grounded=meta.get("grounded", True),
                max_concurrency=meta.get("max_concurrency", self.settings.max_concurrency),
            )
            handle = RunHandle(
                meta["id"],
                meta["topic"],
                s,
                run_dir,
                EventLog(run_dir / "events.jsonl"),
                status=status,
                error=meta.get("error"),
            )
            handle.log.closed = True
            if (run_dir / "result.json").exists():
                handle.result = ReportResult.model_validate_json(
                    (run_dir / "result.json").read_text(encoding="utf-8")
                )
            self.runs[handle.id] = handle
            self._write_meta(handle)

    # ---- lifecycle -----------------------------------------------------------------
    def start(self, req: ResearchRequest) -> RunHandle:
        overrides: dict[str, Any] = {}
        if req.grounded is not None:
            overrides["grounded"] = req.grounded
        if req.max_concurrency is not None:
            overrides["max_concurrency"] = req.max_concurrency
        s = dataclasses.replace(self.settings, **overrides)
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        handle = RunHandle(
            run_id, req.topic.strip(), s, run_dir, EventLog(run_dir / "events.jsonl")
        )
        self.runs[run_id] = handle
        self._launch(handle)
        return handle

    def resume(self, run_id: str) -> RunHandle:
        handle = self.runs[run_id]
        if handle.status not in (RunStatus.FAILED, RunStatus.INTERRUPTED):
            raise ValueError(
                f"run is {handle.status.value}; only failed/interrupted runs can resume"
            )
        handle.log.reopen()
        handle.error = None
        self._launch(handle, resumed=True)
        return handle

    def _launch(self, handle: RunHandle, resumed: bool = False) -> None:
        handle.status = RunStatus.RUNNING
        self._write_meta(handle)
        # Fire-and-forget background task: the HTTP request that started it returns at once.
        handle.task = asyncio.create_task(self._execute(handle, resumed), name=f"run-{handle.id}")

    async def _execute(self, handle: RunHandle, resumed: bool) -> None:
        if resumed:
            await handle.log.publish("resumed", {"from_checkpoint": True})
        pipeline = ResearchPipeline(
            handle.settings,
            self.llm,
            self.search,
            sink=handle.log.publish,
            checkpoint=Checkpoint(handle.dir),
        )
        try:
            handle.result = await pipeline.run(handle.topic)
            (handle.dir / "result.json").write_text(
                handle.result.model_dump_json(indent=2), encoding="utf-8"
            )
            (handle.dir / "report.md").write_text(handle.result.markdown, encoding="utf-8")
            handle.status = RunStatus.COMPLETED
            await handle.log.publish("done", {"status": "completed"})
        except asyncio.CancelledError:
            handle.status = RunStatus.INTERRUPTED
            await handle.log.publish(
                "error", {"message": "run interrupted (server shutdown); resumable"}
            )
            raise
        except Exception as exc:  # noqa: BLE001 - report every failure to the client
            log.exception("run %s failed", handle.id)
            handle.status = RunStatus.FAILED
            handle.error = f"{type(exc).__name__}: {exc}"
            await handle.log.publish("error", {"message": handle.error, "resumable": True})
        finally:
            self._write_meta(handle)
            await handle.log.close()

    async def shutdown(self) -> None:
        tasks = [h.task for h in self.runs.values() if h.task and not h.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _write_meta(self, handle: RunHandle) -> None:
        (handle.dir / "meta.json").write_text(json.dumps(handle.meta()), encoding="utf-8")
