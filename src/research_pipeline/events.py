"""Run-scoped event emission via a ContextVar.

The LLM and search clients are *shared* across concurrent runs (one connection
pool, one global rate limit), yet a retry inside a client should show up in the
SSE stream of the run that caused it. Passing an emitter through every call
signature is noisy, so the pipeline sets a ContextVar at the start of a run.
``asyncio.create_task`` copies the current context, so every node task spawned
for that run sees the same emitter — and runs never see each other's.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]

_current_sink: ContextVar[EventSink | None] = ContextVar("event_sink", default=None)


def set_sink(sink: EventSink | None):
    return _current_sink.set(sink)


def reset_sink(token) -> None:
    _current_sink.reset(token)


async def emit(event_type: str, data: dict[str, Any]) -> None:
    sink = _current_sink.get()
    if sink is not None:
        await sink(event_type, data)
