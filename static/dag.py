"""A small async DAG executor.

Design goals:

* **Readiness-driven, not level-by-level.** A node starts the moment its own
  dependencies finish. A level-by-level scheduler (``gather`` all searches, then
  ``gather`` all summaries) waits for the *slowest* search before *any* summary
  begins; here ``summarize:t1`` can overlap with ``search:t3``.
* **Bounded concurrency.** A semaphore caps how many nodes run at once, so the
  pipeline cannot open 50 sockets to a rate-limited API. ``max_concurrency=1``
  turns the same graph into the sequential baseline used by the benchmark.
* **Partial-failure tolerance.** ``optional`` nodes may fail; their dependents
  receive a :class:`NodeFailure` instead of a value and decide how to degrade.
  A failing required node cancels everything still in flight (fail fast).
* **Resumability.** ``precomputed`` results (loaded from a checkpoint) are
  treated as already-finished nodes and never re-executed.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

NodeFn = Callable[[Mapping[str, Any]], Awaitable[Any]]
EventHook = Callable[[str, dict[str, Any]], Awaitable[None]]


class DAGError(Exception):
    pass


class CycleError(DAGError):
    pass


class NodeExecutionError(DAGError):
    def __init__(self, node_id: str, cause: BaseException):
        super().__init__(f"node {node_id!r} failed: {cause!r}")
        self.node_id = node_id
        self.cause = cause


@dataclass(frozen=True)
class NodeFailure:
    """Placeholder result for an optional node that failed."""

    node_id: str
    error: str


@dataclass
class Node:
    id: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    optional: bool = False


@dataclass
class DAG:
    nodes: dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise DAGError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Also validates that every dependency exists."""
        indeg = {nid: 0 for nid in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes.values():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise DAGError(f"{node.id!r} depends on unknown node {dep!r}")
                indeg[node.id] += 1
                children[dep].append(node.id)
        queue = deque(nid for nid, d in indeg.items() if d == 0)
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in children[nid]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            cyclic = sorted(nid for nid, d in indeg.items() if d > 0)
            raise CycleError(f"cycle detected among {cyclic}")
        return order


@dataclass
class NodeTiming:
    queued_at: float
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def run_seconds(self) -> float:
        return self.finished_at - self.started_at

    @property
    def wait_seconds(self) -> float:
        return self.started_at - self.queued_at


class DAGExecutor:
    def __init__(self, max_concurrency: int = 4, on_event: EventHook | None = None):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self.on_event = on_event
        self.timings: dict[str, NodeTiming] = {}

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(kind, data)

    async def run(
        self, dag: DAG, *, precomputed: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        order = dag.topological_order()
        results: dict[str, Any] = {k: v for k, v in (precomputed or {}).items() if k in dag.nodes}
        pending = [nid for nid in order if nid not in results]

        children: dict[str, list[str]] = defaultdict(list)
        waiting_on: dict[str, int] = {}
        for nid in pending:
            node = dag.nodes[nid]
            unmet = [d for d in node.deps if d not in results]
            waiting_on[nid] = len(unmet)
            for dep in unmet:
                children[dep].append(nid)

        for nid in results:
            await self._emit("node_skipped", {"node": nid, "reason": "checkpoint"})

        sem = asyncio.Semaphore(self.max_concurrency)

        async def execute(node: Node) -> Any:
            async with sem:
                self.timings[node.id].started_at = time.perf_counter()
                await self._emit("node_started", {"node": node.id})
                inputs = {d: results[d] for d in node.deps}
                return await node.fn(inputs)

        ready = deque(nid for nid in pending if waiting_on[nid] == 0)
        running: dict[asyncio.Task[Any], str] = {}
        try:
            while ready or running:
                while ready:
                    nid = ready.popleft()
                    self.timings[nid] = NodeTiming(queued_at=time.perf_counter())
                    task = asyncio.create_task(execute(dag.nodes[nid]), name=nid)
                    running[task] = nid

                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    nid = running.pop(task)
                    timing = self.timings[nid]
                    timing.finished_at = time.perf_counter()
                    exc = task.exception()
                    if exc is None:
                        results[nid] = task.result()
                        await self._emit(
                            "node_completed",
                            {
                                "node": nid,
                                "seconds": round(timing.run_seconds, 3),
                                "result": results[nid],
                            },
                        )
                    elif dag.nodes[nid].optional:
                        results[nid] = NodeFailure(nid, f"{type(exc).__name__}: {exc}")
                        await self._emit("node_failed", {"node": nid, "error": results[nid].error})
                    else:
                        await self._emit(
                            "node_failed", {"node": nid, "error": repr(exc), "fatal": True}
                        )
                        raise NodeExecutionError(nid, exc) from exc

                    for child in children[nid]:
                        waiting_on[child] -= 1
                        if waiting_on[child] == 0:
                            ready.append(child)
        finally:
            # Fail fast / external cancellation: never leak running tasks.
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
        return results
