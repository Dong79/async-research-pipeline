import asyncio
import time

import pytest

from research_pipeline.dag import (
    DAG,
    CycleError,
    DAGExecutor,
    Node,
    NodeExecutionError,
    NodeFailure,
)


def sleeper(value, seconds=0.05, log=None):
    async def fn(inputs):
        if log is not None:
            log.append(("start", value, time.perf_counter()))
        await asyncio.sleep(seconds)
        return value

    return fn


def test_topological_order_and_cycle_detection():
    dag = DAG()
    dag.add(Node("a", sleeper("a")))
    dag.add(Node("b", sleeper("b"), deps=("a",)))
    assert dag.topological_order() == ["a", "b"]

    cyclic = DAG()
    cyclic.add(Node("x", sleeper("x"), deps=("y",)))
    cyclic.add(Node("y", sleeper("y"), deps=("x",)))
    with pytest.raises(CycleError):
        cyclic.topological_order()


def test_parallel_branches_respect_concurrency_bound():
    in_flight = 0
    peak = 0

    async def tracked(_):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1

    dag = DAG()
    for i in range(8):
        dag.add(Node(f"n{i}", tracked))
    asyncio.run(DAGExecutor(max_concurrency=3).run(dag))
    assert peak == 3


def test_parallel_is_faster_than_sequential():
    def build():
        dag = DAG()
        for i in range(4):
            dag.add(Node(f"s{i}", sleeper(i, 0.1)))
            dag.add(Node(f"m{i}", sleeper(i, 0.1), deps=(f"s{i}",)))
        dag.add(Node("w", sleeper("w", 0.05), deps=tuple(f"m{i}" for i in range(4))))
        return dag

    t0 = time.perf_counter()
    asyncio.run(DAGExecutor(1).run(build()))
    seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    asyncio.run(DAGExecutor(4).run(build()))
    par = time.perf_counter() - t0
    assert seq > 0.8 and par < 0.4


def test_readiness_scheduling_does_not_wait_for_slowest_sibling():
    """summarize(fast) must start before search(slow) finishes."""
    log = []
    dag = DAG()
    dag.add(Node("search_fast", sleeper("f", 0.02, log)))
    dag.add(Node("search_slow", sleeper("s", 0.3, log)))
    dag.add(Node("sum_fast", sleeper("sf", 0.01, log), deps=("search_fast",)))
    dag.add(Node("sum_slow", sleeper("ss", 0.01, log), deps=("search_slow",)))
    start = time.perf_counter()
    asyncio.run(DAGExecutor(4).run(dag))
    sf_start = next(t for kind, v, t in log if v == "sf")
    assert sf_start - start < 0.15


def test_inputs_are_passed_from_dependencies():
    async def add(inputs):
        return inputs["a"] + inputs["b"]

    dag = DAG()
    dag.add(Node("a", sleeper(2, 0)))
    dag.add(Node("b", sleeper(3, 0)))
    dag.add(Node("c", add, deps=("a", "b")))
    assert asyncio.run(DAGExecutor(2).run(dag))["c"] == 5


def test_optional_failure_degrades_and_required_failure_cancels():
    async def boom(_):
        raise RuntimeError("nope")

    async def consumer(inputs):
        return isinstance(inputs["bad"], NodeFailure)

    dag = DAG()
    dag.add(Node("bad", boom, optional=True))
    dag.add(Node("after", consumer, deps=("bad",)))
    assert asyncio.run(DAGExecutor(2).run(dag))["after"] is True

    cancelled = asyncio.Event()

    async def long(_):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def go():
        dag = DAG()
        dag.add(Node("fatal", boom))
        dag.add(Node("long", long))
        with pytest.raises(NodeExecutionError):
            await DAGExecutor(2).run(dag)
        assert cancelled.is_set()

    asyncio.run(go())


def test_precomputed_nodes_are_not_rerun():
    calls = []

    async def counted(inputs):
        calls.append(1)
        return "fresh"

    dag = DAG()
    dag.add(Node("a", counted))
    dag.add(Node("b", sleeper("b", 0), deps=("a",)))
    out = asyncio.run(DAGExecutor(2).run(dag, precomputed={"a": "cached"}))
    assert out["a"] == "cached" and not calls
