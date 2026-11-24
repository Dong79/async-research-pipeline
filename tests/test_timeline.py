import asyncio

from research_pipeline.config import Settings
from research_pipeline.llm import FakeLLM
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.search import FakeSearch
from research_pipeline.timeline import render_ascii, render_svg, sort_spans


def run(concurrency: int):
    s = Settings(fake_latency_scale=0.05, max_concurrency=concurrency)
    events = []

    async def sink(kind, data):
        events.append((kind, data))

    report = asyncio.run(
        ResearchPipeline(s, FakeLLM(s), FakeSearch(s), sink=sink).run("Heat pumps")
    )
    return report, events


def test_every_node_gets_a_complete_span():
    report, events = run(4)
    nodes = {sp["node"] for sp in report.timeline}
    tasks = {f"t{i}" for i in range(1, 5)}
    assert nodes == {"plan", "write"} | {f"search:{t}" for t in tasks} | {
        f"summarize:{t}" for t in tasks
    }
    for sp in report.timeline:
        assert sp["status"] == "done" and sp["queued"] <= sp["start"] <= sp["end"]
    # live events: one "running" and one "done" span event per node; report carries the timeline
    span_events = [d for k, d in events if k == "span"]
    assert len(span_events) == 2 * len(nodes)
    assert next(d for k, d in events if k == "report")["timeline"] == report.timeline


def _overlaps(a, b):
    return a["start"] < b["end"] - 0.002 and b["start"] < a["end"] - 0.002


def test_sequential_nodes_never_overlap_and_parallel_ones_do():
    seq, _ = run(1)
    dag_nodes = [sp for sp in seq.timeline if sp["node"] != "plan"]
    assert not any(_overlaps(a, b) for i, a in enumerate(dag_nodes) for b in dag_nodes[i + 1 :])

    par, _ = run(4)
    searches = [sp for sp in par.timeline if sp["stage"] == "search"]
    assert any(_overlaps(a, b) for i, a in enumerate(searches) for b in searches[i + 1 :])
    write = next(sp for sp in par.timeline if sp["node"] == "write")
    assert all(
        write["start"] >= sp["end"] - 0.002 for sp in par.timeline if sp["stage"] == "summarize"
    )


def test_renderers():
    report, _ = run(4)
    text = render_ascii(report.timeline)
    assert text.count("\n") == len(report.timeline) and "write" in text
    order = [sp["node"] for sp in sort_spans(report.timeline)]
    assert (
        order[0] == "plan" and order[-1] == "write" and order[1:3] == ["search:t1", "summarize:t1"]
    )
    svg = render_svg([("Parallel", report.timeline)], title="t")
    assert svg.startswith("<svg") and svg.count("<rect") > len(report.timeline)
