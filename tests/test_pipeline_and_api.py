import asyncio
import json

import httpx
import pytest

from research_pipeline.api import create_app
from research_pipeline.checkpoint import Checkpoint
from research_pipeline.config import Settings
from research_pipeline.judge import GroundingJudge
from research_pipeline.llm import FakeLLM
from research_pipeline.models import ResearchRequest
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.runs import EventLog
from research_pipeline.search import FakeSearch, SearchClient

FAST = Settings(fake_latency_scale=0, checkpoint_dir="unused")


def test_pipeline_end_to_end_offline():
    events = []

    async def sink(t, d):
        events.append(t)

    report = asyncio.run(
        ResearchPipeline(FAST, FakeLLM(FAST), FakeSearch(FAST), sink=sink).run(
            "Heat pumps in cold climates"
        )
    )
    assert report.markdown.startswith("# Heat pumps")
    assert "## References" in report.markdown
    # every citation in the body points at a listed source
    ids = {s.id for s in report.sources}
    assert report.citation_stats["invalid_citations_removed"] == 0
    assert ids == set(range(1, len(ids) + 1))
    for needed in ("plan", "task_status", "task_summary", "report_delta", "report"):
        assert needed in events


def test_retries_are_surfaced_and_run_still_succeeds():
    s = Settings(fake_latency_scale=0, fake_failure_rate=0.3, max_retries=8, retry_base_delay=0)
    retries = []

    async def sink(t, d):
        if t == "retry":
            retries.append(d)

    asyncio.run(ResearchPipeline(s, FakeLLM(s), FakeSearch(s), sink=sink).run("Offshore wind"))
    assert retries, "expected injected 429/503s to be retried"


class BrokenSearch:
    def __init__(self, inner: SearchClient, bad_word: str):
        self.inner, self.bad = inner, bad_word

    async def search(self, query, max_results):
        if self.bad in query:
            raise RuntimeError("search backend down")
        return await self.inner.search(query, max_results)

    async def aclose(self):
        pass


def test_one_failed_branch_degrades_instead_of_failing():
    search = BrokenSearch(FakeSearch(FAST), "challenges")
    report = asyncio.run(ResearchPipeline(FAST, FakeLLM(FAST), search).run("Lithium mining"))
    assert report.metrics["degraded_tasks"] == 1
    assert sum(t.degraded for t in report.tasks) == 1


def test_checkpoint_resume_skips_completed_nodes(tmp_path):
    calls = []

    class CountingSearch(FakeSearch):
        async def _search_once(self, query, max_results):
            calls.append(query)
            return await super()._search_once(query, max_results)

    ckpt = Checkpoint(tmp_path)
    asyncio.run(
        ResearchPipeline(FAST, FakeLLM(FAST), CountingSearch(FAST), checkpoint=ckpt).run(
            "Geothermal energy"
        )
    )
    first = len(calls)

    # Simulate a crash after searches: drop everything except plan + search results.
    state = json.loads((tmp_path / "state.json").read_text())
    state["nodes"] = {k: v for k, v in state["nodes"].items() if k.startswith("search:")}
    (tmp_path / "state.json").write_text(json.dumps(state))

    asyncio.run(
        ResearchPipeline(
            FAST, FakeLLM(FAST), CountingSearch(FAST), checkpoint=Checkpoint(tmp_path)
        ).run("Geothermal energy")
    )
    assert len(calls) == first  # no search re-executed on resume


def test_judge_scores_report():
    report = asyncio.run(
        ResearchPipeline(FAST, FakeLLM(FAST), FakeSearch(FAST)).run("Congestion pricing")
    )
    score = asyncio.run(GroundingJudge(FakeLLM(FAST)).score(report))
    assert score.n_claims > 5
    assert 0 <= score.unsupported_rate < 0.5
    assert score.summary()["judge_error"] == 0


def test_event_log_replay_and_heartbeat():
    async def go():
        log = EventLog()
        for i in range(5):
            await log.publish("tick", {"i": i})
        got = []

        async def reader():
            async for ev in log.subscribe(after_id=3, heartbeat=0.05):
                got.append(None if ev is None else ev.id)
                if ev is not None and ev.id == 6:
                    break

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.12)  # idle -> heartbeats
        await log.publish("tick", {"i": 5})
        await asyncio.wait_for(task, 1)
        assert got[:2] == [4, 5] and None in got and got[-1] == 6

    asyncio.run(go())


@pytest.mark.parametrize("grounded", [True, False])
def test_api_sse_stream_and_last_event_id(tmp_path, grounded):
    settings = Settings(fake_latency_scale=0, checkpoint_dir=str(tmp_path))
    app = create_app(settings, heartbeat_seconds=0.2)

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                r = await c.post(
                    "/api/runs",
                    json=ResearchRequest(topic="Passkeys adoption", grounded=grounded).model_dump(),
                )
                assert r.status_code == 202
                run_id = r.json()["id"]

                r = await c.get(f"/api/runs/{run_id}/events")
                assert r.headers["content-type"].startswith("text/event-stream")
                ids = [int(line[4:]) for line in r.text.splitlines() if line.startswith("id: ")]
                types = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
                assert types[0] == "run_started" and types[-1] == "done"
                assert ids == list(range(1, len(ids) + 1))

                r = await c.get(
                    f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(ids[-3])}
                )
                replay = [int(line[4:]) for line in r.text.splitlines() if line.startswith("id: ")]
                assert replay == ids[-2:]

                detail = (await c.get(f"/api/runs/{run_id}")).json()
                assert detail["status"] == "completed" and detail["grounded"] is grounded

    asyncio.run(go())
