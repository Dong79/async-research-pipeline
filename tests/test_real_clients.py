"""The real-provider code paths, tested offline against mocked HTTP servers.

``httpx.MockTransport`` answers requests in-process, so these tests exercise the actual
OpenAI-compatible client (JSON + SSE streaming + retries + parameter adaptation), page
fetching/extraction and DuckDuckGo search wiring — without API keys or network.
"""

import asyncio
import json

import httpx
import pytest

from research_pipeline.budget import count_tokens, select_passages
from research_pipeline.config import Settings, load_dotenv
from research_pipeline.llm import OpenAICompatibleLLM, build_judge_llm
from research_pipeline.models import Source
from research_pipeline.pipeline import ResearchPipeline
from research_pipeline.retry import PermanentError
from research_pipeline.search import CachedSearch, DuckDuckGoSearch, FakeSearch
from research_pipeline.web import fetch_page_text, html_to_text

REAL = Settings(
    llm_provider="openai",
    llm_api_key="test-key",
    llm_base_url="https://llm.test/v1",
    retry_base_delay=0,
    fake_latency_scale=0,
)
FAST = Settings(fake_latency_scale=0)


def completion(content, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def sse(text):
    chunks = [text[i : i + 24] for i in range(0, len(text), 24)]
    body = "".join(
        f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks
    )
    return httpx.Response(
        200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )


def client(handler, settings=REAL):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleLLM(settings, http)


# ----------------------------------------------------------------- LLM client


def test_full_pipeline_through_the_real_llm_client():
    """Planner, 3 summarizers and the streaming writer, all via HTTP to a mock provider."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer test-key"
        system = body["messages"][0]["content"].lower()
        if body.get("stream"):
            return sse("# Batteries\n\n## Overview\n\nSolid electrolytes improve safety [1].\n")
        if "planner" in system:
            tasks = [
                {"title": f"Aspect {i}", "intent": "why", "query": f"battery aspect {i}"}
                for i in range(3)
            ]
            return completion(json.dumps({"tasks": tasks}))
        return completion("- Solid electrolytes improve safety [1].")

    llm = client(handler)
    report = asyncio.run(ResearchPipeline(REAL, llm, FakeSearch(FAST)).run("Solid-state batteries"))
    assert "Solid electrolytes improve safety [1]" in report.markdown
    assert "## References" in report.markdown and len(report.sources) == 1
    assert [bool(b.get("stream")) for b in seen].count(True) == 1 and len(seen) == 5
    assert llm.usage.calls == 5


def test_rate_limit_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return completion("hello")

    llm = client(handler)
    out = asyncio.run(llm.complete([{"role": "user", "content": "hi"}], purpose="t"))
    assert out == "hello" and llm.usage.retries == 1 and len(calls) == 2


def test_adapts_to_models_that_reject_max_tokens_and_temperature():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens' "
                        "is not supported with this model. Use 'max_completion_tokens' instead."
                    }
                },
            )
        if "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported value: 'temperature' does "
                        "not support 0.2 with this model. Only the default (1) value is supported."
                    }
                },
            )
        return completion("ok")

    llm = client(handler)
    msgs = [{"role": "user", "content": "hi"}]
    assert asyncio.run(llm.complete(msgs, purpose="t", max_tokens=50)) == "ok"
    assert len(bodies) == 3 and bodies[-1]["max_completion_tokens"] == 50
    asyncio.run(llm.complete(msgs, purpose="t", max_tokens=50))
    assert len(bodies) == 4  # remembered: second call succeeds first time


def test_concurrent_requests_in_flight_during_adaptation_are_resent():
    """Requests already sent with max_tokens when the first 400 arrives must not fail."""

    async def handler_async(request):
        body = json.loads(request.content)
        await asyncio.sleep(0.01)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens'. "
                        "Use 'max_completion_tokens' instead."
                    }
                },
            )
        return completion("ok")

    async def go():
        llm = client(handler_async)
        msgs = [{"role": "user", "content": "hi"}]
        return await asyncio.gather(
            *(llm.complete(msgs, purpose="t", max_tokens=9) for _ in range(6))
        )

    assert asyncio.run(go()) == ["ok"] * 6


def test_stream_adapts_before_first_token():
    def handler(request):
        body = json.loads(request.content)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                text="HTTP 400 Unsupported parameter: 'max_tokens'; use 'max_completion_tokens'",
            )
        return sse("streamed text works")

    async def go():
        llm = client(handler)
        return "".join(
            [
                d
                async for d in llm.stream(
                    [{"role": "user", "content": "x"}], purpose="w", max_tokens=99
                )
            ]
        )

    assert asyncio.run(go()) == "streamed text works"


def test_empty_output_from_reasoning_model_gives_clear_error():
    llm = client(lambda r: completion("", finish="length"))
    with pytest.raises(PermanentError, match="non-reasoning"):
        asyncio.run(llm.complete([{"role": "user", "content": "hi"}], purpose="t", max_tokens=20))


def test_bad_key_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    with pytest.raises(PermanentError, match="401"):
        asyncio.run(client(handler).complete([{"role": "user", "content": "hi"}], purpose="t"))
    assert len(calls) == 1


def test_judge_client_only_when_configured():
    assert build_judge_llm(REAL) is None
    import dataclasses

    j = build_judge_llm(
        dataclasses.replace(REAL, judge_base_url="https://other.test/v1", judge_model="big")
    )
    assert j is not None and j.settings.llm_model == "big" and j.settings.llm_api_key == "test-key"


def test_dotenv_handles_bom_and_inline_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '﻿LLM_MODEL=deepseek-chat  # cheap\nLLM_API_KEY="sk-abc # not a comment"\n', encoding="utf-8"
    )
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["LLM_MODEL"] == "deepseek-chat"
    assert os.environ["LLM_API_KEY"] == "sk-abc # not a comment"


# ----------------------------------------------------------------- web pages + search

PAGE = """<html><head><title>Battery news</title><script>var tracking = 1;</script></head>
<body><nav>Home | About | Contact us today for great deals and offers</nav>
<article><h1>Solid-state batteries</h1>
<p>Solid-state batteries replace the liquid electrolyte with a solid one, which improves safety considerably.</p>
<p>Toyota has said it plans to launch vehicles using the technology in 2027, according to the company.</p>
<p>Manufacturing cost remains the main obstacle, analysts said in a report published this year.</p></article>
<footer>Copyright 2026 all rights reserved by the publisher of this website</footer></body></html>"""


def test_html_to_text_keeps_article_drops_chrome():
    title, text = html_to_text(PAGE)
    assert title == "Battery news"
    assert "liquid electrolyte" in text and "Toyota" in text
    for junk in ("tracking", "Contact us", "Copyright"):
        assert junk not in text


def web_handler(request):
    if request.url.path == "/good":
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html; charset=utf-8"})
    if request.url.path == "/pdf":
        return httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"})
    return httpx.Response(404)


def test_fetch_page_text_handles_success_errors_and_non_html():
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(web_handler)) as http:
            good = await fetch_page_text(http, "https://site.test/good")
            missing = await fetch_page_text(http, "https://site.test/missing")
            pdf = await fetch_page_text(http, "https://site.test/pdf")
            return good, missing, pdf

    good, missing, pdf = asyncio.run(go())
    assert "Toyota" in good and missing is None and pdf is None


def test_duckduckgo_search_fetches_pages_and_falls_back_to_snippets(monkeypatch):
    class FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=10):
            if query == "nothing":
                raise Exception("No results found.")
            return [
                {"title": "Good", "href": "https://site.test/good", "body": "short snippet"},
                {"title": "Gone", "href": "https://site.test/missing", "body": "snippet kept"},
            ]

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(web_handler)) as http:
            s = DuckDuckGoSearch(FAST, http)
            return await s.search("solid state battery", 5), await s.search("nothing", 5)

    results, empty = asyncio.run(go())
    assert "Toyota" in results[0].content  # full page replaced the snippet
    assert results[1].content == "snippet kept"  # fetch failed: snippet stays
    assert empty == []  # "no results" is an answer, not an error to retry


def test_select_passages_keeps_relevant_text_from_deep_in_a_page():
    intro = "\n".join(
        f"Welcome paragraph {i} about our newsletter, events and site membership."
        for i in range(30)
    )
    fact = "The solid electrolyte reached 25 mS/cm ionic conductivity in the 2024 trial."
    text = intro + "\n" + fact + "\n" + intro
    out = select_passages(text, "solid electrolyte ionic conductivity", 120)
    assert fact in out and count_tokens(out) <= 125


def test_cached_search_appends_jsonl_and_reloads(tmp_path):
    calls = []

    class Counting(FakeSearch):
        async def _search_once(self, query, max_results):
            calls.append(query)
            return await super()._search_once(query, max_results)

    path = tmp_path / "cache.jsonl"
    c1 = CachedSearch(Counting(FAST), str(path))
    asyncio.run(c1.search("q one", 3))
    asyncio.run(c1.search("q one", 3))
    c2 = CachedSearch(Counting(FAST), str(path))
    again = asyncio.run(c2.search("q one", 3))
    assert calls == ["q one"] and isinstance(again[0], Source)
    assert len(path.read_text().splitlines()) == 1
