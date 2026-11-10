"""Search backends behind one async interface, each with its own concurrency cap and retries."""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any, Protocol

import httpx

from .config import Settings
from .events import emit
from .llm import classify_http_error
from .models import Source
from .retry import RetryableError, RetryPolicy, retry_async


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int) -> list[Source]: ...

    async def aclose(self) -> None: ...


class _Base:
    name = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sem = asyncio.Semaphore(settings.search_concurrency)
        self._policy = RetryPolicy(
            settings.max_retries, settings.retry_base_delay, settings.retry_max_delay
        )

    async def search(self, query: str, max_results: int) -> list[Source]:
        async def attempt() -> list[Source]:
            async with self._sem:
                return await self._search_once(query, max_results)

        async def hook(n: int, exc: BaseException, delay: float) -> None:
            await emit(
                "retry",
                {
                    "where": f"search:{self.name}",
                    "attempt": n,
                    "error": str(exc),
                    "delay": round(delay, 2),
                },
            )

        return await retry_async(attempt, self._policy, on_retry=hook)

    async def _search_once(self, query: str, max_results: int) -> list[Source]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class TavilySearch(_Base):
    """https://docs.tavily.com — POST /search, Bearer auth."""

    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        super().__init__(settings)
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when SEARCH_PROVIDER=tavily")
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10))
        self._owns = http is None

    async def _search_once(self, query: str, max_results: int) -> list[Source]:
        body = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_raw_content": True,
        }
        try:
            resp = await self._http.post(
                self.URL,
                json=body,
                headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        classify_http_error(resp)
        out = []
        for r in resp.json().get("results", []):
            # raw_content (full page) is often much longer than the snippet -> pruning matters.
            content = (r.get("raw_content") or r.get("content") or "").strip()
            out.append(
                Source(
                    title=r.get("title") or r.get("url", ""),
                    url=r["url"],
                    content=content,
                    score=float(r.get("score") or 0),
                )
            )
        return out

    async def aclose(self) -> None:
        if self._owns:
            await self._http.aclose()


class DuckDuckGoSearch(_Base):
    """Free, no key. The ``ddgs`` library is synchronous, so it runs in a worker thread
    (``asyncio.to_thread``) to avoid blocking the event loop — and the SSE streams on it."""

    name = "duckduckgo"

    async def _search_once(self, query: str, max_results: int) -> list[Source]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install 'research-pipeline[ddg]' to use DuckDuckGo") from exc

        def run() -> list[dict[str, Any]]:
            return list(DDGS().text(query, max_results=max_results))

        try:
            rows = await asyncio.to_thread(run)
        except Exception as exc:  # ddgs raises its own RatelimitException etc.
            raise RetryableError(f"duckduckgo: {exc}") from exc
        n = len(rows) or 1
        return [
            Source(
                title=r.get("title", ""),
                url=r.get("href", ""),
                content=r.get("body", ""),
                score=1 - i / n,
            )
            for i, r in enumerate(rows)
            if r.get("href")
        ]


class FakeSearch(_Base):
    """Deterministic synthetic corpus with realistic latency, duplicates and noise."""

    name = "fake"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._rng = random.Random(11)

    async def _search_once(self, query: str, max_results: int) -> list[Source]:
        scale = self.settings.fake_latency_scale
        if scale > 0:
            await asyncio.sleep(self._rng.uniform(0.6, 1.4) * scale)
        if self.settings.fake_failure_rate and self._rng.random() < self.settings.fake_failure_rate:
            raise RetryableError("HTTP 503 (simulated)")
        key = hashlib.md5(query.encode()).hexdigest()[:8]
        words = query.split()
        subject = " ".join(words[:4])
        docs = []
        for i in range(max_results):
            facts = [
                f"{subject} was examined in a {2019 + i} study covering {120 + 37 * i} organizations.",
                f"According to the report, adoption of {subject} grew by {8 + 3 * i} percent year over year.",
                f"Researchers noted that {' '.join(words[-3:])} remains a central concern for practitioners.",
                f"The document also discusses {subject} in the context of cost, reliability and governance.",
            ]
            filler = " ".join(
                f"Additional background paragraph {j} about unrelated site navigation and cookies."
                for j in range(6)
            )
            docs.append(
                Source(
                    title=f"{subject.title()} — source {i + 1}",
                    url=f"https://example{i}.org/{key}/article",
                    content=" ".join(facts) + " " + filler,
                    score=round(1 - i * 0.12, 2),
                )
            )
        if docs:  # a syndicated duplicate with a cosmetically different URL
            dup = docs[0]
            docs.append(
                dup.model_copy(update={"url": dup.url.replace("https://", "https://www.") + "/"})
            )
        return docs


class CachedSearch:
    """Memoizes results per (query, max_results), optionally on disk.

    Used by the eval so baseline and grounded modes see *identical* sources —
    otherwise differences in retrieval would be confounded with the prompt change.
    """

    def __init__(self, inner: SearchClient, path: str | None = None):
        import json
        from pathlib import Path

        self.inner = inner
        self._path = Path(path) if path else None
        self._cache: dict[str, list[dict[str, Any]]] = {}
        if self._path and self._path.exists():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        self._locks: dict[str, asyncio.Lock] = {}

    async def search(self, query: str, max_results: int) -> list[Source]:
        import json

        key = f"{max_results}::{query}"
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # concurrent callers for the same key share one request
            if key not in self._cache:
                results = await self.inner.search(query, max_results)
                self._cache[key] = [s.model_dump() for s in results]
                if self._path:
                    self._path.write_text(json.dumps(self._cache), encoding="utf-8")
        return [Source.model_validate(s) for s in self._cache[key]]

    async def aclose(self) -> None:
        await self.inner.aclose()


def build_search(settings: Settings, http: httpx.AsyncClient | None = None) -> SearchClient:
    if settings.search_provider == "fake":
        return FakeSearch(settings)
    if settings.search_provider == "tavily":
        return TavilySearch(settings, http)
    if settings.search_provider in {"duckduckgo", "ddg"}:
        return DuckDuckGoSearch(settings)
    raise ValueError(f"unknown SEARCH_PROVIDER {settings.search_provider!r}")
