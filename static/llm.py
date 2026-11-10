"""LLM clients: an OpenAI-compatible HTTP client and a deterministic offline fake.

"OpenAI-compatible" covers OpenAI, DeepSeek, Qwen (DashScope compatible mode),
Groq, Together, OpenRouter, Ollama, vLLM... — anything exposing
``POST {base_url}/chat/completions``.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .budget import count_tokens
from .config import Settings
from .events import emit
from .retry import PermanentError, RetryableError, RetryPolicy, retry_async

Messages = list[dict[str, str]]


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class LLMClient(Protocol):
    usage: Usage

    async def complete(
        self,
        messages: Messages,
        *,
        purpose: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        model: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: Messages,
        *,
        purpose: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


def classify_http_error(resp: httpx.Response) -> None:
    """Map HTTP status to retryable/permanent errors."""
    if resp.status_code < 400:
        return
    if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
        retry_after = None
        if ra := resp.headers.get("retry-after"):
            try:
                retry_after = float(ra)
            except ValueError:
                retry_after = None
        raise RetryableError(f"HTTP {resp.status_code}", retry_after=retry_after)
    raise PermanentError(f"HTTP {resp.status_code}: {resp.text[:500]}")


async def _on_retry(where: str, attempt: int, exc: BaseException, delay: float) -> None:
    await emit(
        "retry", {"where": where, "attempt": attempt, "error": str(exc), "delay": round(delay, 2)}
    )


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        self.settings = settings
        self.usage = Usage()
        self._url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, connect=10)
        )
        self._owns_http = http is None
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._policy = RetryPolicy(
            settings.max_retries, settings.retry_base_delay, settings.retry_max_delay
        )

    def _payload(
        self, messages: Messages, temperature: float, max_tokens: int | None, model: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return payload

    async def complete(
        self, messages, *, purpose, temperature=0.2, max_tokens=None, model=None, meta=None
    ) -> str:
        payload = self._payload(messages, temperature, max_tokens, model)

        async def attempt() -> str:
            async with self._sem:
                try:
                    resp = await self._http.post(self._url, json=payload, headers=self._headers)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
            classify_http_error(resp)
            data = resp.json()
            usage = data.get("usage") or {}
            self.usage.calls += 1
            self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
            self.usage.completion_tokens += usage.get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"] or ""

        async def hook(n: int, exc: BaseException, delay: float) -> None:
            self.usage.retries += 1
            await _on_retry(f"llm:{purpose}", n, exc, delay)

        return await retry_async(attempt, self._policy, on_retry=hook)

    async def stream(
        self, messages, *, purpose, temperature=0.3, max_tokens=None, meta=None
    ) -> AsyncIterator[str]:
        """Stream content deltas by consuming the provider's own SSE stream.

        Retries are only safe *before the first token*: once text has been forwarded
        to our client, retrying would duplicate it, so a mid-stream failure is raised.
        """
        payload = self._payload(messages, temperature, max_tokens, None) | {"stream": True}
        attempt = 0
        while True:
            emitted = False
            try:
                async with self._sem:
                    try:
                        async with self._http.stream(
                            "POST", self._url, json=payload, headers=self._headers
                        ) as resp:
                            if resp.status_code >= 400:
                                await resp.aread()
                                classify_http_error(resp)
                            out_chars = 0
                            async for line in resp.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                body = line[5:].strip()
                                if body == "[DONE]":
                                    break
                                chunk = json.loads(body)
                                choices = chunk.get("choices") or []
                                delta = (
                                    (choices[0].get("delta") or {}).get("content")
                                    if choices
                                    else None
                                )
                                if delta:
                                    emitted = True
                                    out_chars += len(delta)
                                    yield delta
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
                self.usage.calls += 1
                self.usage.prompt_tokens += sum(count_tokens(m["content"]) for m in messages)
                self.usage.completion_tokens += out_chars // 4
                return
            except RetryableError as exc:
                attempt += 1
                if emitted or attempt > self._policy.max_retries:
                    raise
                delay = self._policy.backoff(attempt)
                self.usage.retries += 1
                await _on_retry(f"llm:{purpose}", attempt, exc, delay)
                await asyncio.sleep(delay)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


# --------------------------------------------------------------------------------------
# Offline fake
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


class FakeLLM:
    """Deterministic stand-in so the whole system runs (and is tested) without API keys.

    Latency model ≈ real providers: time-to-first-token + prefill + per-token decode,
    scaled by ``fake_latency_scale``. Optionally injects transient 429s.
    It uses the structured ``meta`` the agents pass rather than parsing prompts.
    """

    TTFT, PREFILL_PER_TOKEN, DECODE_PER_TOKEN = 0.35, 0.00015, 0.010

    def __init__(self, settings: Settings, seed: int = 7):
        self.settings = settings
        self.usage = Usage()
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._rng = random.Random(seed)
        self._policy = RetryPolicy(
            settings.max_retries,
            settings.retry_base_delay * settings.fake_latency_scale,
            settings.retry_max_delay,
        )

    async def _simulate(
        self, prompt_tokens: int, completion_tokens: int, ttft_only: bool = False
    ) -> None:
        scale = self.settings.fake_latency_scale
        if scale <= 0:
            await asyncio.sleep(0)
            return
        decode = 0 if ttft_only else completion_tokens * self.DECODE_PER_TOKEN
        await asyncio.sleep((self.TTFT + prompt_tokens * self.PREFILL_PER_TOKEN + decode) * scale)

    def _maybe_fail(self) -> None:
        if self.settings.fake_failure_rate and self._rng.random() < self.settings.fake_failure_rate:
            raise RetryableError("HTTP 429 (simulated)", retry_after=None)

    async def complete(
        self, messages, *, purpose, temperature=0.2, max_tokens=None, model=None, meta=None
    ) -> str:
        meta = meta or {}
        prompt_tokens = sum(count_tokens(m["content"]) for m in messages)

        async def attempt() -> str:
            async with self._sem:
                self._maybe_fail()
                text = self._respond(purpose, meta)
                await self._simulate(prompt_tokens, count_tokens(text))
            self.usage.calls += 1
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += count_tokens(text)
            return text

        async def hook(n: int, exc: BaseException, delay: float) -> None:
            self.usage.retries += 1
            await _on_retry(f"llm:{purpose}", n, exc, delay)

        return await retry_async(attempt, self._policy, on_retry=hook)

    async def stream(
        self, messages, *, purpose, temperature=0.3, max_tokens=None, meta=None
    ) -> AsyncIterator[str]:
        meta = meta or {}
        prompt_tokens = sum(count_tokens(m["content"]) for m in messages)
        async with self._sem:
            await self._simulate(prompt_tokens, 0, ttft_only=True)
            text = self._respond(purpose, meta)
            pieces = re.findall(r"\S+\s*", text)
            per_piece = self.DECODE_PER_TOKEN * 1.3 * self.settings.fake_latency_scale
            for i in range(0, len(pieces), 8):
                if per_piece > 0:
                    await asyncio.sleep(per_piece * 8)
                yield "".join(pieces[i : i + 8])
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += count_tokens(text)

    async def aclose(self) -> None:
        return None

    # ---- canned behaviours ---------------------------------------------------------

    def _respond(self, purpose: str, meta: dict[str, Any]) -> str:
        if purpose == "plan":
            return self._plan(meta["topic"], meta.get("max_tasks", 4))
        if purpose == "summarize":
            return self._summarize(meta["sources"])
        if purpose == "write":
            return self._write(meta["topic"], meta["sections"])
        if purpose == "judge":
            return self._judge(meta["claims"], meta["sources"])
        return "OK"

    @staticmethod
    def _plan(topic: str, max_tasks: int) -> str:
        aspects = [
            (
                "Background and definitions",
                "Establish what the topic is and its key terms",
                "overview definition",
            ),
            (
                "Current state and key players",
                "Identify the latest developments and main actors",
                "latest developments",
            ),
            (
                "Evidence and real-world applications",
                "Find concrete examples, data, and use cases",
                "applications case studies data",
            ),
            (
                "Challenges, risks and outlook",
                "Understand limitations, debates, and future direction",
                "challenges risks future outlook",
            ),
        ][: max(3, min(max_tasks, 4))]
        tasks = [
            {"title": f"{t}: {topic}"[:90], "intent": i, "query": f"{topic} {q}"}
            for t, i, q in aspects
        ]
        return "Here is the plan:\n```json\n" + json.dumps({"tasks": tasks}, indent=2) + "\n```"

    @staticmethod
    def _summarize(sources: list[dict[str, Any]]) -> str:
        lines = []
        for s in sources[:5]:
            first = re.split(r"(?<=[.!?])\s+", s["content"].strip())[0]
            lines.append(f"- {first.rstrip('.')} [{s['id']}].")
        return "\n".join(lines) if lines else "The sources do not address this sub-question."

    @staticmethod
    def _write(topic: str, sections: list[dict[str, str]]) -> str:
        parts = [
            f"# {topic}",
            "",
            "## Overview",
            "",
            f"This report synthesizes findings on {topic} across {len(sections)} sub-questions.",
            "",
        ]
        for sec in sections:
            parts += [f"## {sec['title']}", "", sec["summary"], ""]
        parts += [
            "## Key takeaways",
            "",
            "- The findings above are drawn only from the cited sources.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _judge(claims: list[dict[str, Any]], sources: dict[str, str]) -> str:
        verdicts = []
        for c in claims:
            cited = [sources.get(str(i), "") for i in c["citations"]]
            claim_words = {w.lower() for w in _WORD.findall(c["text"]) if len(w) > 3}
            best = max(
                (
                    len(claim_words & {w.lower() for w in _WORD.findall(t)})
                    / (len(claim_words) or 1)
                    for t in cited
                ),
                default=0.0,
            )
            if not c["citations"]:
                label = "unsupported"
            elif best >= 0.6:
                label = "supported"
            elif best >= 0.3:
                label = "partially_supported"
            else:
                label = "unsupported"
            verdicts.append(
                {
                    "claim": c["n"],
                    "label": label,
                    "reason": f"lexical overlap {best:.2f} (fake judge)",
                }
            )
        return json.dumps({"verdicts": verdicts})


def build_llm(settings: Settings, http: httpx.AsyncClient | None = None) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM(settings)
    if settings.llm_provider in {"openai", "openai-compatible", "deepseek", "qwen"}:
        return OpenAICompatibleLLM(settings, http)
    raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}")
