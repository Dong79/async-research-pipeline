"""Exponential backoff with full jitter.

Why full jitter? If 8 parallel workers all hit a 429 at the same moment and all
sleep exactly 1s, 2s, 4s..., they retry in lock-step and collide again
("thundering herd"). Sampling the delay uniformly from [0, cap] spreads them out.
See: AWS Architecture Blog, "Exponential Backoff And Jitter".
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """Transient failure (429, 5xx, timeout). Optionally carries a server Retry-After hint."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Failure that will not get better by retrying (400, 401, bad JSON after repair...)."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0

    def backoff(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before retry number ``attempt`` (1-based), full jitter."""
        cap = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return (rng or random).uniform(0, cap)


RetryHook = Callable[[int, BaseException, float], Awaitable[None] | None]


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: RetryHook | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``fn`` until it succeeds, retrying only :class:`RetryableError`.

    Anything else (including ``asyncio.CancelledError``) propagates immediately —
    retrying a cancelled task would make the pipeline impossible to stop.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except RetryableError as exc:
            attempt += 1
            if attempt > policy.max_retries:
                raise
            delay = policy.backoff(attempt)
            if exc.retry_after is not None:
                delay = max(delay, min(exc.retry_after, policy.max_delay * 4))
            if on_retry is not None:
                res = on_retry(attempt, exc, delay)
                if asyncio.iscoroutine(res):
                    await res
            await sleep(delay)
