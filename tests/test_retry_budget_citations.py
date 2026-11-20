import asyncio

import pytest

from research_pipeline.budget import count_tokens, fit_sections, normalize_url, prune_sources
from research_pipeline.citations import (
    extract_citations,
    remap_citations,
    split_claims,
    validate_citations,
)
from research_pipeline.jsonutil import parse_json_loose
from research_pipeline.models import Source
from research_pipeline.retry import PermanentError, RetryableError, RetryPolicy, retry_async

# ---------------------------------------------------------------- retry


def test_retry_succeeds_after_transient_errors_with_growing_backoff():
    attempts, sleeps = [], []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableError("429")
        return "ok"

    async def fake_sleep(d):
        sleeps.append(d)

    policy = RetryPolicy(max_retries=4, base_delay=1.0, max_delay=100)
    assert asyncio.run(retry_async(flaky, policy, sleep=fake_sleep)) == "ok"
    assert len(attempts) == 3 and len(sleeps) == 2
    assert 0 <= sleeps[0] <= 1.0 and 0 <= sleeps[1] <= 2.0  # full jitter within the cap


def test_retry_honours_retry_after_and_gives_up():
    sleeps = []

    async def always():
        raise RetryableError("429", retry_after=3.0)

    async def fake_sleep(d):
        sleeps.append(d)

    with pytest.raises(RetryableError):
        asyncio.run(retry_async(always, RetryPolicy(2, 0.1, 1.0), sleep=fake_sleep))
    assert sleeps == [3.0, 3.0]


def test_permanent_errors_are_not_retried():
    calls = []

    async def bad():
        calls.append(1)
        raise PermanentError("401")

    with pytest.raises(PermanentError):
        asyncio.run(retry_async(bad, RetryPolicy(5, 0, 0)))
    assert len(calls) == 1


def test_backoff_cap():
    p = RetryPolicy(10, 0.5, 4.0)
    assert all(p.backoff(n) <= 4.0 for n in range(1, 20))


# ---------------------------------------------------------------- budget


def _src(i, url, content, score=0.5):
    return Source(id=i, title=f"t{i}", url=url, content=content, score=score)


def test_prune_dedupes_ranks_and_respects_budget():
    relevant = "Solid state batteries use a solid electrolyte. " * 40
    noise = "Cookie banner and site navigation text. " * 40
    sources = [
        _src(1, "https://www.a.com/x/", noise, 0.9),
        _src(2, "https://a.com/x", noise, 0.9),  # same page as #1
        _src(3, "https://b.com/y", relevant, 0.5),
    ]
    kept, stats = prune_sources(
        sources, "solid state battery electrolyte", budget=300, per_source_cap=200
    )
    assert stats.duplicates_removed == 1
    assert kept[0].url == "https://b.com/y"  # relevance beats the engine's own score
    assert [s.id for s in kept] == list(range(1, len(kept) + 1))  # renumbered densely
    assert sum(count_tokens(s.content) for s in kept) <= 300


def test_normalize_url():
    assert normalize_url("https://WWW.Example.com/a/") == normalize_url(
        "https://example.com/a?utm=1"
    )


def test_fit_sections_is_fair():
    short, long = "short note. " * 5, "long note sentence. " * 400
    out = fit_sections([short, long], budget=300)
    assert out[0] == short
    assert count_tokens(out[1]) <= 300 - count_tokens(short) + 1


# ---------------------------------------------------------------- citations


def test_extract_and_remap():
    text = "A fact [1]. Another [2, 3]. Bad [9]."
    assert extract_citations(text) == [1, 2, 3, 9]
    assert remap_citations(text, {1: 10, 2: 11, 3: 12}) == "A fact [10]. Another [11][12]. Bad ."


def test_validate_removes_invalid_and_counts_uncited():
    md = "# T\n\n## Section\n\nThe battery market grew 30 percent last year [1][7].\nIt is widely believed to be the future of transport.\n\n## References\n[1] x"
    cleaned, stats = validate_citations(md, {1})
    assert "[7]" not in cleaned and "[1]" in cleaned
    assert stats.invalid_citations_removed == 1
    assert stats.uncited_claims == 1 and stats.cited_claims == 1


def test_split_claims_keeps_trailing_citations_and_skips_references():
    md = "## A\n\n- First claim about something important. [2] Second claim about another thing [3].\n\n## References\n[2] foo bar baz qux quux"
    claims = split_claims(md)
    assert [c.citations for c in claims] == [[2], [3]]


def test_parse_json_loose():
    assert parse_json_loose('Sure!\n```json\n{"tasks": [1, 2,]}\n```') == {"tasks": [1, 2]}
    assert parse_json_loose('prefix [{"a": 1}] suffix') == [{"a": 1}]
    with pytest.raises(ValueError):
        parse_json_loose("no json here")
