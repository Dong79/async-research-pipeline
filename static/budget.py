"""Token counting and token-budgeted context pruning.

The summarizer for each sub-task receives *at most* ``budget`` tokens of source
text. Pruning steps, in order:

1. Deduplicate by normalised URL and by content fingerprint (mirrors/syndication).
2. Score each source for relevance to the sub-task query (BM25-style term overlap
   blended with the search engine's own score).
3. Cap each source at ``per_source_cap`` tokens, cutting on a sentence boundary.
4. Greedily keep the highest-scoring sources until the budget is spent.
5. Renumber the survivors 1..k so citation markers stay dense and valid.

Smaller prompts are faster (latency scales with input tokens) and, empirically,
less noisy context reduces the chance the model cites the wrong source.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .models import Source

try:  # exact counts when tiktoken + its encoding file are available
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - depends on environment
    _ENC = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # Heuristic: ~4 chars/token for English, never fewer tokens than words.
    return max(len(text) // 4, len(text.split()))


_SENT_END = re.compile(r"(?<=[.!?。！？])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why will with".split()
)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim to ``max_tokens`` preferring a sentence boundary, never mid-word."""
    if count_tokens(text) <= max_tokens:
        return text
    out: list[str] = []
    used = 0
    for sent in _SENT_END.split(text):
        t = count_tokens(sent + " ")  # include the joining space so the sum bounds the result
        if used + t > max_tokens:
            break
        out.append(sent)
        used += t
    if out:
        return " ".join(out)
    # A single giant sentence: fall back to cutting on words.
    words = text.split()
    while words and count_tokens(" ".join(words)) > max_tokens:
        words = words[: max(1, int(len(words) * 0.8))]
        if len(words) == 1:
            break
    return " ".join(words) + " …"


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, "", ""))


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def relevance_scores(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """BM25 over the (tiny) candidate set, normalised to [0, 1]."""
    q_terms = set(_terms(query))
    if not docs or not q_terms:
        return [0.0] * len(docs)
    tokenized = [_terms(d) for d in docs]
    avgdl = sum(len(t) for t in tokenized) / len(tokenized) or 1.0
    n = len(docs)
    df = Counter(term for toks in tokenized for term in set(toks))
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        s = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / avgdl))
        scores.append(s)
    top = max(scores) or 1.0
    return [s / top for s in scores]


@dataclass
class PruneStats:
    input_sources: int
    kept_sources: int
    duplicates_removed: int
    tokens_before: int
    tokens_after: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def prune_sources(
    sources: list[Source], query: str, budget: int, per_source_cap: int
) -> tuple[list[Source], PruneStats]:
    tokens_before = sum(count_tokens(s.content) for s in sources)

    # 1. dedupe
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    unique: list[Source] = []
    for s in sources:
        u = normalize_url(s.url)
        fp = hashlib.sha1(" ".join(_terms(s.content))[:2000].encode()).hexdigest()
        if u in seen_urls or fp in seen_content or not s.content.strip():
            continue
        seen_urls.add(u)
        seen_content.add(fp)
        unique.append(s)

    # 2. score: 70% lexical relevance, 30% search-engine score
    lex = relevance_scores(query, [f"{s.title} {s.content}" for s in unique])
    engine_top = max((s.score for s in unique), default=0.0) or 1.0
    scored = sorted(
        zip(unique, lex, strict=True),
        key=lambda p: 0.7 * p[1] + 0.3 * (p[0].score / engine_top),
        reverse=True,
    )

    # 3 + 4. cap each, then greedy fill
    kept: list[Source] = []
    used = 0
    for src, _ in scored:
        text = truncate_to_tokens(src.content, per_source_cap)
        t = count_tokens(text)
        if used + t > budget:
            remaining = budget - used
            if remaining < 80:  # not worth a stub
                continue
            text = truncate_to_tokens(text, remaining)
            t = count_tokens(text)
        kept.append(src.model_copy(update={"content": text}))
        used += t

    # 5. renumber densely
    kept = [s.model_copy(update={"id": i}) for i, s in enumerate(kept, start=1)]
    return kept, PruneStats(
        input_sources=len(sources),
        kept_sources=len(kept),
        duplicates_removed=len(sources) - len(unique),
        tokens_before=tokens_before,
        tokens_after=used,
    )


def fit_sections(sections: list[str], budget: int) -> list[str]:
    """Shrink several text sections to fit a shared budget, fairly.

    Short sections keep everything; the leftover budget is redistributed to longer
    ones (water-filling), so one verbose summary cannot starve the others.
    """
    sizes = [count_tokens(s) for s in sections]
    if sum(sizes) <= budget:
        return sections
    alloc = [0] * len(sections)
    remaining_budget = budget
    pending = sorted(range(len(sections)), key=lambda i: sizes[i])
    while pending:
        share = remaining_budget // len(pending)
        i = pending[0]
        if sizes[i] <= share:
            alloc[i] = sizes[i]
            remaining_budget -= sizes[i]
            pending.pop(0)
        else:
            for j in pending:
                alloc[j] = share
            break
    return [truncate_to_tokens(s, a) for s, a in zip(sections, alloc, strict=True)]
