"""LLM-as-a-Judge citation-grounding audit.

For one report:
1. Split it into sentence-level claims (deterministic — see ``citations.split_claims``).
2. Send batches of claims, each with the *excerpts of only the sources it cites*, to a
   judge model at temperature 0 and ask for a label per claim.
3. Aggregate: unsupported-claim rate, citation precision, etc.

Known judge failure modes and the mitigations used here:
* Position/verbosity bias  -> per-claim labels against fixed evidence, no pairwise A/B.
* Judge using world knowledge -> prompt restricts it to cited excerpts only.
* Run-to-run noise -> temperature 0, and the eval compares modes on the *same* queries.
* Judge != human -> ``eval/agreement.py`` measures Cohen's kappa on a hand-labelled sample.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .budget import truncate_to_tokens
from .citations import split_claims
from .jsonutil import parse_json_loose
from .llm import LLMClient
from .models import ReportResult

LABELS = ("supported", "partially_supported", "unsupported", "not_a_claim")


@dataclass
class ClaimVerdict:
    n: int
    text: str
    citations: list[int]
    label: str
    reason: str = ""


@dataclass
class GroundingScore:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def counts(self) -> Counter[str]:
        return Counter(v.label for v in self.verdicts)

    @property
    def n_claims(self) -> int:
        return sum(1 for v in self.verdicts if v.label not in ("not_a_claim", "judge_error"))

    @property
    def unsupported_rate(self) -> float:
        return self.counts["unsupported"] / self.n_claims if self.n_claims else 0.0

    @property
    def supported_rate(self) -> float:
        return self.counts["supported"] / self.n_claims if self.n_claims else 0.0

    @property
    def citation_precision(self) -> float:
        """Of claims that cite something, how many are fully supported by what they cite."""
        cited = [
            v
            for v in self.verdicts
            if v.citations and v.label not in ("not_a_claim", "judge_error")
        ]
        return sum(v.label == "supported" for v in cited) / len(cited) if cited else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "n_claims": self.n_claims,
            **{k: self.counts[k] for k in (*LABELS, "judge_error")},
            "uncited_claims": sum(
                1
                for v in self.verdicts
                if not v.citations and v.label not in ("not_a_claim", "judge_error")
            ),
            "unsupported_rate": round(self.unsupported_rate, 4),
            "supported_rate": round(self.supported_rate, 4),
            "citation_precision": round(self.citation_precision, 4),
        }


class GroundingJudge:
    def __init__(
        self,
        llm: LLMClient,
        model: str | None = None,
        batch_size: int = 20,
        excerpt_tokens: int = 700,
        concurrency: int = 4,
    ):
        self.llm = llm
        self.model = model
        self.batch_size = batch_size
        self.excerpt_tokens = excerpt_tokens
        self._sem = asyncio.Semaphore(concurrency)

    async def score(self, report: ReportResult) -> GroundingScore:
        claims = split_claims(report.markdown)
        source_text = {
            s.id: truncate_to_tokens(s.content, self.excerpt_tokens) for s in report.sources
        }
        batches = [claims[i : i + self.batch_size] for i in range(0, len(claims), self.batch_size)]
        results = await asyncio.gather(
            *(
                self._judge_batch(b, offset * self.batch_size, source_text)
                for offset, b in enumerate(batches)
            )
        )
        return GroundingScore([v for batch in results for v in batch])

    async def _judge_batch(
        self, claims, offset: int, source_text: dict[int, str]
    ) -> list[ClaimVerdict]:
        numbered = [(offset + i + 1, c) for i, c in enumerate(claims)]
        cited_ids = sorted({i for _, c in numbered for i in c.citations if i in source_text})
        sources_block = (
            "\n\n".join(f"[{i}] {source_text[i]}" for i in cited_ids) or "(no sources cited)"
        )
        claims_block = "\n".join(
            f"{n}. {c.text}  -- cites: {', '.join(map(str, c.citations)) or 'NOTHING'}"
            for n, c in numbered
        )
        messages = [
            {"role": "system", "content": prompts.JUDGE_SYSTEM},
            {
                "role": "user",
                "content": prompts.JUDGE_USER.format(sources=sources_block, claims=claims_block),
            },
        ]
        meta = {
            "claims": [{"n": n, "text": c.text, "citations": c.citations} for n, c in numbered],
            "sources": {str(i): source_text[i] for i in cited_ids},
        }
        async with self._sem:
            raw = await self.llm.complete(
                messages, purpose="judge", temperature=0.0, model=self.model, meta=meta
            )
        try:
            by_n = {int(v["claim"]): v for v in parse_json_loose(raw).get("verdicts", [])}
        except (ValueError, AttributeError, KeyError, TypeError):
            by_n = {}
        out = []
        for n, c in numbered:
            v = by_n.get(n, {})
            label = v.get("label", "").strip().lower()
            if label not in LABELS:
                # Excluded from rates rather than guessed; the count is reported so a flaky
                # judge is visible instead of silently skewing the metric.
                label, reason = "judge_error", "judge returned no usable verdict"
            else:
                reason = v.get("reason", "")
            # Rule: a factual claim with an invalid citation number cannot be grounded.
            if label == "supported" and any(i not in source_text for i in c.citations):
                label, reason = "partially_supported", "cites a non-existent source"
            out.append(ClaimVerdict(n, c.text, c.citations, label, reason))
        return out
