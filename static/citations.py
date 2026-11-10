"""Citation utilities: parse, remap, validate, and split text into checkable claims."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches [1], [2, 3], [2][3] is two matches.
CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_REF_HEADING = re.compile(
    r"^#{1,6}\s*(references|sources|bibliography|参考文献|来源)\b.*$", re.I | re.M
)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])(?:\s*(?:\[\d[\d,\s]*\])+)?\s+(?=[A-Z0-9\"“(*一-鿿])")


def extract_citations(text: str) -> list[int]:
    ids: list[int] = []
    for m in CITE_RE.finditer(text):
        for part in m.group(1).split(","):
            n = int(part.strip())
            if n not in ids:
                ids.append(n)
    return ids


def remap_citations(text: str, mapping: dict[int, int]) -> str:
    """Rewrite local citation numbers (per sub-task) to global ones; drop unknown numbers.

    Each summarizer numbers its own sources 1..k. Before the writer merges summaries,
    ``[1]`` from task A and ``[1]`` from task B must become different global ids,
    otherwise the final report silently attributes claims to the wrong source.
    """

    def sub(m: re.Match[str]) -> str:
        ids = [mapping[int(p)] for p in m.group(1).split(",") if int(p.strip()) in mapping]
        return "".join(f"[{i}]" for i in dict.fromkeys(ids))

    return CITE_RE.sub(sub, text)


def strip_references_section(markdown: str) -> str:
    """Remove any LLM-written references list; we append a canonical one instead."""
    m = _REF_HEADING.search(markdown)
    return markdown[: m.start()].rstrip() if m else markdown


@dataclass
class Claim:
    text: str
    citations: list[int] = field(default_factory=list)
    section: str = ""


def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)  # list markers
    return line.replace("**", "").replace("__", "").strip()


def split_claims(markdown: str, min_words: int = 5) -> list[Claim]:
    """Split a markdown report into sentence-level claims with their citations.

    Headings, code blocks, tables and the references section are skipped. Citation
    markers that trail a sentence (``... in 2023. [4]``) stay with that sentence.
    """
    body = strip_references_section(markdown)
    claims: list[Claim] = []
    section = ""
    in_code = False
    for raw in body.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not raw.strip() or raw.lstrip().startswith("|"):
            continue
        if raw.lstrip().startswith("#"):
            section = raw.lstrip("# ").strip()
            continue
        line = _clean_line(raw)
        for sent in _split_sentences(line):
            words = CITE_RE.sub("", sent).split()
            if len(words) < min_words:
                continue
            claims.append(
                Claim(text=sent.strip(), citations=extract_citations(sent), section=section)
            )
    return claims


def _split_sentences(line: str) -> list[str]:
    parts: list[str] = []
    last = 0
    for m in _SENT_SPLIT.finditer(line):
        parts.append(line[last : m.end()].strip())
        last = m.end()
    parts.append(line[last:].strip())
    return [p for p in parts if p]


@dataclass
class ValidationStats:
    invalid_citations_removed: int = 0
    uncited_claims: int = 0
    cited_claims: int = 0
    dropped_claims: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def validate_citations(
    markdown: str, valid_ids: set[int], drop_uncited: bool = False
) -> tuple[str, ValidationStats]:
    """Remove citation markers pointing at sources that do not exist; count uncited claims.

    With ``drop_uncited`` a body sentence that makes a claim but cites nothing is removed.
    """
    stats = ValidationStats()

    def fix(m: re.Match[str]) -> str:
        ids = [int(p) for p in m.group(1).split(",")]
        good = [i for i in ids if i in valid_ids]
        stats.invalid_citations_removed += len(ids) - len(good)
        return "".join(f"[{i}]" for i in good)

    cleaned = CITE_RE.sub(fix, markdown)

    for claim in split_claims(cleaned):
        if claim.citations:
            stats.cited_claims += 1
        else:
            stats.uncited_claims += 1
            if drop_uncited and claim.section.lower() not in {"overview", "introduction"}:
                cleaned = cleaned.replace(claim.text, "", 1)
                stats.dropped_claims += 1
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, stats
