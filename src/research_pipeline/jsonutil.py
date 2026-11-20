"""Tolerant JSON extraction from LLM output (code fences, chatty prefixes, trailing commas)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json_loose(text: str) -> Any:
    candidates: list[str] = []
    candidates += [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    spans = []
    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        s, e = text.find(start_ch), text.rfind(end_ch)
        if s != -1 and e > s:
            spans.append((s, text[s : e + 1]))
    candidates += [c for _, c in sorted(spans)]  # outermost structure first
    last_err: Exception | None = None
    for cand in candidates:
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(attempt.strip())
            except json.JSONDecodeError as exc:
                last_err = exc
    raise ValueError(f"no valid JSON found in model output: {last_err}")
