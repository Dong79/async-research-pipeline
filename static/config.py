"""Configuration loaded from environment variables (and an optional .env file).

Everything is a plain frozen dataclass so it is cheap to copy with overrides
(``dataclasses.replace``) — the benchmark uses this to compare a sequential
run (max_concurrency=1) against the parallel DAG with otherwise identical settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments. Real env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    # --- LLM -----------------------------------------------------------------
    llm_provider: str = "openai"  # "openai" = any OpenAI-compatible API, or "fake" (offline)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    judge_model: str = ""  # defaults to llm_model; use a stronger model if you can

    # --- Search --------------------------------------------------------------
    search_provider: str = "fake"  # "tavily" | "duckduckgo" | "fake"
    tavily_api_key: str = ""
    results_per_query: int = 5

    # --- Concurrency & resilience ---------------------------------------------
    max_concurrency: int = 4  # DAG nodes in flight at once (1 == sequential baseline)
    llm_concurrency: int = 4  # concurrent LLM requests (provider rate limit)
    search_concurrency: int = 4  # concurrent search requests
    max_retries: int = 4  # attempts = 1 + retries
    retry_base_delay: float = 0.5  # seconds
    retry_max_delay: float = 8.0
    request_timeout: float = 60.0

    # --- Planning & context budgets --------------------------------------------
    min_tasks: int = 3
    max_tasks: int = 5
    summarizer_token_budget: int = 2500  # tokens of source text given to each summarizer
    per_source_token_cap: int = 600
    writer_token_budget: int = 5000  # tokens of task summaries given to the writer

    # --- Behaviour -----------------------------------------------------------
    grounded: bool = True  # strict citation prompts + post-hoc citation validation
    prune_context: bool = True  # relevance-ranked, token-budgeted source selection
    drop_uncited: bool = False  # grounded mode: delete body sentences that cite nothing
    checkpoint_dir: str = ".runs"
    fake_latency_scale: float = 1.0  # 0 makes the fake LLM/search instant (tests)
    fake_failure_rate: float = 0.0  # inject transient 429s to exercise retries

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.llm_model

    @classmethod
    def from_env(cls, dotenv: bool = True, **overrides: object) -> Settings:
        if dotenv:
            load_dotenv()
        values: dict[str, object] = {}
        for f in fields(cls):
            env = os.environ.get(f.name.upper())
            if env is None:
                continue
            if f.type in ("bool", bool):
                values[f.name] = env.strip().lower() in {"1", "true", "yes", "on"}
            elif f.type in ("int", int):
                values[f.name] = int(env)
            elif f.type in ("float", float):
                values[f.name] = float(env)
            else:
                values[f.name] = env
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
