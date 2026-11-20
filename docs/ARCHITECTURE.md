# Architecture & design decisions

## Request lifecycle

1. `POST /api/runs` validates the topic, creates a `RunHandle` (run id, per-run settings,
   `EventLog`, run directory) and starts `RunManager._execute` with `asyncio.create_task`.
   The response (202) returns immediately.
2. `ResearchPipeline.run` sets a ContextVar event sink, then:
   - **plan** — `PlannerAgent` asks for JSON; malformed output is fed back for up to two
     self-repair attempts; tasks are validated, deduplicated, clipped to `MAX_TASKS`.
   - **DAG** — for each sub-task `search:tN → summarize:tN`, all feeding `write`.
3. `DAGExecutor` schedules ready nodes under a semaphore. Every completion triggers a
   checkpoint write and SSE events.
4. `write` merges summaries (global citation numbering), fits them into the writer budget,
   streams the report (`report_delta`), validates citations, appends references.
5. `done` / `error` closes the log; SSE subscribers drain remaining events and end.

## Why these choices

**Readiness-driven DAG over `asyncio.gather` per stage.** `gather(searches)` then
`gather(summaries)` has a barrier: all summaries wait for the slowest search. With readiness
scheduling, the critical path is `plan + max_i(search_i + summarize_i) + write` instead of
`plan + max(search) + max(summarize) + write`.

**Two levels of concurrency limits.** `MAX_CONCURRENCY` bounds DAG nodes in flight (and makes
the sequential baseline a config switch). `LLM_CONCURRENCY` / `SEARCH_CONCURRENCY` are
per-provider semaphores shared by *all runs* in the process, because rate limits are per API
key, not per run.

**Full-jitter backoff.** Parallel workers that hit a 429 together would retry together.
`delay = uniform(0, min(cap, base·2^attempt))`, raised to `Retry-After` if the server sent one.
Only `RetryableError` (408/409/425/429/5xx, timeouts, connection errors) is retried;
`CancelledError` is never swallowed.

**Streaming + retries.** A stream can be retried only before the first delta reaches the
client; afterwards a retry would duplicate text, so the error propagates.

**Event log instead of queue.** Multiple subscribers, replay by offset (`Last-Event-ID`),
producer never blocked by slow consumers, persisted to `events.jsonl` for post-restart replay.
Heartbeat comments every 15 s keep proxies/load balancers from closing idle connections;
`X-Accel-Buffering: no` stops nginx buffering. Client disconnect stops the *stream*, not the run.

**ContextVar for run-scoped events.** Clients are shared across runs; tasks inherit the
creating task's context, so a retry deep inside `llm.py` lands in the correct run's stream
without threading an emitter through every signature.

**Citation remapping.** Each summarizer numbers its sources 1..k. Merging summaries naively
makes `[1]` ambiguous across sections — a silent mis-attribution bug. Sources are deduped by
normalized URL into a global table and every summary is rewritten before the writer sees it;
after writing, cited sources are renumbered densely by first appearance.

**Deterministic validation instead of another LLM call.** Invalid citation ids are removed and
uncited claims counted with regexes — zero latency, zero cost. Semantic checking is left to
the judge in the offline eval.

**Atomic checkpoints.** `write tmp → os.replace` is atomic on POSIX/NTFS, so a crash never
leaves half a JSON file. Writes run in `asyncio.to_thread` to keep the loop responsive.

## Scaling beyond one process

| Concern | Today | Next |
|---|---|---|
| Event log | in-memory list + JSONL | Redis Streams (`XADD` / `XREAD` with ids = Last-Event-ID) |
| Checkpoints | local JSON | Postgres row per node or S3 object |
| Work execution | task in API process | queue (Celery/Arq/Temporal) + workers; API only streams |
| Rate limits | per-process semaphore | distributed token bucket (Redis) |
| Caching | none in serving path | cache search by (query, provider, day) |
