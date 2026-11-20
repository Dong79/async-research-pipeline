"""FastAPI app.

POST /api/runs                 start a run (returns 202 + run id immediately)
GET  /api/runs                 list runs
GET  /api/runs/{id}            status + final report
GET  /api/runs/{id}/events     SSE stream (supports Last-Event-ID / ?after=N replay)
POST /api/runs/{id}/resume     resume a failed/interrupted run from its checkpoint
GET  /                         minimal web UI

Start/stream are two endpoints on purpose: the browser's ``EventSource`` can only
issue GET requests without a body, so the topic is POSTed first and the stream is
then opened by id. It also means many viewers can watch one run.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .config import Settings
from .llm import build_llm
from .models import ResearchRequest
from .runs import RunManager
from .search import build_search

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, heartbeat_seconds: float = 15.0) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One pooled HTTP client for the whole process (keep-alive, fewer TLS handshakes).
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, connect=10),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        llm, search = build_llm(settings, http), build_search(settings, http)
        manager = RunManager(settings, llm, search)
        manager.load_existing()
        app.state.manager = manager
        try:
            yield
        finally:
            await manager.shutdown()
            await http.aclose()

    app = FastAPI(title="Async Research Pipeline", version="0.1.0", lifespan=lifespan)

    def manager(request: Request) -> RunManager:
        return request.app.state.manager

    def get_run(request: Request, run_id: str):
        handle = manager(request).runs.get(run_id)
        if handle is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return handle

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "llm": settings.llm_provider, "search": settings.search_provider}

    @app.post("/api/runs", status_code=202)
    async def start_run(req: ResearchRequest, request: Request):
        handle = manager(request).start(req)
        return {**handle.meta(), "events_url": f"/api/runs/{handle.id}/events"}

    @app.get("/api/runs")
    async def list_runs(request: Request):
        return [h.meta() for h in manager(request).runs.values()]

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str, request: Request):
        h = get_run(request, run_id)
        return {**h.meta(), "result": h.result.model_dump() if h.result else None}

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, request: Request):
        get_run(request, run_id)
        try:
            h = manager(request).resume(run_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return h.meta()

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        after: int = 0,
        last_event_id: str | None = Header(default=None),
    ):
        handle = get_run(request, run_id)
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after

        async def gen():
            yield "retry: 3000\n\n"  # browser reconnect delay hint
            async for ev in handle.log.subscribe(start, heartbeat=heartbeat_seconds):
                if await request.is_disconnected():
                    break  # stop *streaming*; the research task keeps running
                if ev is None:
                    yield ": keep-alive\n\n"  # comment line: keeps proxies from closing idle conns
                    continue
                payload = json.dumps(
                    {"type": ev.type, "data": ev.data, "ts": ev.ts}, ensure_ascii=False
                )
                yield f"id: {ev.id}\nevent: {ev.type}\ndata: {payload}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx response buffering
            },
        )

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


def main() -> None:  # console script: research-server
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("research_pipeline.api:create_app", factory=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
