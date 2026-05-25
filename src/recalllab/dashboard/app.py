"""FastAPI Failure Gallery — minimal read-only viewer for the trace store.

This module imports FastAPI/Jinja2 at top level and is therefore optional;
the CLI imports it lazily and prints an install hint when the
``[dashboard]`` extras aren't present.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from recalllab.core.traces.schema import (
    AssertionResult,
    CapabilitySkip,
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)
from recalllab.core.traces.sqlite_store import TraceStore

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def build_app(trace_path: Path) -> FastAPI:
    store = TraceStore(trace_path)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    trace_path_display = str(trace_path)

    app = FastAPI(
        title="RecallLab Failure Gallery",
        description="Read-only viewer for the local trace store.",
    )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        runs = store.list_runs(limit=500)
        groups: dict[str, list[dict[str, Any]]] = {
            "failed": [],
            "skipped": [],
            "passed": [],
            "errors": [],
        }
        for run in runs:
            view = _run_card(run)
            if run.status == RunStatus.FAILED:
                groups["failed"].append(view)
            elif run.status == RunStatus.SKIPPED:
                groups["skipped"].append(view)
            elif run.status == RunStatus.PASSED:
                groups["passed"].append(view)
            else:
                groups["errors"].append(view)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "trace_path": trace_path_display,
                "failed": groups["failed"],
                "skipped": groups["skipped"],
                "passed": groups["passed"],
                "errors": groups["errors"],
            },
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def detail(request: Request, run_id: str) -> Any:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "trace_path": trace_path_display,
                "run": _run_detail(run),
            },
        )

    return app


def run_server(trace_path: Path, host: str, port: int) -> None:
    """Build the app and serve it via uvicorn (blocking)."""
    import uvicorn

    app = build_app(trace_path)
    print(f"  trace store : {trace_path}")
    print(f"  dashboard at: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------- view helpers


def _run_card(run: ContractRun) -> dict[str, Any]:
    short_id = run.contract_id.split("::")[-1] or run.contract_id
    return {
        "id": run.id,
        "short_id": short_id,
        "contract_id": run.contract_id,
        "provider": run.provider,
        "events_count": len(run.events),
        "started_at_human": _format_ts(run.started_at),
        "failed_reason": _failed_reason(run),
        "last_recall_query": _last_recall_query(run.events),
        "capability_skips": run.capability_skips,
    }


def _run_detail(run: ContractRun) -> dict[str, Any]:
    duration_ms: float | None = None
    if run.finished_at is not None:
        duration_ms = (run.finished_at - run.started_at).total_seconds() * 1000

    # ``passed`` is three-valued from v0.2.2: True / False / None. ``None``
    # is a short-circuit placeholder (Decision #9) and must NOT render as
    # a failure in the Failure Gallery — ``not a.passed`` would catch it.
    failed_assertion_seqs = {
        a.sequence for a in run.assertions if a.passed is False
    }
    events_view = [
        {
            "sequence": ev.sequence,
            "kind": ev.kind.value,
            "payload_summary": _payload_summary(ev),
            "latency_ms": ev.latency_ms,
            "row_class": (
                "assert-fail"
                if ev.kind == EventKind.ASSERT and ev.sequence in failed_assertion_seqs
                else ""
            ),
        }
        for ev in run.events
    ]

    return {
        "id": run.id,
        "contract_id": run.contract_id,
        "short_id": run.contract_id.split("::")[-1] or run.contract_id,
        "provider": run.provider,
        "status_class": _status_class(run.status),
        "status_upper": run.status.value.upper(),
        "started_at_human": _format_ts(run.started_at),
        "duration_ms": duration_ms,
        "failed_reason": _failed_reason(run),
        "events": events_view,
        "capability_skips": run.capability_skips,
    }


def _failed_reason(run: ContractRun) -> str | None:
    # See _run_detail for the three-valued ``passed`` rationale.
    failed: list[AssertionResult] = [
        a for a in run.assertions if a.passed is False
    ]
    if failed:
        first = failed[0]
        return first.reason or f"{first.mode}({first.expected!r})"
    if run.status == RunStatus.FAILED:
        return (
            "pytest assertion or unhandled exception "
            "(no DSL assertion result captured)"
        )
    if run.status == RunStatus.ERROR:
        return "fixture or setup error before contract body ran"
    return None


def _last_recall_query(events: list[TraceEvent]) -> str | None:
    for ev in reversed(events):
        if ev.kind == EventKind.RECALL:
            query = ev.payload.get("query")
            if isinstance(query, str):
                return query
    return None


def _payload_summary(event: TraceEvent) -> str:
    payload = event.payload
    if event.kind == EventKind.GIVEN_USER:
        return f"user_id={payload.get('user_id')!r}"
    if event.kind == EventKind.REMEMBER:
        text = payload.get("text")
        return f"text={text!r}" if isinstance(text, str) else repr(payload)
    if event.kind == EventKind.RECALL:
        query = payload.get("query")
        results = payload.get("results")
        n_results = len(results) if isinstance(results, list) else 0
        return f"query={query!r}  →  {n_results} result(s)"
    if event.kind == EventKind.FORGET:
        return (
            f"matching={payload.get('matching')!r}, "
            f"deleted={payload.get('deleted')}"
        )
    if event.kind == EventKind.ASSERT:
        mode = payload.get("mode")
        passed = payload.get("passed")
        expected = payload.get("expected")
        verdict = "PASS" if passed else "FAIL"
        return f"{mode}({expected!r}) → {verdict}"
    if event.kind == EventKind.MUTATION:
        return repr(payload)
    return repr(payload)


def _format_ts(ts: datetime) -> str:
    formatted: str = ts.strftime("%Y-%m-%d %H:%M:%S %Z")
    return formatted.strip()


def _status_class(status: RunStatus) -> str:
    if status == RunStatus.FAILED:
        return "failed"
    if status == RunStatus.SKIPPED:
        return "skipped"
    if status == RunStatus.PASSED:
        return "passed"
    return "error"


__all__ = ["CapabilitySkip", "build_app", "run_server"]
