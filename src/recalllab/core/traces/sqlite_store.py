"""SQLite-backed trace store for contract runs.

Each run is stored as one row with the full ``ContractRun`` serialised to
JSON in the ``data`` column. Cheap to instantiate; opens a fresh connection
per operation. Sufficient for v0.1 — no event-level queries needed yet.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from recalllab.core.traces.schema import ContractRun, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    contract_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


class TraceStore:
    """Append-only store of ``ContractRun`` records."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Defensive: write a .gitignore next to the trace database so users
        # don't accidentally commit traces (which contain raw memory text,
        # potentially PII or credentials). Tracking the .gitignore itself is
        # fine — only the trace files are excluded.
        gitignore = self.path.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n!.gitignore\n")
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def write_run(self, run: ContractRun) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs
                    (id, contract_id, provider, started_at, finished_at, status, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.contract_id,
                    run.provider,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.status.value,
                    run.model_dump_json(),
                ),
            )

    def get_run(self, run_id: str) -> ContractRun | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT data FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return ContractRun.model_validate_json(row[0])

    def list_runs(self, *, limit: int = 100) -> list[ContractRun]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT data FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ContractRun.model_validate_json(r[0]) for r in rows]

    def get_latest_run_by_status(self, status: RunStatus) -> ContractRun | None:
        """Return the most recent run matching ``status``, or ``None`` if none exist.

        Uses the ``idx_runs_status`` + ``idx_runs_started_at`` indexes so the
        search is bounded by the result, not by the size of the store. A
        previous paginate-and-filter implementation could silently miss
        older failed runs once a trace store accumulated more than 200
        passed runs since the last failure — the kind of bug that only
        bites in long-lived CI environments. ``recalllab record
        --latest-failure`` calls this directly.
        """
        with self._connect() as con:
            row = con.execute(
                """
                SELECT data
                FROM runs
                WHERE status = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (status.value,),
            ).fetchone()
        if row is None:
            return None
        return ContractRun.model_validate_json(row[0])
