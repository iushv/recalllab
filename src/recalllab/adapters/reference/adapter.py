"""In-process SQLite reference memory adapter.

Uses FTS5 for retrieval when available (the standard CPython build supports
it on macOS, most Linux distros, and the official Windows installer); falls
back to an in-process keyword-overlap retriever otherwise. Declares all
capability flags True — provenance, scores, forget, tenant-delete, and
candidate trace are all natively supported, so all six v0.1 example
contracts run green against it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from recalllab.adapters.base import CapabilityFlags, Episode, Recalled

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
"""

_FTS5_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    text,
    content='episodes',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text)
        VALUES('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text)
        VALUES('delete', old.rowid, old.text);
    INSERT INTO episodes_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _has_fts5() -> bool:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE _probe USING fts5(x);")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


_FTS5_AVAILABLE = _has_fts5()


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)}


class ReferenceMemoryAdapter:
    """SQLite-backed reference memory provider."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._con = sqlite3.connect(path)
        self._fts5 = _FTS5_AVAILABLE
        self._init_schema()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ReferenceMemoryAdapter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._con.executescript(_BASE_SCHEMA)
        if self._fts5:
            self._con.executescript(_FTS5_SCHEMA)
        self._con.commit()

    # ----------------------------------------------------------------- protocol
    def capabilities(self) -> CapabilityFlags:
        return CapabilityFlags(
            supports_forget=True,
            supports_tenant_delete=True,
            supports_provenance=True,
            supports_scores=True,
            supports_candidate_trace=True,
            supports_cost_trace=False,
        )

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        eid = episode_id or uuid.uuid4().hex
        created_at = datetime.now(tz=UTC)
        self._con.execute(
            """
            INSERT INTO episodes (id, user_id, text, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                eid,
                user_id,
                text,
                created_at.isoformat(),
                json.dumps(metadata) if metadata is not None else None,
            ),
        )
        self._con.commit()
        return Episode(
            id=eid,
            user_id=user_id,
            text=text,
            created_at=created_at,
            metadata=metadata,
        )

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        k: int = 5,
    ) -> list[Recalled]:
        if not query.strip():
            return []
        if self._fts5:
            return self._recall_fts5(user_id, query, k)
        return self._recall_fallback(user_id, query, k)

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        if episode_id is None and matching is None:
            raise ValueError("forget requires either matching= or episode_id=")
        if episode_id is not None:
            cursor = self._con.execute(
                "DELETE FROM episodes WHERE user_id = ? AND id = ?",
                (user_id, episode_id),
            )
            self._con.commit()
            return cursor.rowcount
        # Match-mode: do NOT pass user input through SQL LIKE — '%' and '_'
        # would act as wildcards, so matching='%' would delete every memory
        # for the user. Filter literally in Python, then delete by id.
        assert matching is not None  # narrowed by the early-return above
        needle = matching.lower()
        candidates = self._con.execute(
            "SELECT id, text FROM episodes WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        ids_to_delete = [eid for eid, text in candidates if needle in text.lower()]
        if not ids_to_delete:
            return 0
        placeholders = ",".join("?" * len(ids_to_delete))
        cursor = self._con.execute(
            f"DELETE FROM episodes WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *ids_to_delete),
        )
        self._con.commit()
        return cursor.rowcount

    def list_episodes(self, user_id: str) -> list[Episode]:
        rows = self._con.execute(
            """
            SELECT id, user_id, text, created_at, metadata
            FROM episodes WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def delete_user(self, user_id: str) -> None:
        self._con.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,))
        self._con.commit()

    # ----------------------------------------------------------------- internals
    def _recall_fts5(self, user_id: str, query: str, k: int) -> list[Recalled]:
        # FTS5 MATCH defaults to AND-of-terms; for natural-language recall
        # queries we want OR semantics with bm25 ranking, so build the query
        # explicitly from tokens.
        tokens = list(_tokenize(query))
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        try:
            rows = self._con.execute(
                """
                SELECT episodes.id, episodes.text, episodes.metadata,
                       bm25(episodes_fts) AS bm25_score
                FROM episodes
                JOIN episodes_fts ON episodes.rowid = episodes_fts.rowid
                WHERE episodes.user_id = ? AND episodes_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (user_id, fts_query, k),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 query parse error (special chars, reserved words, etc.).
            # Fall back to keyword overlap so contracts don't fail on quirky text.
            return self._recall_fallback(user_id, query, k)
        return [
            Recalled(
                text=row[1],
                episode_id=row[0],
                score=-row[3],  # bm25 lower is better; negate so higher = better
                metadata=json.loads(row[2]) if row[2] else None,
            )
            for row in rows
        ]

    def _recall_fallback(self, user_id: str, query: str, k: int) -> list[Recalled]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        rows = self._con.execute(
            """
            SELECT id, text, metadata FROM episodes WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        scored: list[tuple[float, str, str, str | None]] = []
        for eid, text, metadata in rows:
            doc_tokens = _tokenize(text)
            overlap = q_tokens & doc_tokens
            if not overlap:
                continue
            score = len(overlap) / len(q_tokens)
            scored.append((score, eid, text, metadata))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Recalled(
                text=text,
                episode_id=eid,
                score=score,
                metadata=json.loads(meta) if meta else None,
            )
            for score, eid, text, meta in scored[:k]
        ]

    @staticmethod
    def _row_to_episode(row: tuple[str, str, str, str, str | None]) -> Episode:
        eid, user_id, text, created_at_iso, metadata_json = row
        return Episode(
            id=eid,
            user_id=user_id,
            text=text,
            created_at=datetime.fromisoformat(created_at_iso),
            metadata=json.loads(metadata_json) if metadata_json else None,
        )
