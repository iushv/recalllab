"""LangGraph Store adapter for RecallLab.

Wraps any ``langgraph.store.base.BaseStore``. Per the LangGraph docs,
``BaseStore`` is the cross-thread long-term memory API; ``BaseCheckpointSaver``
is for thread-scoped graph state and is *not* a memory layer — so the
adapter targets ``BaseStore``.

One namespace per ``user_id`` (the ``(user_id,)`` tuple pattern) so cross-
user isolation is enforced by the store. Episode IDs are uuid4 hex strings
used as the store key, providing provenance for free.

Note on retrieval ranking: the default ``InMemoryStore`` does *not* perform
similarity search. It returns items in insertion order with ``score=None``.
Configure an embedding index on the store if you want real ranking; the
adapter forwards through whatever ``score`` the store provides.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore

from recalllab.adapters.base import CapabilityFlags, Episode, Recalled


class LangGraphStoreAdapter:
    """Adapter that exposes a ``langgraph.store.BaseStore`` as a ``MemoryProvider``."""

    def __init__(
        self,
        store: BaseStore,
        *,
        supports_scores: bool = False,
        supports_candidate_trace: bool = False,
        scan_limit: int = 10_000,
    ) -> None:
        self._store = store
        self._scan_limit = scan_limit
        # Conservative defaults: scores/candidate-trace require an indexed
        # store; users with a configured embedding index can opt in.
        self._capabilities = CapabilityFlags(
            supports_forget=True,
            supports_tenant_delete=True,
            supports_provenance=True,
            supports_scores=supports_scores,
            supports_candidate_trace=supports_candidate_trace,
            supports_cost_trace=False,
        )

    # ------------------------------------------------------------ provider API
    def capabilities(self) -> CapabilityFlags:
        return self._capabilities

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
        value: dict[str, Any] = {
            "text": text,
            "created_at": created_at.isoformat(),
        }
        if metadata is not None:
            value["metadata"] = metadata
        self._store.put((user_id,), eid, value)
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
        items = self._store.search((user_id,), query=query, limit=k)
        return [self._to_recalled(item) for item in items]

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
            existing = self._store.get((user_id,), episode_id)
            if existing is None:
                return 0
            self._store.delete((user_id,), episode_id)
            return 1
        # Match-mode: scan the namespace, delete items whose text contains
        # the match substring. Bounded by ``scan_limit`` — if we reach that
        # bound, refuse silent partial success: forget-compliance contracts
        # must not pass on small fixtures and then fail at scale. Either
        # raise scan_limit on the adapter or use episode-id forget.
        items = self._store.search((user_id,), query=None, limit=self._scan_limit)
        if len(items) >= self._scan_limit:
            raise RuntimeError(
                f"forget(matching=...) hit scan_limit={self._scan_limit}; "
                f"deletion would be incomplete. Increase scan_limit on the "
                f"adapter, or use episode_id-based forget for guaranteed "
                f"semantics."
            )
        deleted = 0
        for item in items:
            text = self._extract_text(item.value)
            if matching is not None and matching.lower() in text.lower():
                self._store.delete((user_id,), item.key)
                deleted += 1
        return deleted

    def list_episodes(self, user_id: str) -> list[Episode]:
        # list_episodes is best-effort and not tied to forget-compliance,
        # so we return what we got without raising — but document the bound.
        items = self._store.search((user_id,), query=None, limit=self._scan_limit)
        return [self._to_episode(user_id, item) for item in items]

    def delete_user(self, user_id: str) -> None:
        items = self._store.search((user_id,), query=None, limit=self._scan_limit)
        if len(items) >= self._scan_limit:
            raise RuntimeError(
                f"delete_user hit scan_limit={self._scan_limit}; deletion "
                f"would be incomplete. Increase scan_limit on the adapter, "
                f"or implement a store-native bulk-delete path."
            )
        for item in items:
            self._store.delete((user_id,), item.key)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _extract_text(value: Any) -> str:
        if isinstance(value, dict):
            text = value.get("text", "")
            return text if isinstance(text, str) else ""
        return ""

    @staticmethod
    def _extract_metadata(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            metadata = value.get("metadata")
            if isinstance(metadata, dict):
                return metadata
        return None

    @classmethod
    def _to_recalled(cls, item: Any) -> Recalled:
        return Recalled(
            text=cls._extract_text(item.value),
            episode_id=item.key,
            score=getattr(item, "score", None),
            metadata=cls._extract_metadata(item.value),
        )

    @classmethod
    def _to_episode(cls, user_id: str, item: Any) -> Episode:
        created_at: datetime
        raw_created = (
            item.value.get("created_at") if isinstance(item.value, dict) else None
        )
        if isinstance(raw_created, str):
            try:
                created_at = datetime.fromisoformat(raw_created)
            except ValueError:
                created_at = datetime.now(tz=UTC)
        else:
            created_at = datetime.now(tz=UTC)
        return Episode(
            id=item.key,
            user_id=user_id,
            text=cls._extract_text(item.value),
            created_at=created_at,
            metadata=cls._extract_metadata(item.value),
        )
