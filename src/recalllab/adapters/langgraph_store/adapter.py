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

import threading
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
        # Process-local lock that serialises the get-then-put critical
        # section of custom-id ``remember`` calls. ``BaseStore`` does not
        # expose a compare-and-set primitive: without this lock two
        # concurrent ``remember(..., episode_id=X)`` calls could both see
        # no existing item and both call ``put``, with the second
        # silently overwriting the first (last-writer-wins) — defeating
        # the "different (text, metadata) raises" guarantee. This is
        # process-local only; cross-process / cross-host shared stores
        # need application-level coordination on the writer side.
        self._write_lock = threading.Lock()
        # Conservative defaults: scores/candidate-trace require an indexed
        # store; users with a configured embedding index can opt in.
        self._capabilities = CapabilityFlags(
            supports_forget=True,
            supports_tenant_delete=True,
            supports_provenance=True,
            supports_scores=supports_scores,
            supports_candidate_trace=supports_candidate_trace,
            supports_cost_trace=False,
            # ``BaseStore.put(namespace, key, value)`` writes at the requested
            # key authoritatively, so RecallLab's mutation idempotency works
            # against any store that conforms to the protocol.
            supports_custom_episode_ids=True,
            # ``BaseStore.search`` with no query enumerates the namespace
            # bounded by ``scan_limit``; the adapter raises when that bound
            # is hit so the listing is either authoritative or fails loudly.
            supports_authoritative_list=True,
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
        # When the caller supplies an explicit episode_id, do a read-before-
        # write so retries are honest idempotent replays rather than silent
        # overwrites. ``BaseStore.put`` always overwrites at the requested
        # key — without this guard a retry would refresh ``created_at`` and
        # could silently replace a previously-stored value if the deterministic
        # key collides with a different text/metadata pair across versions.
        #
        # The get+put pair runs under ``self._write_lock`` so concurrent
        # in-process writers cannot race past the existence check and
        # produce a last-writer-wins overwrite (BaseStore has no CAS
        # primitive). For multi-process / multi-host shared stores the
        # caller is responsible for application-level coordination —
        # noted on the lock attribute.
        if episode_id is not None:
            with self._write_lock:
                existing = self._store.get((user_id,), episode_id)
                if existing is not None:
                    return self._idempotent_or_collide(
                        user_id=user_id,
                        episode_id=episode_id,
                        text=text,
                        metadata=metadata,
                        existing_value=existing.value,
                    )
                return self._do_put(
                    user_id=user_id,
                    eid=episode_id,
                    text=text,
                    metadata=metadata,
                )
        return self._do_put(
            user_id=user_id,
            eid=uuid.uuid4().hex,
            text=text,
            metadata=metadata,
        )

    def _do_put(
        self,
        *,
        user_id: str,
        eid: str,
        text: str,
        metadata: dict[str, Any] | None,
    ) -> Episode:
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

    def _idempotent_or_collide(
        self,
        *,
        user_id: str,
        episode_id: str,
        text: str,
        metadata: dict[str, Any] | None,
        existing_value: Any,
    ) -> Episode:
        """Return the existing episode if it matches; raise on collision.

        Mirrors ``ReferenceMemoryAdapter`` semantics so mutation retry
        idempotency is a real guarantee across both v0.1 adapters that
        declare ``supports_custom_episode_ids=True``. Comparison is
        text + metadata (order-insensitive dict equality); the original
        ``created_at`` is preserved.
        """
        if not isinstance(existing_value, dict):
            # Defensive: BaseStore should always give us a dict, but if the
            # store has been bypassed by raw writes we refuse to claim
            # idempotent replay over an unknown shape.
            raise ValueError(
                f"episode_id {episode_id!r} already exists in LangGraph store "
                f"with non-dict value; refusing to silently overwrite"
            )
        existing_text = existing_value.get("text")
        existing_metadata = existing_value.get("metadata")
        if existing_text == text and existing_metadata == metadata:
            existing_created_at_iso = existing_value.get("created_at")
            created_at = (
                datetime.fromisoformat(existing_created_at_iso)
                if isinstance(existing_created_at_iso, str)
                else datetime.now(tz=UTC)
            )
            return Episode(
                id=episode_id,
                user_id=user_id,
                text=text,
                created_at=created_at,
                metadata=metadata,
            )
        raise ValueError(
            f"episode_id {episode_id!r} already exists in LangGraph store "
            f"with different (text, metadata); refusing to silently overwrite. "
            f"Pass a unique id or call forget(episode_id=...) first."
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
        # ``list_episodes`` is now part of the liveness oracle for
        # ``with_stale_repeats`` (gated on ``supports_authoritative_list``).
        # To make that capability honest, refuse to return a silently
        # truncated subset: when the scan saturates the configured limit,
        # the result might be missing a live source episode and the
        # mutation pipeline could mis-diagnose a real episode as deleted.
        # Raise loudly so users either raise ``scan_limit`` or accept a
        # documented failure rather than getting incorrect resurrection
        # decisions. Matches the existing ``forget(matching=...)`` and
        # ``delete_user`` semantics on this same adapter.
        items = self._store.search((user_id,), query=None, limit=self._scan_limit)
        if len(items) >= self._scan_limit:
            raise RuntimeError(
                f"list_episodes hit scan_limit={self._scan_limit}; the "
                f"returned subset is bounded and cannot be treated as an "
                f"authoritative listing. Increase scan_limit on the adapter "
                f"(``LangGraphStoreAdapter(store, scan_limit=...)``) before "
                f"using ``with_stale_repeats`` against namespaces that may "
                f"exceed the bound."
            )
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
