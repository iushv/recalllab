"""Provider-agnostic interfaces every memory adapter implements.

The MemoryProvider protocol is intentionally tiny. Capability flags let
contracts skip cleanly when a provider can't satisfy an assertion (e.g.
provenance against a Store that doesn't expose source episode IDs).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class CapabilityFlags(BaseModel):
    """What a memory adapter can answer for.

    Drives both contract skips ("provenance contracts skipped on this provider")
    and report N/A cells in the dashboard.
    """

    model_config = ConfigDict(frozen=True)

    supports_forget: bool = False
    supports_tenant_delete: bool = False
    supports_provenance: bool = False
    supports_scores: bool = False
    supports_candidate_trace: bool = False
    supports_cost_trace: bool = False


class Episode(BaseModel):
    """One stored memory event."""

    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    text: str
    created_at: datetime
    metadata: dict[str, Any] | None = None


class Recalled(BaseModel):
    """One memory returned from a recall query.

    `episode_id` and `score` are populated only when the adapter declares
    `supports_provenance` / `supports_scores` respectively.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    episode_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None


@runtime_checkable
class MemoryProvider(Protocol):
    """The minimal contract every adapter implements.

    All operations are scoped per `user_id`. Adapters that don't support a
    capability should still implement the method but may raise
    `NotImplementedError`; capability flags tell contracts not to call into
    unsupported paths in the first place.
    """

    def capabilities(self) -> CapabilityFlags: ...

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode: ...

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        k: int = 5,
    ) -> list[Recalled]: ...

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int: ...

    def list_episodes(self, user_id: str) -> list[Episode]: ...

    def delete_user(self, user_id: str) -> None: ...
