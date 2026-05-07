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
    supports_custom_episode_ids: bool = False
    # ``list_episodes(user_id)`` returns *every* live episode for the user,
    # not a best-effort subset. Used by mutation guards that need a real
    # "does this id still exist?" check (e.g. ``with_stale_repeats``
    # refusing to resurrect a forgotten source). The MCP adapter defaults
    # this to ``False`` because its ``list_episodes`` is a wildcard
    # ``recall`` and many memory servers reject empty queries or cap the
    # result set; flip it via
    # ``MCPMemoryConfig.list_episodes_is_authoritative`` only after
    # verifying against your specific server. Reference + LangGraph
    # adapters declare it ``True``.
    supports_authoritative_list: bool = False


class UnconfirmedRemoteWriteError(Exception):
    """Raised by an adapter when an upstream provider may have written but
    the adapter cannot confirm.

    Canonical case: the configurable MCP adapter sends a custom
    ``episode_id`` to the remote ``remember`` tool, the server returns a
    response without the expected id field, and we have no way to tell
    whether the row landed remotely. The mutation pipeline catches this
    exception, records the unconfirmed id in the ``MUTATION`` trace
    event's ``unconfirmed_writes`` list, then re-raises so pytest fails
    loudly. The Failure Gallery can then show "the contract crashed at
    this index — your provider may have an orphan row at this id; check
    it."

    Distinct from plain ``ValueError`` (which adapters raise for
    pre-write collisions / detected misconfiguration where no remote
    state changed) because the trace needs to record the requested id —
    silently dropping it would hide hosted-provider partial failures.
    """

    def __init__(
        self,
        requested_episode_id: str,
        raw_response: Any = None,
        message: str = "",
    ) -> None:
        super().__init__(
            message
            or f"provider may have written at episode_id={requested_episode_id!r} "
            f"but did not confirm; treat as a possibly-orphaned remote row"
        )
        self.requested_episode_id = requested_episode_id
        self.raw_response = raw_response


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
