"""Pydantic config for the configurable MCP memory adapter.

Every field here either names a tool on the upstream MCP server or the
argument/result-field key the adapter uses to translate between the
``MemoryProvider`` protocol and that server. Capability flags are derived
from which optional tools and result fields are present:

- ``forget_tool is None`` → ``supports_forget=False``
- ``recall_episode_id_field is None`` → ``supports_provenance=False``
- ``recall_score_field is None`` → ``supports_scores=False``
- ``honors_custom_episode_ids=False`` (the default) →
  ``supports_custom_episode_ids=False``. Mutations (``with_distractors``,
  ``with_stale_repeats``) refuse to run because retry idempotency cannot
  be guaranteed: many MCP memory servers either ignore the ``episode_id``
  argument or rewrite it server-side, which would silently turn a retried
  mutation into a fresh bulk-insert. Flip this to ``True`` only after
  verifying against your specific server that writes are addressable by
  the ID RecallLab requests. The adapter still verifies at runtime that
  the returned ID matches the requested ID, so a lie raises rather than
  corrupting the trace.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MCPMemoryConfig(BaseModel):
    """Explicit tool and argument mapping for any MCP memory server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Connection — for production usage. Tests pass a ``transport=`` override
    # (typically an in-process FastMCP server) directly to the adapter.
    server_url: str

    # Tool names on the upstream server.
    remember_tool: str
    recall_tool: str
    forget_tool: str | None = None

    # Argument names the adapter uses when calling each tool.
    user_arg: str = "user_id"
    text_arg: str = "text"
    query_arg: str = "query"
    k_arg: str = "limit"
    matching_arg: str = "matching"
    episode_id_arg: str = "episode_id"

    # Response field paths within each tool's structured output.
    remember_episode_id_field: str = "episode_id"
    recall_results_field: str = "results"
    recall_text_field: str = "text"
    recall_episode_id_field: str | None = "episode_id"
    recall_score_field: str | None = None
    forget_deleted_field: str = "deleted"

    # Whether the upstream tool honours the ``episode_id`` argument
    # authoritatively (i.e. the server writes at the requested ID and
    # subsequent retries with the same ID address the same record). Default
    # ``False`` — most hosted MCP memory tools assign their own IDs.
    honors_custom_episode_ids: bool = False

    # Whether the adapter's ``list_episodes`` (a wildcard ``recall``)
    # actually enumerates every episode for the user. Default ``False``
    # because many MCP memory servers reject empty queries or cap the
    # result set, which would make ``list_episodes`` a best-effort
    # subset rather than a real listing. Flip to ``True`` only after
    # verifying against your specific server. ``with_stale_repeats``
    # uses this flag to decide whether to run a provider-side liveness
    # check on the source remember (it always does a trace-based check,
    # which catches in-contract ``forget`` calls regardless).
    list_episodes_is_authoritative: bool = False

    # Optional connect timeout, in seconds.
    timeout: float | None = None
