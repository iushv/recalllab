"""Pydantic config for the configurable MCP memory adapter.

Every field here either names a tool on the upstream MCP server or the
argument/result-field key the adapter uses to translate between the
``MemoryProvider`` protocol and that server. Capability flags are derived
from which optional tools and result fields are present:

- ``forget_tool is None`` → ``supports_forget=False``
- ``recall_episode_id_field is None`` → ``supports_provenance=False``
- ``recall_score_field is None`` → ``supports_scores=False``
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

    # Optional connect timeout, in seconds.
    timeout: float | None = None
