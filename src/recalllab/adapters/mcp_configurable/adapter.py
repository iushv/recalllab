"""Configurable MCP memory adapter.

Wraps any MCP memory server (Mem0 hosted, Graphiti's MCP, custom in-house)
behind the ``MemoryProvider`` protocol. Tool names, argument names, and
response field paths are all driven by ``MCPMemoryConfig`` — no per-server
custom code required.

The adapter is async-internal (FastMCP's client is async) but exposes the
sync ``MemoryProvider`` interface; sync↔async bridging is done with a
short-lived thread per call. That avoids the "cannot run asyncio.run from
a running loop" pitfall when contracts execute under pytest-asyncio, and
the latency overhead is negligible for testing workloads. Production
high-throughput use can wait for v0.2.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from fastmcp import Client

from recalllab.adapters.base import CapabilityFlags, Episode, Recalled
from recalllab.adapters.mcp_configurable.config import MCPMemoryConfig


class MCPMemoryAdapter:
    """Configurable MCP-backed memory provider."""

    def __init__(
        self,
        config: MCPMemoryConfig,
        *,
        transport: Any | None = None,
    ) -> None:
        self._config = config
        # Tests pass an in-process FastMCP server here; production uses the
        # default (config.server_url string).
        self._transport: Any = (
            transport if transport is not None else config.server_url
        )
        self._capabilities = CapabilityFlags(
            supports_forget=config.forget_tool is not None,
            supports_tenant_delete=False,
            supports_provenance=config.recall_episode_id_field is not None,
            supports_scores=config.recall_score_field is not None,
            supports_candidate_trace=False,
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
        args: dict[str, Any] = {
            self._config.user_arg: user_id,
            self._config.text_arg: text,
        }
        if episode_id is not None:
            args[self._config.episode_id_arg] = episode_id
        result = self._call_tool(self._config.remember_tool, args)
        eid = (
            self._extract(result, self._config.remember_episode_id_field)
            if isinstance(result, dict)
            else None
        )
        if not isinstance(eid, str):
            eid = episode_id or uuid.uuid4().hex
        return Episode(
            id=eid,
            user_id=user_id,
            text=text,
            created_at=datetime.now(tz=UTC),
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
        args: dict[str, Any] = {
            self._config.user_arg: user_id,
            self._config.query_arg: query,
            self._config.k_arg: k,
        }
        result = self._call_tool(self._config.recall_tool, args)
        if not isinstance(result, dict):
            return []
        items = self._extract(result, self._config.recall_results_field)
        if not isinstance(items, list):
            return []
        return [self._to_recalled(item) for item in items if isinstance(item, dict)]

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        if self._config.forget_tool is None:
            raise NotImplementedError(
                "this MCP server does not declare a forget tool; "
                "supports_forget is False"
            )
        if matching is None and episode_id is None:
            raise ValueError("forget requires either matching= or episode_id=")
        args: dict[str, Any] = {self._config.user_arg: user_id}
        if matching is not None:
            args[self._config.matching_arg] = matching
        if episode_id is not None:
            args[self._config.episode_id_arg] = episode_id
        result = self._call_tool(self._config.forget_tool, args)
        if not isinstance(result, dict):
            return 0
        deleted = self._extract(result, self._config.forget_deleted_field)
        return int(deleted) if isinstance(deleted, int | float) else 0

    def list_episodes(self, user_id: str) -> list[Episode]:
        # Best-effort via a wildcard recall — most MCP memory servers don't
        # expose a dedicated list-episodes tool. Returns [] if the server
        # rejects an empty query.
        try:
            recalled = self.recall(user_id, "*", k=10_000)
        except Exception:
            return []
        return [
            Episode(
                id=r.episode_id or uuid.uuid4().hex,
                user_id=user_id,
                text=r.text,
                created_at=datetime.now(tz=UTC),
                metadata=r.metadata,
            )
            for r in recalled
        ]

    def delete_user(self, user_id: str) -> None:
        raise NotImplementedError(
            "delete_user requires a tenant-delete tool (not in v0.1)"
        )

    # ---------------------------------------------------------------- internals
    def _call_tool(self, tool: str, arguments: dict[str, Any]) -> Any:
        async def _run() -> Any:
            client_kwargs: dict[str, Any] = {}
            if self._config.timeout is not None:
                client_kwargs["timeout"] = self._config.timeout
            async with Client(self._transport, **client_kwargs) as client:
                result = await client.call_tool(tool, arguments)
            return self._coerce_result(result)

        # Run on a fresh thread so we never collide with a running event
        # loop in the calling context (pytest-asyncio, ipykernel, etc.).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(asyncio.run, _run())
            return future.result()

    @staticmethod
    def _coerce_result(result: Any) -> Any:
        # FastMCP returns CallToolResult; .data carries the structured Python
        # value when the tool declares structured output. Fall back to the
        # raw object so users with non-structured tools still get something.
        data = getattr(result, "data", None)
        if data is not None:
            return data
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        return result

    @staticmethod
    def _extract(payload: Any, path: str | None) -> Any:
        """Walk a dotted path through nested dicts.

        ``"results"`` returns ``payload["results"]``; ``"data.results"``
        returns ``payload["data"]["results"]``. Returns ``None`` if any
        segment is missing or hits a non-dict.
        """
        if path is None:
            return None
        current: Any = payload
        for segment in path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
        return current

    def _to_recalled(self, item: dict[str, Any]) -> Recalled:
        text_val = self._extract(item, self._config.recall_text_field)
        text = text_val if isinstance(text_val, str) else ""
        episode_id: str | None = None
        if self._config.recall_episode_id_field is not None:
            raw = self._extract(item, self._config.recall_episode_id_field)
            if isinstance(raw, str):
                episode_id = raw
        score: float | None = None
        if self._config.recall_score_field is not None:
            raw_score = self._extract(item, self._config.recall_score_field)
            if isinstance(raw_score, int | float):
                score = float(raw_score)
        metadata_raw = item.get("metadata")
        metadata = (
            cast(dict[str, Any], metadata_raw)
            if isinstance(metadata_raw, dict)
            else None
        )
        return Recalled(
            text=text,
            episode_id=episode_id,
            score=score,
            metadata=metadata,
        )
