"""Integration test: configurable MCP adapter against an in-process FastMCP server.

Builds a tiny FastMCP server with a dict-backed memory store, points the
``MCPMemoryAdapter`` at it via the ``transport=`` constructor kwarg, then
drives the six v0.1 failure modes through both the raw provider API and
the contract DSL. This is the primary proof that the configurable mapping
(``MCPMemoryConfig``) actually plumbs every operation correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from fastmcp import FastMCP

from recalllab.adapters.base import MemoryProvider
from recalllab.adapters.mcp_configurable import MCPMemoryAdapter, MCPMemoryConfig
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.traces.schema import ContractRun


@pytest.fixture
def mcp_server() -> FastMCP:
    server: FastMCP = FastMCP("recalllab-test-memory")
    storage: dict[str, list[dict[str, str]]] = {}

    @server.tool
    def memory_add(user_id: str, text: str) -> dict[str, str]:
        episode_id = uuid4().hex
        storage.setdefault(user_id, []).append(
            {"episode_id": episode_id, "text": text}
        )
        return {"episode_id": episode_id}

    @server.tool
    def memory_search(
        user_id: str, query: str, limit: int = 5
    ) -> dict[str, list[dict[str, str]]]:
        # Simple keyword-overlap retrieval — enough to drive the six
        # contracts deterministically without an embedding index.
        episodes = storage.get(user_id, [])
        query_tokens = set(query.lower().split())
        scored: list[tuple[int, dict[str, str]]] = []
        for episode in episodes:
            text_tokens = set(episode["text"].lower().split())
            overlap = len(query_tokens & text_tokens)
            if overlap:
                scored.append((overlap, episode))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {
            "results": [
                {"text": ep["text"], "episode_id": ep["episode_id"]}
                for _, ep in scored[:limit]
            ]
        }

    @server.tool
    def memory_forget(
        user_id: str,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, int]:
        episodes = storage.get(user_id, [])
        before = len(episodes)
        if episode_id is not None:
            storage[user_id] = [
                ep for ep in episodes if ep["episode_id"] != episode_id
            ]
        elif matching is not None:
            needle = matching.lower()
            storage[user_id] = [
                ep for ep in episodes if needle not in ep["text"].lower()
            ]
        else:
            return {"deleted": 0}
        return {"deleted": before - len(storage.get(user_id, []))}

    return server


@pytest.fixture
def mcp_adapter(mcp_server: FastMCP) -> MCPMemoryAdapter:
    config = MCPMemoryConfig(
        server_url="in-process",
        remember_tool="memory_add",
        recall_tool="memory_search",
        forget_tool="memory_forget",
    )
    return MCPMemoryAdapter(config, transport=mcp_server)


def test_protocol_compliance(mcp_adapter: MCPMemoryAdapter) -> None:
    assert isinstance(mcp_adapter, MemoryProvider)


def test_capabilities_derive_from_config(mcp_adapter: MCPMemoryAdapter) -> None:
    caps = mcp_adapter.capabilities()
    assert caps.supports_forget is True
    assert caps.supports_provenance is True
    assert caps.supports_scores is False
    assert caps.supports_tenant_delete is False
    assert caps.supports_cost_trace is False


def test_remember_returns_episode_id_from_tool(mcp_adapter: MCPMemoryAdapter) -> None:
    ep = mcp_adapter.remember("ayush", "I live in Mumbai.")
    assert ep.id  # populated from the tool's structured output
    assert ep.user_id == "ayush"
    assert ep.text == "I live in Mumbai."


def test_recall_round_trip_with_provenance(
    mcp_adapter: MCPMemoryAdapter,
) -> None:
    seeded = mcp_adapter.remember("ayush", "The launch is on March 21.")
    results = mcp_adapter.recall("ayush", "when is the launch", k=5)
    assert results
    assert any("march 21" in r.text.lower() for r in results)
    assert all(r.episode_id is not None for r in results)
    assert any(r.episode_id == seeded.id for r in results)


def test_tenant_isolation(mcp_adapter: MCPMemoryAdapter) -> None:
    mcp_adapter.remember("alice", "Project codename: Aurora.")
    bob = mcp_adapter.recall("bob", "project codename", k=5)
    assert bob == []


def test_forget_by_matching_removes_expected_episode(
    mcp_adapter: MCPMemoryAdapter,
) -> None:
    mcp_adapter.remember("ayush", "I am allergic to peanuts.")
    pre = mcp_adapter.recall("ayush", "allergic peanuts", k=5)
    assert any("peanut" in r.text.lower() for r in pre)
    deleted = mcp_adapter.forget("ayush", matching="peanuts")
    assert deleted == 1
    post = mcp_adapter.recall("ayush", "allergic peanuts", k=5)
    assert all("peanut" not in r.text.lower() for r in post)


def test_six_failure_modes_via_contract_dsl(
    mcp_adapter: MCPMemoryAdapter,
) -> None:
    """Drive all six v0.1 failure modes through the actual contract DSL."""
    run = ContractRun(
        id=uuid4().hex,
        contract_id="integration::six_modes_via_dsl",
        provider="mcp",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(mcp_adapter, run)

    # 1. cross-session recall
    contract.given_user("u_cross")
    contract.remember("My birthday is December 28.")
    for topic in ("weather", "sports", "movies", "books", "food", "travel"):
        contract.remember(f"We talked about {topic}.")
    contract.should_recall("When is my birthday?", contains="December 28")

    # 2. temporal updates
    contract.given_user("u_temporal")
    contract.remember("I live in Bangalore.")
    contract.remember("Correction: I moved to Mumbai.")
    contract.should_recall("Where do I live now?", contains="Mumbai")

    # 3. contradiction resolution
    contract.given_user("u_contradiction")
    contract.remember("My job title is Junior Engineer.")
    contract.remember(
        "Update on my job title: I was promoted to Senior Engineer."
    )
    contract.should_recall(
        "What is my job title now?", contains="Senior Engineer"
    )

    # 4. provenance — every recall result must carry an episode_id
    contract.given_user("u_prov")
    seeded = contract.remember("The product launch is scheduled for March 21.")
    results = contract.recall("when is the product launch", k=5)
    assert results
    assert all(r.episode_id is not None for r in results)
    assert any(r.episode_id == seeded.id for r in results)

    # 5. tenant isolation
    contract.given_user("alice").remember("Project codename: Aurora.")
    contract.should_recall("project codename", contains="Aurora")
    contract.given_user("bob")
    contract.should_recall("project codename", excludes="Aurora")

    # 6. forget compliance
    contract.given_user("u_forget")
    contract.remember("I am allergic to peanuts.")
    contract.should_recall("What am I allergic to?", contains="peanut")
    deleted = contract.forget(matching="peanuts")
    assert deleted == 1
    contract.should_recall("What am I allergic to?", excludes="peanut")

    # Trace integrity
    assert len(run.events) > 0
    assert all(a.passed for a in run.assertions), [
        a.reason for a in run.assertions if not a.passed
    ]


def test_capabilities_react_to_config_overrides(mcp_server: FastMCP) -> None:
    """Removing optional fields/tools should drop the corresponding capability."""
    config: dict[str, Any] = {
        "server_url": "in-process",
        "remember_tool": "memory_add",
        "recall_tool": "memory_search",
        "forget_tool": None,  # disable forget
        "recall_episode_id_field": None,  # disable provenance
        "recall_score_field": "score",  # claim scores
    }
    adapter = MCPMemoryAdapter(MCPMemoryConfig(**config), transport=mcp_server)
    caps = adapter.capabilities()
    assert caps.supports_forget is False
    assert caps.supports_provenance is False
    assert caps.supports_scores is True
