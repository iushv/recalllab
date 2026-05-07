"""Provenance — every recalled fact must cite a source episode.

Marked ``recalllab_optional("supports_provenance")``: providers that don't
expose source-episode IDs (e.g. some MCP backends) have this contract
auto-skipped. The reference adapter declares the capability and runs it.
"""

import pytest


@pytest.mark.recalllab_optional("supports_provenance")
def test_recalled_fact_cites_source_episode(memory_contract):
    memory_contract.given_user("ayush")
    seeded = memory_contract.remember("The product launch is scheduled for March 21.")
    results = memory_contract.recall("when is the product launch", k=5)
    assert results, "expected at least one recalled memory"
    assert all(r.episode_id is not None for r in results), (
        "every recalled memory must include a source episode_id "
        "when supports_provenance is True"
    )
    assert any(r.episode_id == seeded.id for r in results), (
        "the originally-seeded episode_id must be reachable from the recall result"
    )
