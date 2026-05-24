"""Embedded scaffold templates used by ``recalllab init``.

Kept in lock-step with ``examples/tests/`` and ``recalllab.toml`` defaults.
If you change one, change the other too — there is no auto-sync in v0.1.
"""

from __future__ import annotations

RECALLLAB_TOML = """\
# RecallLab configuration. See https://github.com/iushv/recalllab.
#
# Defaults are designed so `recalllab init && pytest tests/memory` runs all
# six example contracts green with no API keys, no Postgres, no infra.
#
# Note on traces: every contract run is persisted to the SQLite trace store
# in plaintext, including the raw text passed to `remember(...)`. Don't seed
# real secrets or PII into your contracts — `.recalllab/` ships with a
# .gitignore so the data stays local, but treat it as you would a debug log.

[provider]
type = "reference"            # or "langgraph_store" or "mcp" once those land

[trace]
path = ".recalllab/traces.sqlite"

[judge]
provider = "none"             # set to "anthropic" to enable judge-based
                              # assertion modes (latest_fact_is,
                              # must_not_answer_as, judge_assertion).
                              # Requires the [judge] extra:
                              #   pip install 'recalllab[judge]'
                              # and ANTHROPIC_API_KEY in the environment.

# [provider.mcp]              # populated only when provider.type = "mcp"
# server_url = "..."
# remember_tool = "memory.add"
# recall_tool = "memory.search"
# forget_tool = "memory.delete"
# user_arg = "user_id"
# query_arg = "query"
"""


_TEST_UPDATED_LOCATION = '''\
"""Temporal updates — the new value must be retrievable after the user updates a fact.

The reference adapter is intentionally minimal (lexical retrieval only); it
will return both the stale and the updated memories, so ``contains`` is the
right assertion mode here. For the stricter "the old value must not be the
*current* answer", use ``must_not_answer_as`` once a judge is configured in
recalllab.toml.
"""


def test_updated_location_overrides_stale_memory(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I live in Bangalore.")
    memory_contract.remember("Correction: I moved to Mumbai.")
    memory_contract.should_recall("Where do I live now?", contains="Mumbai")
'''


_TEST_CROSS_SESSION = '''\
"""Cross-session recall — a fact stated early must remain recallable later."""


def test_birthday_persists_across_many_turns(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("My birthday is December 28.")
    for topic in ("weather", "sports", "movies", "books", "food", "travel"):
        memory_contract.remember(f"We talked about {topic}.")
    memory_contract.should_recall("When is my birthday?", contains="December 28")
'''


_TEST_CONTRADICTION = '''\
"""Contradiction resolution — when a fact is updated, the new value must be retrievable.

Like ``test_updated_location``, this asserts the new value is *present* in
the recalled context. For the stricter "the older value is no longer the
current answer", use ``must_not_answer_as`` once a judge is configured in
recalllab.toml.
"""


def test_promotion_overrides_old_title(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("My job title is Junior Engineer.")
    memory_contract.remember(
        "Update on my job title: I was promoted to Senior Engineer."
    )
    memory_contract.should_recall(
        "What is my job title now?", contains="Senior Engineer"
    )
'''


_TEST_PROVENANCE = '''\
"""Provenance — every recalled fact must cite a source episode.

Marked ``recalllab_optional("supports_provenance")``: providers that do not
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
'''


_TEST_TENANT_ISOLATION = '''\
"""Tenant isolation — User B must not see User A's memories."""


def test_user_b_cannot_see_user_a_memories(memory_contract):
    memory_contract.given_user("alice").remember("Project codename: Aurora.")
    # Precondition: alice can recall her own data, otherwise the cross-tenant
    # excludes assertion below would be vacuously true.
    memory_contract.should_recall(
        "What is the project codename?", contains="Aurora"
    )

    memory_contract.given_user("bob")
    memory_contract.should_recall(
        "What is the project codename?", excludes="Aurora"
    )
'''


_TEST_FORGET_COMPLIANCE = '''\
"""Forget compliance — after ``forget(...)``, the data must not appear in recalls."""


def test_forget_removes_allergy_immediately(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I am allergic to peanuts.")

    # Precondition: prove the fact is retrievable BEFORE we forget it,
    # otherwise the post-forget excludes assertion is vacuously true.
    memory_contract.should_recall("What am I allergic to?", contains="peanut")

    deleted = memory_contract.forget(matching="peanuts")
    assert deleted == 1, f"expected exactly 1 memory removed, got {deleted}"

    memory_contract.should_recall("What am I allergic to?", excludes="peanut")
'''


SCAFFOLD_CONTRACTS: dict[str, str] = {
    "test_updated_location.py": _TEST_UPDATED_LOCATION,
    "test_cross_session.py": _TEST_CROSS_SESSION,
    "test_contradiction.py": _TEST_CONTRADICTION,
    "test_provenance.py": _TEST_PROVENANCE,
    "test_tenant_isolation.py": _TEST_TENANT_ISOLATION,
    "test_forget_compliance.py": _TEST_FORGET_COMPLIANCE,
}
