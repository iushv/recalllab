# Concepts

RecallLab tests *your* memory expectations, not someone else's leaderboard.
This page covers the building blocks: failure modes, contracts, providers,
capability flags, and traces.

## The six failure modes

A contract test in RecallLab usually targets one of six categories. Each
maps to a real production problem with agent memory:

| # | Category | What goes wrong |
|---|---|---|
| 1 | Cross-session recall | A fact stated early in the conversation is no longer retrievable later. |
| 2 | Temporal updates | The memory system returns the stale value after the user updated it. |
| 3 | Contradiction resolution | Two contradictory facts are both surfaced, with no resolution. |
| 4 | Provenance | A recalled fact has no traceable source episode ID. |
| 5 | Privacy / tenant isolation | User B sees a fact stored for User A. |
| 6 | Forget compliance | After ``forget(...)``, the memory still appears in recalls. |

The example contracts under ``examples/tests/`` demonstrate one test per
category. ``recalllab init`` scaffolds them into your repo so you can copy,
adapt, and add your own on top.

## Contracts

A *contract* is a pytest test that uses the ``memory_contract`` fixture.
The fixture is a small DSL with five primary verbs (``given_user``,
``remember``, ``recall``, ``forget``, ``should_recall``) plus a few
mutation hooks. Every method records a ``TraceEvent`` so failures are
debuggable from the persisted trace.

Assertions are explicit about what they're testing:

- ``contains`` — a literal substring (or any of a list) must appear in the
  recall response. Vacuously false when nothing is recalled.
- ``excludes`` — none of the listed values may appear. Vacuously true when
  nothing is recalled, so contracts using ``excludes`` against an empty
  recall should add a precondition that proves the data is normally
  retrievable (see the forget / tenant-isolation examples).
- ``latest_fact_is`` / ``must_not_answer_as`` / ``judge_assertion`` —
  judge-driven modes for the cases ``contains``/``excludes`` can't capture
  precisely. Land in v0.2; gated on a configured ``[judge]`` section in
  ``recalllab.toml``.

## Providers

A *provider* is anything that implements the tiny ``MemoryProvider``
protocol: ``remember``, ``recall``, ``forget``, ``list_episodes``,
``delete_user``, ``capabilities``. The v0.1 set:

- **reference** — in-process SQLite (FTS5 with keyword-overlap fallback).
  Declares ``supports_forget``, ``supports_tenant_delete``,
  ``supports_provenance``, ``supports_scores``, and
  ``supports_candidate_trace``; ``supports_cost_trace`` is ``False`` because
  no token usage is incurred. Runs the six examples green out of the box.
- **langgraph_store** — wraps ``langgraph.store.BaseStore`` (cross-thread
  long-term memory; *not* a checkpointer, which is thread-scoped graph
  state). Set ``[provider]\ntype = "langgraph_store"`` in
  ``recalllab.toml`` to drive the same six contracts against this provider.
  The default ``InMemoryStore`` does not perform similarity search, so
  ``score`` flows through as ``None``; configure an embedding index on the
  store and pass ``supports_scores=True`` to opt in.
- **mcp_configurable** — generic MCP memory adapter with an explicit
  tool-name mapping (``MCPMemoryConfig``); no per-server custom code
  required. Set ``[provider]\ntype = "mcp"`` plus a ``[provider.mcp]``
  table naming ``server_url``, ``remember_tool``, ``recall_tool``, and
  optionally ``forget_tool``. Capability flags derive from which optional
  tools and result fields you configure: omitting ``forget_tool`` flips
  ``supports_forget`` to ``False``, omitting ``recall_episode_id_field``
  flips ``supports_provenance`` off, and so on.

## Capability flags

Different providers expose different surface area, so RecallLab refuses to
hide that fact behind misleading zeros. Every adapter declares its
``CapabilityFlags`` (``supports_forget``, ``supports_provenance``,
``supports_scores``, …). Contracts can opt in to capability gating with a
marker:

```python
import pytest

@pytest.mark.recalllab_optional("supports_provenance")
def test_recalled_fact_cites_source_episode(memory_contract):
    ...
```

When the configured provider lacks the capability, the contract is skipped
and the skip is **persisted to the trace store** with a ``CapabilitySkip``
entry — so the future Failure Gallery can render an explicit "N/A —
capability not supported" cell rather than dropping the row.

## Traces

Every contract run produces a ``ContractRun`` record stored as JSON in
``.recalllab/traces.sqlite``. The record includes:

- The full ordered ``TraceEvent`` log (``given_user`` → ``remember`` →
  ``recall`` → ``assert`` → ...), with per-event latency.
- The list of ``AssertionResult``s.
- Any ``CapabilitySkip`` entries.
- Pytest's authoritative pass/fail/skip outcome, captured via a hookwrapper
  so raw ``assert`` failures and provider exceptions are recorded faithfully.

The Failure Gallery reads from this store. The store is **plaintext** —
every recall query, every ``remember(...)`` text, and every recall result
is JSON-encoded into the SQLite database. Treat ``.recalllab/`` as a debug
log: don't seed real secrets or PII into your contracts. ``TraceStore``
writes a ``.recalllab/.gitignore`` on first use so the trace files stay
out of version control automatically.
