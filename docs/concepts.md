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
  precisely. Land in v0.2.x; gated on a configured ``[judge]`` section in
  ``recalllab.toml``.

## Mutations *(v0.2.0)*

Mutations are deterministic, seeded transformations applied to the active
user's namespace *before* the next recall. They sharpen contracts by
adding adversarial pressure without requiring an LLM judge. Every
mutation emits a ``MUTATION`` trace event capturing what was injected
(and which episode IDs were created), so the Failure Gallery can show
exactly what the contract was up against when it failed.

- ``with_distractors(n, *, seed=0)`` — sample ``n`` distractor texts
  from a fixed pool and remember them under the active user. The sample
  is reproducible: the same ``(n, seed)`` pair always produces the same
  list in the same order. **Default ``seed=0``** so contracts are
  replayable without callers having to supply a seed; pass an explicit
  non-zero seed to vary the sample.
- ``with_stale_repeats(*, times)`` — re-store the most recent ``remember``
  for the active user ``times`` more times. Resolves "most recent" by
  walking back through the trace events for the *active* user, so it
  can be used safely across multiple users in one contract. Before
  repeating, two layers of resurrection guard fire so the mutation
  never re-writes deleted content under a mutation id:
  - **Trace-based (always on, every provider):** walk FORGET events
    after the source REMEMBER. If any FORGET cited the source by id,
    or used a ``matching=`` substring contained in the source text,
    raise — the source is considered dead. The walk starts after the
    most recent REMEMBER, so a ``forget → fresh remember`` sequence is
    not false-positived.
  - **Provider-side (gated on ``supports_authoritative_list``):**
    additionally check ``provider.list_episodes(user_id)`` for
    defense in depth against external state. The configurable MCP
    adapter defaults this capability to ``False`` because its
    ``list_episodes`` is a best-effort wildcard ``recall``; flip
    ``MCPMemoryConfig.list_episodes_is_authoritative = True`` only
    after verifying the upstream server enumerates exhaustively.

**Episode IDs are deterministic.** Both mutations request episode IDs
of the form ``mut-{type}-{sha256[:16]}-{index:04d}`` derived from
``contract_id`` + ``user_id`` + invocation counter + mutation type + key
(seed for distractors, content hash for stale repeats) + index. The
``user_id`` and invocation counter prevent collisions when:

- the same mutation is invoked under two users in one contract (same
  seed → distinct IDs because the user_id differs); or
- the same mutation is invoked twice under one user (same args →
  distinct IDs because the invocation counter differs).

**``seed`` must be a plain ``int``.** ``with_distractors(seed=...)``
calls ``validate_seed`` before anything else and raises ``TypeError``
for ``None``, strings, or booleans. Without this guard,
``random.Random(None)`` would silently re-seed from system entropy
while the deterministic episode IDs derived from ``str(seed)`` stayed
fixed at the string ``"None"`` — guaranteeing same-id / different-text
collisions on the next replay. Pass an explicit non-zero integer to
vary the sample.

**Mutation retry idempotency is gated on a provider capability.**
``CapabilityFlags.supports_custom_episode_ids`` declares that the
adapter writes at the ``episode_id`` RecallLab requests. Both mutations
check this flag before any provider call and refuse to run when it is
``False``, recording a ``MUTATION`` event with ``status="unsupported"``
so the trace explains why nothing was written. The pipeline also
verifies the returned ``episode.id`` matches the requested ID after
every write; a provider that declares the capability but rewrites the
ID server-side raises ``RuntimeError`` mid-loop with
``status="partial_failed"`` so the contract fails loudly rather than
trusting a lying server. Capability matrix:

- ``ReferenceMemoryAdapter`` — ``True`` (SQLite primary key holds).
- ``LangGraphStoreAdapter`` — ``True``: ``BaseStore.put(namespace, key,
  value)`` writes at the requested key authoritatively.
- ``MCPMemoryAdapter`` — ``False`` by default. Flip
  ``MCPMemoryConfig.honors_custom_episode_ids = True`` only after
  verifying the upstream server actually addresses subsequent writes
  by the supplied ID.

**Custom-id replay semantics, per adapter.**

- ``ReferenceMemoryAdapter``: ``remember(..., episode_id=X)`` returns
  the existing episode unchanged when a row with the same id + user +
  text + metadata already exists. Metadata comparison is canonicalised
  via ``json.dumps(..., sort_keys=True)`` so dict-literal ordering is
  irrelevant. Any real collision — same id but different user,
  different text, *or* different metadata — raises ``ValueError``
  rather than silently overwriting.
- ``LangGraphStoreAdapter``: same semantics, via a read-before-write
  on ``BaseStore.get((user_id,), episode_id)``. If the key exists with
  matching text + metadata the original ``created_at`` is preserved
  (a retry is a real no-op, not a refresh); otherwise the write
  raises rather than letting ``BaseStore.put`` overwrite silently.
- ``MCPMemoryAdapter``: when the caller supplies ``episode_id``, the
  upstream tool's response MUST contain a string at
  ``remember_episode_id_field``. If it doesn't, ``remember`` raises
  ``ValueError`` rather than fabricating a successful id echo.
  Whatever id the server returns lands in ``inserted_episode_ids`` so
  the trace stays honest, and the mutation pipeline's mismatch check
  raises if it differs from the requested id. ``episode_id=None``
  writes still fall back to a locally generated UUID so non-mutation
  usage is unaffected.

**Mid-mutation failures are traced, then re-raised.** When
``provider.remember(...)`` raises mid-loop, the ``MUTATION`` event
records ``status="partial_failed"``, the partial
``inserted_episode_ids`` so far, and the exception repr in ``error``.
The original exception is re-raised so pytest fails loudly. There is no
silent rollback in v0.2.0; the trace explains the orphaned writes
instead of hiding them.

Two failure shapes are distinguished:

- **Confirmed-then-failed.** The provider returned an ``Episode`` for
  some iterations before raising later. Those ids appear in
  ``inserted_episode_ids``.
- **Unconfirmed remote write.** The adapter raised
  ``UnconfirmedRemoteWriteError`` — the upstream provider was already
  called but the response was unusable (e.g. an MCP server that
  silently writes the row but returns no episode id field). The
  requested id appears in the new ``unconfirmed_writes`` list with
  semantics "may have landed remotely; check your provider state."
  Without this distinction the trace would lie about hosted-provider
  partial failures by reporting nothing for the failing iteration.

**Same-contract retry resumes at the same invocation.** A partial-failed
mutation leaves its fingerprint ``(mutation_type, user_id, key)`` in
an in-flight map on the ``MemoryContract``. Re-invoking the same
mutation (same user, same seed/source-text *and same count*) reuses
the original invocation number, so the deterministic episode IDs the
retry requests match the orphan rows from the failed attempt. The key
encodes both the seed/text-hash and the requested count (``n`` for
distractors, ``times`` for stale repeats); stale_repeats additionally
includes an *ordinal* over prior REMEMBER events of the same
``(user, text)`` — so a later call with a different ``n``/``times``,
or against a fresh remember of the same text, gets a distinct
fingerprint and a fresh invocation rather than silently piggybacking
on the failed slot. The ordinal is preferred over the source event's
raw trace ``sequence`` because the sequence shifts whenever any
earlier event is added to the contract, which would have broken
deterministic IDs under ordinary refactors. Idempotent adapters
(reference, langgraph) return existing episodes for the writes that
already landed; the retry resumes from the failure point rather than
double-writing. Successful completion clears the fingerprint, so two
*deliberate* successful calls of the same mutation under one user still
get distinct invocations — the round-2 "no-collide-on-repeat" property
is preserved.

Both mutations are scoped to the active user. They cannot leak into
other tenants because every write goes through
``provider.remember(user_id, ...)``. The remaining mutations from the
v0.1 plan (``with_paraphrases``, ``with_tenant_swap``,
``with_delete_reinsert``) land later in the v0.2 line once the
trace-to-test and judge stories are in place.

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
