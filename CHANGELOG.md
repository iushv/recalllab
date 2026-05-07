# Changelog

All notable changes to RecallLab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-18

### Added — contract mutations

- ``MemoryContract.with_distractors(n, *, seed=0)`` — inject ``n``
  deterministic distractor episodes into the active user's namespace
  before the next recall. The distractor pool is fixed; the same
  ``(n, seed)`` pair always produces the same texts in the same order.
  **Default ``seed=0``** so contracts are replayable without callers
  having to supply a seed; pass an explicit non-zero seed to vary the
  sample.
- ``MemoryContract.with_stale_repeats(*, times)`` — duplicate the most
  recent ``remember`` for the active user ``times`` more times. Resolves
  "most recent" by walking back through trace events for the active user,
  so it works across multi-user contracts.
- **Deterministic episode IDs for mutation writes.** Both mutations
  request IDs of the form ``mut-{type}-{sha256[:16]}-{index:04d}``
  derived from ``contract_id`` + ``user_id`` + invocation counter +
  mutation type + key (seed for distractors, source-text content hash
  for stale repeats) + index, and pass them via ``episode_id=`` into
  ``provider.remember(...)``. Including ``user_id`` and an
  invocation counter prevents collisions when two users invoke the same
  mutation with the same seed in one contract, and when the same
  mutation is invoked twice for one user.
- **Reference adapter custom-id idempotency.**
  ``ReferenceMemoryAdapter.remember(..., episode_id=X)`` is now
  idempotent when a row with the same ``id`` + ``user_id`` + ``text``
  already exists — it returns the existing episode without re-inserting,
  so a retried mutation against the same persistent store does not raise
  on the PRIMARY KEY constraint. A real collision (same id, different
  user or different text) raises ``ValueError`` rather than silently
  overwriting. ``LangGraphStoreAdapter`` already inherits idempotency
  from ``BaseStore.put`` (which overwrites). The configurable MCP
  adapter's idempotency depends on the upstream tool — if it rejects
  duplicate IDs or accepts duplicates without dedup, that behaviour
  flows through unchanged.
- **Partial-failure tracing.** When ``provider.remember(...)`` raises
  mid-loop, the ``MUTATION`` event records
  ``status="partial_failed"``, the partial ``inserted_episode_ids`` so
  far, and ``error=repr(exc)``. The original exception is re-raised so
  pytest fails loudly. No silent rollback in v0.2.0; the trace explains
  orphan writes against hosted providers.
- Both mutations emit a ``MUTATION`` ``TraceEvent`` with payload schema:
  ``{type, user_id, status, inserted_episode_ids, ...}`` plus
  mutation-specific fields (``seed``/``requested`` for distractors,
  ``times``/``source_episode_id`` for stale repeats).
- Both are scoped to the active user: distractors do not leak across
  tenants and repeats do not duplicate other users' facts.
- ``with_paraphrases`` is intentionally deferred — without a judge or a
  real paraphrase model it would be word-shuffling, not honest paraphrasing.
- **Strict ``seed`` type validation.** ``validate_seed`` (called from
  ``sample_distractors`` and ``MemoryContract.with_distractors``) raises
  ``TypeError`` for anything other than a plain ``int`` — including
  ``None``, strings, and booleans. Without this, ``random.Random(None)``
  would silently reseed from system entropy while the mutation pipeline
  still derived deterministic episode IDs from ``str(seed)``, producing
  same-id-different-text collisions on retry. Regression-tested for each
  rejected type.
- **Provider capability gate for mutation idempotency.** Added
  ``CapabilityFlags.supports_custom_episode_ids`` (default ``False``).
  ``ReferenceMemoryAdapter`` and ``LangGraphStoreAdapter`` declare it
  ``True``; the configurable MCP adapter exposes
  ``MCPMemoryConfig.honors_custom_episode_ids`` (default ``False``) so
  users explicitly opt in only against MCP servers they have verified to
  write at the requested ID. Both mutation methods refuse to run when the
  flag is ``False``, raise ``RuntimeError``, and emit a ``MUTATION``
  trace event with ``status="unsupported"`` before any provider call —
  no ghost writes against providers that can't guarantee retry
  idempotency.
- **Runtime verification of returned episode IDs.** The mutation pipeline
  now asserts ``episode.id == requested_id`` after every provider
  ``remember`` call. A lying provider (one that declares the capability
  but rewrites the ID server-side) raises ``RuntimeError``, marks the
  ``MUTATION`` event ``status="partial_failed"`` with the partial
  ``inserted_episode_ids`` showing what actually landed, and re-raises so
  pytest fails loudly. Both ``requested_episode_ids`` and
  ``inserted_episode_ids`` are recorded in the trace for forensics.
- **Metadata-aware idempotency in the reference adapter.**
  ``ReferenceMemoryAdapter.remember(..., episode_id=X)`` now compares
  ``metadata`` (canonicalised via ``json.dumps(..., sort_keys=True)``) in
  addition to ``user_id`` and ``text`` before returning the existing row.
  A retry with the same id and text but corrected provenance / tags
  raises ``ValueError`` rather than silently returning the stale stored
  metadata — closing a quiet data-corruption path where idempotent
  replays would mask metadata updates.
- **MCP adapter: strict response-id verification on custom writes.**
  ``MCPMemoryAdapter.remember(..., episode_id=X)`` no longer falls back
  to ``X`` when the upstream tool returns no string at
  ``remember_episode_id_field``. The fallback was silently fabricating a
  successful verification at the mutation pipeline (the contract DSL's
  ``episode.id == requested_id`` check would pass even if the server had
  ignored ``X`` or changed its response schema). The adapter now raises
  ``ValueError`` instead, so retries against a server that doesn't echo
  the requested id fail loudly rather than double-writing hosted memory.
  Non-custom writes (``episode_id=None``) still fall back to a locally
  generated UUID; the strict guard only applies when the caller asserted
  a specific id. Regression-tested with an FastMCP server that drops the
  response field and another that rewrites the id server-side.
- **LangGraph custom-id writes are race-safe in-process.** The
  ``LangGraphStoreAdapter.remember`` get-then-put critical section now
  runs under a process-local ``threading.Lock``. ``BaseStore`` has no
  compare-and-set primitive, so without serialisation two concurrent
  ``remember(..., episode_id=X)`` calls with different text could both
  see no existing item and both call ``put`` — last-writer-wins, which
  would have silently overwritten user data while the capability flag
  claimed collisions are caught. The lock makes the round-7 "different
  text/metadata raises" guarantee real under in-process concurrency
  (parallel mutation retries, threaded test setups). Multi-process /
  multi-host shared stores still need application-level coordination;
  noted on the adapter's ``_write_lock`` attribute. Pinned by two
  regression tests: two threads racing the same id with different text
  (assert exactly one wins, one raises, store has one row not two),
  and three threads racing the same id+text+metadata (assert all return
  the same Episode idempotently with no duplicate rows).
- **Abandoned in-flight mutations are surfaced in the trace.** When a
  mutation partial-fails and the next call under the same user uses a
  different fingerprint (e.g. ``remember(A) → with_stale_repeats
  partial-fail → remember(B) → with_stale_repeats``), the new
  ``MUTATION`` event records the abandoned fingerprint(s) under the
  ``abandoned_in_flight`` payload field. Round-11 Codex finding: the
  second mutation is correctly a fresh call (different source ⇒
  different fingerprint), but without this surface the Failure Gallery
  would show a clean "completed" event while orphan rows from the
  failed first mutation still sat in storage. The new field exposes the
  abandoned mutation_type / key / invocation so the dashboard can
  cross-reference the original ``partial_failed`` event. Pinned by two
  regression tests: normal mutations report ``abandoned_in_flight=[]``
  (consistent payload shape), and the canonical "different-source after
  partial-failure" path surfaces exactly one abandoned fingerprint.
- **Unconfirmed remote writes are traced.** Added typed
  ``UnconfirmedRemoteWriteError`` exception. The MCP adapter raises it
  (instead of plain ``ValueError``) when the upstream ``remember`` tool
  was called but the response lacks the configured episode-id field —
  i.e. the row may have landed remotely but the adapter cannot confirm.
  The mutation pipeline catches this exception, appends the requested
  id to a new ``unconfirmed_writes`` list on the ``MUTATION`` trace
  event payload, then re-raises so pytest fails loudly. Previously the
  trace would have shown an empty ``inserted_episode_ids`` for that
  iteration while the remote store held an orphan row at the
  deterministic id — defeating the partial-failure forensic guarantee
  for exactly the hosted-provider case the round-7 fixes were trying
  to harden. Pinned by two regression tests: an MCP server that
  silently writes but returns no id (asserts ``unconfirmed_writes`` is
  populated), and a successful capability-gated path (asserts the
  field is always present with consistent shape).
- **LangGraph ``list_episodes`` honest at scale.** The adapter declares
  ``supports_authoritative_list=True`` so ``with_stale_repeats`` uses
  ``list_episodes`` as a liveness oracle. To make that capability
  honest, ``list_episodes`` now raises ``RuntimeError`` when the
  underlying scan saturates ``scan_limit`` — matching the existing
  semantics on ``forget(matching=...)`` and ``delete_user``. Without
  this, a live source episode past the bound would be mis-diagnosed as
  deleted and the mutation would refuse to run against a perfectly
  valid namespace. Pinned by two regression tests: direct
  ``list_episodes`` saturation and end-to-end propagation through
  ``with_stale_repeats``.
- **``with_stale_repeats`` resurrection guard.** Two-layered so it
  works on every provider without relying on protocol behaviour nobody
  promised:
  1. **Trace-based detection (always on).** Walk FORGET events after
     the source REMEMBER. If any FORGET cited the source's episode id,
     or used a ``matching=`` substring contained in the source text,
     raise ``RuntimeError("forgotten in this contract")``. The walk
     starts after the *most recent* source remember, so a
     forget-then-recreate sequence does not false-positive.
  2. **Provider-side liveness (gated on capability).** A new
     ``CapabilityFlags.supports_authoritative_list`` declares that
     ``list_episodes(user_id)`` returns every live episode for the
     user, not a best-effort subset. Reference and LangGraph adapters
     declare it ``True``; the configurable MCP adapter exposes
     ``MCPMemoryConfig.list_episodes_is_authoritative`` (default
     ``False``) so the listing isn't treated as a liveness oracle on
     servers whose wildcard recall doesn't enumerate. Authoritative-
     list providers run the extra check for defense in depth against
     provider-side eviction; non-authoritative providers rely on the
     trace-based layer.
  Pinned by seven tests in total: forget-by-matching, forget-by-id,
  fresh-remember-after-forget, MCP-style non-authoritative listing must
  not block normal stale_repeats, trace guard fires on non-authoritative
  providers, forget-then-recreate doesn't false-positive, substring
  match catches partial-text forgets.
- **Tighter partial-failure retry fingerprints.** The in-flight key for
  ``with_distractors`` now includes ``n`` (was ``(type, user, seed)``);
  ``with_stale_repeats`` now includes ``times`` and an ordinal count of
  prior remembers of the same ``(user, text)`` pair (was
  ``(type, user, text_hash)``). Without these additions, a later call
  with the same seed/text but a different count — or a stale_repeats
  call against a fresh remember with the same text — would clear the
  prior partial-failed mutation's in-flight slot, record a completed
  event with the wrong requested count, and leave the original orphan
  rows orphaned from the trace. The discriminator for stale_repeats is
  an *ordinal* over prior matching remembers rather than the source
  event's trace ``sequence`` — the sequence-based form was brittle
  (any unrelated earlier event shifted it, breaking deterministic IDs
  across logically-equivalent reruns of the same contract). Ordinal is
  invariant under unrelated edits, distinct between back-to-back
  same-text remembers, and stable across fresh runs of the same
  contract code. Pinned by five regression tests covering distinct-n
  distractors, distinct-times stale repeats, same-text / distinct-
  source stale repeats, stability under unrelated-prior-event edits,
  and back-to-back same-text calls staying distinct.
- **Partial-failure retry resumes at the same invocation.** The mutation
  pipeline now tracks an in-flight fingerprint
  ``(mutation_type, user_id, key)`` for every mutation that has not yet
  completed successfully. A retry of the same fingerprint reuses the
  original invocation number so the deterministic episode IDs match the
  orphan rows from the failed attempt; idempotent adapters (reference,
  langgraph) return existing episodes for the writes that already
  landed and the retry resumes from the failure point rather than
  double-writing. Successful completion clears the fingerprint, so two
  *deliberate* successful calls of the same mutation under one user
  still get distinct invocations (preserves the round-2
  "no-collide-on-repeat" property). Pinned by four regression tests:
  retry-resume for both ``with_distractors`` and ``with_stale_repeats``,
  deliberate re-call after success allocating fresh ids, and
  fingerprint isolation across users.
- **LangGraph adapter: real read-before-write idempotency on custom ids.**
  ``LangGraphStoreAdapter.remember(..., episode_id=X)`` now does a
  ``BaseStore.get((user_id,), X)`` before any put. If the key exists with
  matching ``text`` and ``metadata`` (dict equality, order-insensitive),
  the existing episode is returned unchanged — preserving the original
  ``created_at`` rather than refreshing it on every retry. If the key
  exists with a different text or metadata, the adapter raises
  ``ValueError`` instead of silently overwriting (which is what
  ``BaseStore.put`` does by default). Previously the adapter declared
  ``supports_custom_episode_ids=True`` but did not deliver idempotent
  replay semantics; this commit makes the capability honest. A new
  ``tests/integration/test_langgraph_adapter.py`` pins the contract with
  seven tests covering same-content replay, key-order-insensitive
  metadata equality, text collisions, metadata collisions, the
  ``None``-vs-``{}`` distinction, and a full mutation-retry round trip.

## [0.1.0] - 2026-05-07

First public release. The wedge: pytest plugin and CI harness for agent
memory, where existing benchmarks (MemoryBench, AMB, MemoryAgentBench) are
*upstream sources* rather than competitors.

### Added

- **pytest11 plugin** with `memory_contract` fixture and `recalllab_optional("supports_X")` marker for capability-gated skips.
- **Contract DSL**: `given_user`, `remember`, `recall`, `forget`, `should_recall(contains=, excludes=)`. Six example contracts ship in `examples/tests/` covering the canonical failure modes (cross-session, temporal updates, contradiction, provenance, tenant isolation, forget compliance).
- **SQLite trace store** at `.recalllab/traces.sqlite` — every `TraceEvent` (`given_user → remember → recall → assert`) persists with per-event latency. `TraceStore` writes a `.recalllab/.gitignore` on first use so traces stay out of version control.
- **Three memory provider adapters:**
  - `reference` — in-process SQLite (FTS5 with keyword-overlap fallback); declares `supports_forget`, `supports_tenant_delete`, `supports_provenance`, `supports_scores`, `supports_candidate_trace`. Runs all six examples green with no API keys, no Postgres, no infra.
  - `langgraph_store` — wraps `langgraph.store.BaseStore` (the cross-thread Store API; *not* `BaseCheckpointSaver`, which is thread-scoped graph state). Exposes `supports_scores=True` only when the underlying store has an embedding index configured.
  - `mcp_configurable` — generic MCP memory adapter with explicit `MCPMemoryConfig` mapping (tool names, argument names, dotted result-field paths). Works with any FastMCP transport — URLs in production, in-process FastMCP servers in tests. Capability flags derive from which optional tools / fields are configured.
- **Failure Gallery dashboard** (FastAPI + htmx, lazy-imported via `[dashboard]` extra). Landing page groups runs by status (failed → skipped → passed → errors); failed cards show contract id, provider, failed assertion reason, and the last recall query. Detail page renders ordered events with latency. Served via `recalllab dashboard --trace --host --port`.
- **CLI** — `recalllab init` scaffolds `tests/memory/` with all six example contracts plus a `recalllab.toml` config; re-runs are idempotent.
- **27 tests** (6 example contracts + 8 MCP integration + 13 regressions for the issues caught in adversarial review). mypy `--strict` clean, ruff clean.

### Security

- `ReferenceMemoryAdapter.forget(matching=...)` filters in Python with literal substring matching, **not** SQL `LIKE` — `'%'` and `'_'` no longer act as wildcards. Locked by parametrised regression tests in `tests/unit/test_codex_review_regressions.py`.
- `LangGraphStoreAdapter.forget(matching=...)` and `delete_user(...)` raise `RuntimeError` when the configured `scan_limit` is hit, refusing to silently report success on an incomplete scan. Episode-id-based forget remains exact regardless of namespace size.
- pytest plugin defers `TraceStore` instantiation to first contract run — installing RecallLab no longer creates `.recalllab/` in unrelated repos.
- `TraceStore` auto-writes `.recalllab/.gitignore` so trace files (which contain raw memory text) stay out of version control.
- Dashboard `--host` help string warns that `0.0.0.0` exposes the trace store on the LAN unauthenticated.

### Deferred to v0.2

- **Trace-to-test generation** — `recalllab record --trace ... --out test_real_failure.py` will turn a real production failure into a checked-in pytest regression.
- **Judge-driven assertion modes** — `latest_fact_is`, `must_not_answer_as`, `judge_assertion`. Gated on a configured `[judge]` section in `recalllab.toml`.
- **Contract mutations** — `with_distractors`, `with_paraphrases`, `with_stale_repeats`, `with_tenant_swap`, `with_delete_reinsert`.
- **Benchmark importers** — LongMemEval, LoCoMo, MemoryAgentBench → contract DSL.
- **YAML form of the DSL.**
- **pgvector / embedding-based reference adapter.**

[unreleased]: https://github.com/iushv/recalllab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/iushv/recalllab/releases/tag/v0.2.0
[0.1.0]: https://github.com/iushv/recalllab/releases/tag/v0.1.0
