# Changelog

All notable changes to RecallLab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2026-06-05

Packaging and CI maintenance only — no functional changes to the
library or its public API.

### Changed

- **License metadata** migrated to the PEP 621 table form
  (`license = { text = "Apache-2.0" }` → SPDX expression), clearing the
  setuptools deprecation warning raised by modern build backends.

### CI

- **GitHub Actions pinned** to Node 24-compatible release versions
  (`checkout@v4.2.2`, `setup-uv@v5`, `upload-artifact@v4.6.2`,
  `download-artifact@v4.3.0`) so the build/publish jobs stay green on
  the current runner images.
- **Release workflow now cuts the GitHub Release automatically.** A new
  `github-release` job runs after a successful PyPI publish and calls
  `gh release create` for the pushed tag, so the Releases page can no
  longer drift behind PyPI.

## [0.2.2] - 2026-05-25

### Added — judge-driven assertion modes

- **Three new ``should_recall`` kwargs**:
  - ``latest_fact_is="X"`` — the latest fact must be present and
    dominant; older facts may appear only as historical framing.
  - ``must_not_answer_as=["X", "Y"]`` — the response must not assert
    any of the listed values as the current state.
  - ``judge_assertion=Rubric(criterion="...", pass_label=...,
    fail_label=...)`` — free-form rubric escape hatch with custom
    pass/fail labels.
- **AnthropicJudge backend.** Lazy-imported, ``temperature=0``,
  model-pinned (default ``claude-haiku-4-5-20251022``). Strict-JSON
  responses parsed via Pydantic with one malformed-JSON retry. Cost
  computed from token usage with configurable per-million prices.
- **NoOpJudge default.** ``[judge].provider = "none"`` (the default)
  disables judge modes. Install ``recalllab[judge]`` and set
  ``ANTHROPIC_API_KEY`` to enable.
- **Fail-loud default (Decision #3b).** A judge-mode kwarg used
  against an unconfigured judge raises ``JudgeUnavailableError`` —
  pytest reports ``ERROR``, never silently skips. Skip is opt-in
  via ``@pytest.mark.recalllab_optional("judge_configured")``.
- **Rule-first short-circuit (Decision #9).** When ``contains`` /
  ``excludes`` are combined with a judge mode in the same call,
  rule-based assertions evaluate first with fail-fast. A failing
  rule-based assertion never spends judge cost; the judge-mode
  kwarg lands as a placeholder ASSERT (``passed=None``) so
  ``recalllab record`` can regenerate the original call faithfully.
- **Diagnostic mode.** ``[judge].always_run = true`` makes the judge
  run even after a rule-based failure (for judge-vs-rule agreement
  comparisons). The rule-based ``AssertionError`` still wins for
  pytest reporting; the judge verdict only enriches the trace.
- **Two cost caps.** ``[judge].max_cost_usd`` bounds one
  ``ContractRun``; ``[judge].max_session_cost_usd`` bounds one
  ``pytest`` invocation (the cap that matters for CI). Both
  enforced post-call with a bounded one-invocation-plus-retry
  overshoot. Under ``pytest-xdist`` the session cap is per-worker;
  the plugin emits a config-time warning.
- **Deterministic prompt assembly.** Same identity tuple ``(query,
  recall_results, expected, rubric, model, mode,
  prompt_template_version)`` → byte-identical prompt every run,
  including the ``blake2s``-derived nonce fence. Verdict stability
  across provider snapshots is explicitly NOT guaranteed.
- **Prompt-injection mitigation.** Every user-supplied envelope
  field is JSON-encoded with ``<``/``>`` escaping; a deterministic
  nonce fence wraps the envelope. ``recall_result`` is truncated to
  16 KB with a marker so a pathological recall can't blow the
  prompt budget. Mitigation, not a guarantee — the model still sees
  hostile content as data and may follow it.
- **Capability resolver.** New ``judge_configured`` capability name
  registered alongside ``supports_*``; ``recalllab_optional`` walks
  every marker on a test (no longer ``get_closest_marker``).
  Unknown capability names raise ``pytest.UsageError`` at
  collection time so typos surface near the collected-items count.
- **Trace schema additions.** ``AssertionResult.passed`` is now
  three-valued (``bool | None``); ``None`` is the Decision #9
  placeholder. ``ContractRun.judge_cost_usd: float`` aggregates
  judge spend across the run (default 0.0). ``TraceEvent.cost_estimate``
  payload schema documented for judge-mode ASSERTs; ``raw_responses``
  stored on ASSERT ``payload`` per §Failed-judge ASSERT lifecycle.
- **Trace-to-test emitter extension.** ``recalllab record`` renders
  the three judge modes as ``should_recall`` kwargs;
  ``judge_assertion`` becomes a ``Rubric(criterion=..., ...)``
  literal with default-value omission for readability. New
  ``--optional-judge`` CLI flag (default off) adds the
  ``recalllab_optional("judge_configured")`` marker — off by default
  per the fail-loud guarantee. Short-circuited judge ASSERTs land
  as a documenting comment in the generated file.
- **133 new tests** across the judge-mode pipeline (311 collected,
  310 passing + 1 skipped when ``pytest-xdist`` isn't installed).

### Changed

- ``[judge]`` is now an optional-dependencies extra
  (``anthropic>=0.40``) added to ``[all]``.
- ``recalllab.toml`` scaffold dropped the ``"or openai"`` mention
  (v0.2.2 ships Anthropic only).
- pytest plugin: ``pytest_configure`` is harmless on unrelated
  runs; judge construction is deferred to first fixture use so a
  misconfigured ``[judge].provider`` only crashes a session that
  actually uses ``memory_contract``.
- New ``JudgeProvider`` Protocol (in this release) includes
  ``capabilities()``, ``evaluate(request)``, plus ``model_name`` and
  ``max_cost_usd`` properties. The DSL uses the properties to build
  ``JudgeRequest`` and enforce the per-run cap. No external v0.2.1
  code could have implemented this Protocol — it is new — but
  internal third-party consumers should note the surface.

### Fixed

- The trace-to-test emitter no longer treats judge modes as
  "unknown" and falls back to plain ``recall(...)``. v0.2.1
  regression tests that relied on that behavior have been updated.

### Migration notes (v0.2.1 → v0.2.2)

- Existing v0.2.1 contracts continue to work unchanged. Rule-based
  ``should_recall(contains=...)`` / ``excludes=...`` semantics are
  identical.
- Existing v0.2.1 traces load because new trace fields have
  defaults: ``ContractRun.judge_cost_usd = 0.0`` and
  ``AssertionResult.passed`` defaults are preserved.
- **External trace consumers must update one pattern**:
  ``AssertionResult.passed`` is now tri-state
  (``True | False | None``). ``None`` is the Decision #9
  short-circuit placeholder. Code that wrote ``if not assertion.passed:``
  must change to ``if assertion.passed is False:`` to avoid
  misclassifying placeholders as failures.

## [0.2.1] - 2026-05-20

### Added — trace-to-test generation

- **``recalllab record`` CLI subcommand.** Reads a recorded
  ``ContractRun`` from the SQLite trace store and emits a self-contained
  pytest regression file that replays the contract step-by-step. Two
  selection modes: ``--run-id <uuid>`` (replay a specific run) and
  ``--latest-failure`` (pick the freshest ``status=failed`` run — turn
  the latest CI failure into a checked-in regression in one command).
- **Pure, byte-stable emitter.** ``trace_to_test_source(run)`` is a pure
  transform over the Pydantic schema; same ``ContractRun`` always
  produces the same bytes. Timestamps, latencies, and the run UUID
  are intentionally absent from the output so re-recording the same
  logical contract produces no diff.
- **Cross-feature coverage.** The emitter renders ``given_user`` /
  ``remember`` / paired ``recall``+``assert`` (as ``should_recall``) /
  unpaired ``recall`` / ``forget(matching=...)`` /
  ``forget(episode_id=...)`` / ``with_distractors`` /
  ``with_stale_repeats``. Mutations recorded with
  ``status="unsupported"`` (capability gate) become a documenting
  comment rather than a re-attempted call; ``status="partial_failed"``
  mutations emit the call with a comment so the regenerated test
  reproduces the failure.
- **Adversarial input safety.** All payload values are rendered via
  ``repr()`` — text with embedded quotes, newlines, backslashes, or
  unicode round-trips into valid Python literals. Every emitted source
  string passes ``compile(...)``.
- **Tests.** 36 new tests covering the four mandatory stability
  properties (deterministic across calls, timestamp-invariant,
  UUID-invariant, content-sensitive), per-event-kind rendering, the
  ``--run-id`` / ``--latest-failure`` paths, and end-to-end round-trip
  (run pytest → trace store → ``recalllab record`` → re-run pytest on
  the emitted file).
- **Recorded episode IDs round-trip into the generated test.**
  ``MemoryContract.remember`` now accepts an optional ``episode_id``
  keyword; the trace-to-test emitter renders it as
  ``remember(text, episode_id='ep-X')`` whenever the recorded
  ``REMEMBER`` event carries an id. Without this, a regenerated
  ``remember("X")`` got a fresh provider-assigned uuid, then any
  subsequent ``forget(episode_id="ep-X")`` from the same trace
  silently deleted nothing — the regression file compiled, ran, and
  passed, but exercised a different contract than the original
  failure. Pinned by five tests including a ``runpy.run_path``-based
  end-to-end check that loads the generated file and asserts the
  forget actually removed the row.
- **DSL preserves v0.1 provider call shape on ordinary remembers.**
  Round-5's episode-id round-trip fix made ``MemoryContract.remember``
  always call ``provider.remember(user_id, text, episode_id=...)``,
  even when the caller didn't supply an id. Legacy third-party
  adapters written against the v0.1 protocol surface have
  ``remember(user_id, text)`` and don't accept the keyword — every
  ordinary ``memory_contract.remember("...")`` raised
  ``TypeError: unexpected keyword argument 'episode_id'``. The
  ``MemoryProvider`` Protocol is ``runtime_checkable`` but Python's
  structural typing only checks method *names*, so a mismatched
  signature passes ``isinstance(x, MemoryProvider)`` and only fails
  at call time. The DSL now forwards ``episode_id`` ONLY when the
  caller supplied it (round-11 Codex finding). The custom-id round-
  trip path is unchanged; the no-id happy path is restored.
- **Mid-trace user switching synthesises ``given_user``.** Round-7's
  synthesis only fired when the active user was ``None`` (trace
  opened with a user-dependent event). Round-10 Codex finding: a
  trace like ``GIVEN_USER alice → REMEMBER alice → REMEMBER bob`` (no
  intervening GIVEN_USER for bob) would replay bob's memory under
  alice — the generated regression silently exercised the wrong
  tenant. The emitter now synthesises a ``given_user`` whenever the
  payload ``user_id`` differs from the active one. Two comment
  templates distinguish the cases (``no GIVEN_USER recorded earlier
  in trace`` vs ``trace switched user_id without an intervening
  GIVEN_USER event``). Pinned by five tests including a
  ``runpy``-based end-to-end check that runs a multi-user trace
  against a real adapter and asserts each user's memory lands in
  the correct namespace.
- **Mutation traces also gate on ``supports_custom_episode_ids``.**
  Round-6 added the ``recalllab_optional`` marker for traces with a
  recorded REMEMBER ``episode_id``, but the detection helper
  explicitly excluded MUTATION events. That assertion was wrong:
  ``with_distractors`` / ``with_stale_repeats`` hit the DSL's
  capability gate at runtime, so a trace recorded on a capable
  provider would have generated a regression that
  ``RuntimeError``-failed on an incapable provider rather than cleanly
  skipping (round-9 Codex finding). The helper now also returns True
  for any non-``unsupported`` MUTATION event with a positive
  ``requested`` / ``times`` count. ``status="unsupported"`` and
  zero-write mutations stay excluded (no DSL call / no gate hit).
  Pinned by five tests covering completed distractors, completed
  stale_repeats, partial_failed mutations, the unsupported-only path,
  and the zero-count short-circuit.
- **Mutation IDs reproduce the recorded trace's IDs.** The mutation
  pipeline derives deterministic episode IDs (``mut-{type}-{sha256[:
  16]}-{index:04d}``) from ``contract_id`` + user + invocation + ....
  A generated test's pytest nodeid is different from the recorded
  run's, so without intervention the regenerated ``with_distractors``
  / ``with_stale_repeats`` wrote under different ``mut-*`` IDs than
  what landed at trace time — any later ``forget(episode_id=<mut-...>)``
  from the trace silently no-op'd. The emitter now prepends a pin
  (``memory_contract.run.contract_id = '<original>'``) to the function
  body whenever the trace contains a non-``unsupported`` MUTATION
  event. Pinned by five tests including a ``runpy``-based end-to-end
  check that builds a real ``with_distractors`` + ``forget(episode_id=
  mut-...)`` trace, runs the generated file against a fresh adapter
  with a deliberately different contract_id, and asserts the forget
  actually removed the targeted row.
- **Generator synthesises missing ``given_user``.** A trace whose
  first event is a user-dependent one (REMEMBER / RECALL / FORGET /
  MUTATION) without a preceding ``GIVEN_USER`` previously produced a
  regression that raised ``RuntimeError("no active user")`` from the
  DSL before reaching the recorded behaviour — a generator-induced
  setup failure that masked the real bug. RecallLab-produced traces
  can't have this shape (``_require_user`` would have raised at trace
  time), but external exports, partial dumps, and schema migrations
  can. The emitter now synthesises a ``given_user(payload['user_id'])``
  with a comment marking that the row didn't come from the trace.
  ``user_id``-less payloads still surface the runtime error (we can't
  honestly invent a user the trace didn't record). Pinned by six tests
  including a ``runpy``-based end-to-end check that loads a no-
  GIVEN_USER trace, runs it against a real ``ReferenceMemoryAdapter``,
  and asserts the remembered text actually landed under the right user.
- **Generated tests gate on ``supports_custom_episode_ids``.** When a
  trace carries a recorded ``episode_id``, the emitter now decorates
  the generated test with
  ``@pytest.mark.recalllab_optional("supports_custom_episode_ids")``
  (and emits the corresponding ``import pytest``). The existing
  pytest-plugin marker logic then auto-skips the test cleanly against
  providers that don't honour custom episode IDs — preventing the
  same silent-replay-against-wrong-id bug as round 5 but at the
  provider boundary instead of the emitter boundary. ID-free traces
  produce minimal output (no pytest import, no decorator). Pinned by
  three tests covering the marker presence, its absence on id-free
  traces, and the direct-call compatibility that the end-to-end
  ``runpy`` test depends on.
- **Reference adapter is thread-safe in-process.** Connection now
  opened with ``check_same_thread=False`` and every public method's
  database access runs under a ``threading.Lock``. The
  SELECT-then-INSERT idempotency check in ``remember(episode_id=X)``
  is therefore atomic — two threads racing the same id with different
  content get deterministic semantics (one writes, the other raises
  ``ValueError``) rather than non-deterministic
  ``IntegrityError`` from the SQLite primary-key constraint. Mirrors
  the round-12 LangGraph adapter pattern. Five concurrency regression
  tests in ``tests/integration/test_reference_adapter_threading.py``.
  Cross-process safety remains explicitly out of scope.
- **Empty packages removed.** ``src/recalllab/core/{runner,judge,
  assertions}/__init__.py`` were carved out for v0.2.x work but stayed
  empty. They've been deleted to keep the architecture map honest;
  mypy source-file count drops from 29 → 26.

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

[unreleased]: https://github.com/iushv/recalllab/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/iushv/recalllab/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/iushv/recalllab/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/iushv/recalllab/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/iushv/recalllab/releases/tag/v0.2.0
[0.1.0]: https://github.com/iushv/recalllab/releases/tag/v0.1.0
