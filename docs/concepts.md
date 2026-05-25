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
  precisely. Shipped in v0.2.2 — see §Judge-driven assertions below.
  Gated on a configured ``[judge]`` section in
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

## Judge-driven assertions *(v0.2.2)*

Rule-based ``contains`` / ``excludes`` cover literal-string expectations
("Mumbai must appear in the recall"). They don't cover *semantic*
ones — temporal correctness, contradiction resolution, free-form
rubrics. v0.2.2 adds three new ``should_recall`` kwargs that escalate
to an LLM judge when rule-based isn't enough:

```python
memory_contract.should_recall(
    "Where do I live?",
    latest_fact_is="Mumbai",
)
memory_contract.should_recall(
    "Where do I live?",
    must_not_answer_as=["Bangalore", "Delhi"],
)
from recalllab import Rubric
memory_contract.should_recall(
    "What did I buy?",
    judge_assertion=Rubric(
        criterion="The response must cite the source episode.",
        pass_label="CITED",
        fail_label="UNCITED",
    ),
)
```

### Configuration

Judge modes require ``[judge]`` configured in ``recalllab.toml``:

```toml
[judge]
provider = "anthropic"           # "none" (default) disables judge modes.
# model = "claude-haiku-4-5-20251022"  # pinned snapshot; bump intentionally.
# max_cost_usd = 0.10            # per-contract cap.
# max_session_cost_usd = 1.00    # per-pytest-invocation cap.
# always_run = false             # diagnostic mode (see below).
```

Install with the ``[judge]`` extra and set ``ANTHROPIC_API_KEY``:

```bash
pip install 'recalllab[judge]'
export ANTHROPIC_API_KEY=sk-...
```

### Fail-loud default (Decision #3b)

A judge-mode kwarg in a committed contract **fails loudly** when
``[judge]`` isn't configured. The DSL raises
``JudgeUnavailableError`` and pytest reports the test as ``ERROR``.
That keeps a forgotten CI configuration visible — silent skips on
semantic checks defeat the purpose of having the checks at all.

Skip is opt-in via the marker:

```python
@pytest.mark.recalllab_optional("judge_configured")
def test_my_contract(memory_contract):
    ...
```

The same marker is the reason ``recalllab record`` defaults the
``--optional-judge`` flag to off: a regenerated regression that lives
in CI should ERROR loudly when judge isn't configured, not silently
skip.

### Combined rule + judge (Decision #9)

A ``should_recall`` call can combine any number of rule-based
kwargs (both ``contains`` and ``excludes`` may be present) with **at
most one** judge-mode kwarg. **Rule-based evaluates first with
fail-fast:** a failing ``contains="X"`` never spends judge cost.
The judge-mode kwarg lands on the trace as a placeholder ASSERT
(``passed=None, reason="short_circuited"``) so ``recalllab record``
can faithfully regenerate the original combined call. Combining two
judge modes in one call raises ``ValueError`` at call time
(Decision #3a).

### Diagnostic mode (always_run)

Set ``[judge].always_run = true`` to run the judge **even after a
rule-based assertion failed**. Useful for comparing judge vs
rule-based agreement across the suite. The judge verdict never
overrides the rule-based ``AssertionError`` for pytest reporting —
the original failure still wins — but the judge ASSERT is recorded
with real cost and verdict. Default is off so failed tests don't
drain judge budget.

### Cost & budget

Two caps, both enforced **post-call**:

- ``max_cost_usd`` bounds one ``ContractRun``.
- ``max_session_cost_usd`` bounds one ``pytest`` invocation — this
  is the cap that matters for CI cost protection.

Worst-case overshoot per cap is one full judge invocation including
its malformed-JSON retry. The next invocation refuses to start once
the running total has reached the cap. Under ``pytest-xdist`` the
session cap is **per-worker** (cross-worker aggregation is a v0.3
follow-up); the plugin emits a session-start warning when xdist is
detected.

### Determinism & drift

Prompt assembly is deterministic — same identity tuple ``(query,
recall_results, expected, rubric, model, mode, prompt_template_version)``
produces the same byte-identical prompt every run, including a
deterministically-derived nonce fence (``blake2s`` of the envelope).
**Judge verdicts are not guaranteed across provider snapshots**:
``temperature=0`` minimizes same-snapshot variance but Anthropic may
update the underlying weights even when the model name is pinned. Mix
judge assertions with rule-based ones deliberately — rule-based tests
are load-bearing, judge tests catch semantic regressions but
introduce model-drift as a CI flake source.

### Prompt-injection mitigation (not guarantee)

Every user-supplied envelope field (``recall_result``, ``query``,
``expected``, ``rubric``) is JSON-encoded inside a structured
envelope; ``<``/``>`` are explicitly escaped on top of JSON's
built-in escaping. A deterministically-derived nonce fence wraps the
envelope so a hostile recall cannot pre-write a closing-tag
injection. This protects *prompt structure* — the model still sees
the hostile string as data and may, in principle, choose to follow
it. Adversarial tests verify the prompt is assembled safely and that
the verdict is stable on a fixed set of known-hostile inputs; they
do not prove the model is unjailbreakable.

For the full design rationale, locked decisions, and the §Failed-judge
ASSERT lifecycle table, see ``docs/judge-assertions.md``.

## Trace-to-test *(v0.2.1)*

``recalllab record`` reads a recorded ``ContractRun`` from the SQLite
trace store and emits a checked-in pytest regression file that replays
the run. The emitter is a pure transform over the persisted
``ContractRun`` schema — no LLM, no external API, no provider calls.
Given a real production failure recorded as a trace, the user gets a
regression test that reproduces the failure on the next CI run.

### Surface

```bash
# Pick a specific run by UUID.
recalllab record --trace .recalllab/traces.sqlite --run-id <uuid> \
                 --out tests/regressions/test_real_failure.py

# Or pick the most recent failure.
recalllab record --latest-failure --out tests/regressions/test_real_failure.py

# --force is required to overwrite an existing file at --out.
```

The output is a self-contained pytest file with a single
``def test_recorded_failure(memory_contract)`` function that walks
through the trace's events in order: ``given_user`` → ``remember`` →
``recall`` / ``should_recall`` → ``forget`` → ``with_distractors`` /
``with_stale_repeats``.

### Adversarial-input safety

Every value that originated from trace payload data is rendered through
``repr()`` or a comment-quarantine helper so hostile recall text,
pytest node IDs, or assertion reasons can't inject code into the
generated regression. Specifically:

- ``contract_id`` and ``status`` go to ``# Source contract: <!r>`` /
  ``# Original status: <!r>`` comment lines — never interpolated into
  the module docstring.
- DSL arg payloads (``text``, ``query``, ``expected``, ``episode_id``,
  ``user_id``) go through ``_py()`` (``repr``-based).
- Multi-line assertion reasons go through ``_safe_comment_lines``,
  which splits on ``\n`` / ``\r`` and re-prefixes every line with
  ``#``. A payload-embedded newline can't escape the comment context
  and turn the next line into executable Python.

Every output is checked with ``compile(source, ..., "exec")`` in the
test suite — any escape-from-comment regression fails at the syntax
level, not silently.

### Fidelity guarantees (recorded ⇔ regenerated)

The point of trace-to-test is that the regenerated test actually
reproduces the original behaviour. Several safeguards make that real
across non-canonical traces:

- **Recorded ``episode_id`` round-trips.** The DSL's
  ``remember(text, episode_id=...)`` overload accepts the recorded
  id; the emitter forwards it. A later ``forget(episode_id=X)`` from
  the same trace addresses the same row in the regenerated run
  (round-5 fix).
- **Capability gating on ``supports_custom_episode_ids``.** Whenever
  the trace's REMEMBER carries an id, OR whenever it contains a
  non-``unsupported`` mutation that writes episodes, the generated
  test is decorated with
  ``@pytest.mark.recalllab_optional("supports_custom_episode_ids")``
  so the pytest plugin skips cleanly on providers that don't honour
  custom IDs (rounds 6 & 9).
- **``contract_id`` pinning for mutation replay.** Mutation episode
  IDs (``mut-{type}-{sha256[:16]}-{index:04d}``) are hashed from
  ``contract_id``. A generated test's pytest nodeid is different from
  the recorded run's, so without intervention the regenerated
  mutation would write under different IDs. The emitter prepends
  ``memory_contract.run.contract_id = '<original>'`` whenever the
  trace has mutations, so the ``mut-*`` IDs reproduce (round-8 fix).
- **Synthesised ``given_user`` for incomplete traces.** A trace that
  starts with REMEMBER / RECALL / FORGET / MUTATION (no preceding
  GIVEN_USER) or that switches user mid-stream without an explicit
  GIVEN_USER would otherwise replay under the wrong tenant — or
  raise ``RuntimeError("no active user")``. The emitter detects
  this and synthesises a ``given_user`` from the event's payload
  ``user_id`` with a documenting comment so the developer can audit
  what was inferred (rounds 7 & 10).
- **Multi-assert ``should_recall`` preserved.** A
  ``should_recall(query, contains=X, excludes=Y)`` records ONE
  RECALL + TWO ASSERT events. The emitter collects every contiguous
  ASSERT after a RECALL and merges them into one call (round-3 fix);
  failed-assertion reasons get per-mode labelled comments.
- **Legacy provider compat.** ``MemoryContract.remember(text)``
  without ``episode_id`` only forwards the kwarg to the provider
  when the caller supplied one. v0.1-era third-party adapters whose
  ``remember`` signature is ``(user_id, text)`` keep working
  (round-11 fix).

### Reproduction-fidelity caveats

Two cases where the regression file documents but doesn't guarantee
faithful reproduction:

- ``status="partial_failed"`` mutations replay the call against a
  fresh adapter. The original failure can have been
  provider-state-dependent (mid-call disconnect, transient race);
  the regenerated test may pass or fail with a different shape.
  The generated file marks this with a ``# NOTE: ... results may
  diverge ...`` comment so the developer reading it knows the
  caveat.
- Recall result *ranking* depends on the provider's retrieval
  backend (BM25 vs keyword overlap vs embeddings). The rule-based
  assertions (``contains`` / ``excludes``) survive ranking changes,
  but the trace's recorded ``results`` array is not replayed
  against — the regenerated recall re-queries the fresh adapter.

A future ``--strict`` mode (v0.3) may turn either case into a hard
failure.

### Write safety

The CLI refuses to overwrite an existing file at ``--out`` by
default; pass ``--force`` to opt in. With ``--force``, the write
goes through ``tempfile.mkstemp`` + ``os.replace`` in the
destination's parent directory, so a crash, signal, or disk-full
event between truncate and the final write can never leave a
half-written regression in the user's tree (round-4 fix).

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
