# RecallLab

> **pytest for agent memory.** Turn memory expectations into regression tests that run in CI.

[![CI](https://github.com/iushv/recalllab/actions/workflows/ci.yml/badge.svg)](https://github.com/iushv/recalllab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict)

Write executable memory contracts (`must-remember`, `must-update`, `must-forget`, `must-isolate`, `must-cite`). Run them on every PR. Inspect failures as traces in a local Failure Gallery. Works against your own provider, [LangGraph Store][langgraph-store], any [MCP][mcp] memory server, or the bundled SQLite reference backend.

[langgraph-store]: https://docs.langchain.com/oss/python/langgraph/add-memory
[mcp]: https://modelcontextprotocol.io/

## What it looks like

```bash
$ uv run pytest tests/memory
====================== 6 passed in 0.01s =======================
```

```python
# tests/memory/test_updated_location.py
def test_updated_location_overrides_stale_memory(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I live in Bangalore.")
    memory_contract.remember("Correction: I moved to Mumbai.")
    memory_contract.should_recall("Where do I live now?", contains="Mumbai")
```

```python
# tests/memory/test_forget_compliance.py
def test_forget_removes_allergy_immediately(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I am allergic to peanuts.")
    # Precondition: prove retrievability BEFORE forget, otherwise the
    # post-forget excludes assertion is vacuously true.
    memory_contract.should_recall("What am I allergic to?", contains="peanut")
    deleted = memory_contract.forget(matching="peanuts")
    assert deleted == 1
    memory_contract.should_recall("What am I allergic to?", excludes="peanut")
```

```python
# tests/memory/test_tenant_isolation.py
def test_user_b_cannot_see_user_a_memories(memory_contract):
    memory_contract.given_user("alice").remember("Project codename: Aurora.")
    memory_contract.should_recall("What is the project codename?", contains="Aurora")
    memory_contract.given_user("bob")
    memory_contract.should_recall("What is the project codename?", excludes="Aurora")
```

The hero examples are deterministic — `pip install recalllab && pytest` runs them green with **no API key, no Postgres, no infra**. Judge-driven assertion modes (`latest_fact_is`, `must_not_answer_as`, `judge_assertion`) shipped in v0.2.2 — set `[judge].provider = "anthropic"` in `recalllab.toml`, `pip install 'recalllab[judge]'`, and export `ANTHROPIC_API_KEY` to enable them. See [`docs/concepts.md`](docs/concepts.md#judge-driven-assertions-v022) for the full design.

## Why

Memory benchmarks tell you which provider is *"best on average."* That doesn't help you when you want to know whether your agent still passes its `must-remember` test on Tuesday after you changed the prompt.

RecallLab is the layer underneath the benchmark suites: a pytest plugin that turns *your app's* memory expectations into executable regression tests. Existing benchmarks become upstream sources for v0.2, not competitors.

| | MemoryBench | AMB / 56-test bench | MemoryAgentBench | Iris / Phoenix | **RecallLab** |
|---|---|---|---|---|---|
| Layer | benchmark suite | benchmark suite | academic benchmark | general agent obs | **per-app regression harness** |
| Audience | benchmark authors | benchmark authors | researchers | infra teams | **app developers writing CI** |
| Test format | benchmark dataset | dataset + scenarios | dataset | freeform eval rules | **pytest tests** |
| pytest-native? | no | no | no | no | **yes (`pytest11`)** |
| Trace → regression test? | no | no | no | partial | **yes (`recalllab record`, v0.2.1)** |
| Contract mutations? | no | no | no | no | **yes (`with_distractors`, `with_stale_repeats`, v0.2.0)** |
| Semantic / judge assertions? | n/a (fixed rubrics) | n/a | n/a | freeform eval rules | **yes (`latest_fact_is`, `must_not_answer_as`, `judge_assertion`, v0.2.2)** |
| Imports other benchmarks? | n/a | n/a | n/a | n/a | **yes (v0.2.x)** |

The wedge: RecallLab is **the test runner**, not the test corpus.

## Quickstart

```bash
# Install (until PyPI publishing lands)
pip install "git+https://github.com/iushv/recalllab.git"
# For judge-driven assertion modes:
pip install "git+https://github.com/iushv/recalllab.git#egg=recalllab[judge]"

# Scaffold tests/memory/ + recalllab.toml
recalllab init

# Run the six example contracts against the SQLite reference backend
pytest tests/memory          #  6 passed in 0.01s
```

To browse failures visually:

```bash
pip install "recalllab[dashboard] @ git+https://github.com/iushv/recalllab.git"
recalllab dashboard          # serves localhost:8080
```

The Failure Gallery groups runs by status (failed → skipped → passed), shows the failed assertion reason and last recall query on each card, and links to a per-run detail page rendering the full ordered event trace (`given_user → remember → recall → assert`) with per-event latency.

To turn a recorded failure into a checked-in pytest regression *(v0.2.1)*:

```bash
recalllab record --latest-failure --out tests/regressions/test_real_failure.py
pytest tests/regressions/test_real_failure.py     # reproduces the CI failure
```

The emitter is a pure transform over the persisted `ContractRun` — byte-stable, no LLM, no external API. Every payload value goes through `repr()` or a comment-quarantine helper so hostile recall text / pytest node IDs / assertion reasons can't inject code into the generated test file. Recorded episode IDs round-trip into the generated test, so any later `forget(episode_id=X)` from the original run actually targets the same row in the regenerated one (no silent no-ops). Refuses to overwrite an existing file unless `--force` is passed, and uses atomic-write (`tempfile.mkstemp` + `os.replace`) when overwriting is permitted so a crash mid-write can't leave a half-written regression in your tree.

## Six failure-mode categories

The example contracts ship with one test per category, all of them runnable on day one against the reference backend:

| # | Category | Tests for |
|---|---|---|
| 1 | Cross-session recall | A fact stated early in the conversation persists later. |
| 2 | Temporal updates | The new value is retrievable after the user updated a fact. |
| 3 | Contradiction resolution | Latest value is surfaced when contradictory facts exist. |
| 4 | Provenance | Every recalled fact cites a source episode ID. |
| 5 | Privacy / tenant isolation | User B never sees memories stored for User A. |
| 6 | Forget compliance | Deleted memories don't appear in subsequent recalls. |

## Providers

Switch providers via `recalllab.toml`:

```toml
[provider]
type = "reference"      # or "langgraph_store" or "mcp"
```

| Provider | Backend | Capability flags |
|---|---|---|
| `reference` | In-process SQLite (FTS5 + keyword fallback) | forget, tenant-delete, provenance, scores, candidate-trace |
| `langgraph_store` | Any [`langgraph.store.BaseStore`][langgraph-store] | forget, tenant-delete, provenance, scores (when indexed) |
| `mcp_configurable` | Any FastMCP-compatible MCP memory server | derived from `MCPMemoryConfig` (tool names, dotted result-field paths) |

All six example contracts pass against all three providers. Capability gating (`@pytest.mark.recalllab_optional("supports_provenance")`) lets contracts skip cleanly when a configured provider lacks a capability — and the skip is persisted to the trace store so the Failure Gallery renders an explicit `N/A — capability not supported` row, not a misleading zero.

## Contract mutations *(v0.2.0)*

Deterministic seeded transformations that stress a contract by polluting
or amplifying the active user's namespace before the next recall.
`with_distractors` defaults to `seed=0`, so contracts are reproducible
without callers having to remember to supply a seed; pass an explicit
non-zero seed to vary the sample.

Every mutation emits a `MUTATION` trace event with the deterministic
episode IDs it requested (`mut-{type}-{hash}-{index}`), the `status`
(`completed` or `partial_failed`), and the captured exception repr on
failure — so the Failure Gallery shows exactly what was injected, and
retries against hosted providers (Mem0, Zep, custom MCP) request the
same IDs rather than silently double-writing with fresh UUIDs.

```python
def test_birthday_survives_distractors(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("My birthday is December 28.")
    memory_contract.with_distractors(n=20, seed=7)
    memory_contract.should_recall("When is my birthday?", contains="December 28")


def test_temporal_update_under_stale_repeats(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I live in Bangalore.")
    memory_contract.with_stale_repeats(times=3)   # 4 copies of the stale fact
    memory_contract.remember("Correction: I moved to Mumbai.")
    memory_contract.should_recall("Where do I live now?", contains="Mumbai")
```

Both mutations are scoped to the active user — distractors never cross
into other tenants, repeats never duplicate someone else's facts. On a
mid-mutation provider exception the trace records the partial set of
inserted IDs with `status="partial_failed"` and re-raises so the test
fails loudly. The remaining mutations from the v0.1 plan
(`with_paraphrases`, `with_tenant_swap`, `with_delete_reinsert`) land in
v0.2.x once judge-based assertions and the trace-to-test workflow are in
place.

## Architecture

```
your tests/memory/                                 your provider
       │                                                  │
       ▼                                                  ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ memory_contract  │ ─▶ │  Contract DSL    │ ─▶ │  MemoryProvider      │
│ pytest fixture   │    │  + assertions    │    │  reference / LG /    │
└──────────────────┘    └────────┬─────────┘    │  MCP / your own      │
                                 │               └──────────────────────┘
                                 ▼
                       ┌──────────────────┐    ┌──────────────────┐
                       │  SQLite trace    │ ──▶│  Failure Gallery │
                       │  store + status  │    │  (FastAPI+htmx)  │
                       │  capture hook    │    │                  │
                       └──────────────────┘    └──────────────────┘
```

## Status

**v0.1** — pytest plugin + DSL with `contains` / `excludes` rule-based assertions, three adapters (`reference`, `langgraph_store`, `mcp_configurable`), Failure Gallery dashboard, 27 tests. Adversarial review passed; the four findings caught are locked behind regression tests in [`tests/unit/test_codex_review_regressions.py`](tests/unit/test_codex_review_regressions.py).

**v0.2.0** — contract mutations: `with_distractors(n, *, seed=0)` and `with_stale_repeats(*, times)`. Deterministic episode IDs, partial-failure tracing, capability gate for `supports_custom_episode_ids`, two-layered resurrection guard, in-flight fingerprint map for retry idempotency. **Twelve rounds** of adversarial review against the mutation system; every finding has a regression test.

**v0.2.1** — trace-to-test generation: `recalllab record --trace <path> --run-id <id> --out test.py` (or `--latest-failure`). Reads a recorded `ContractRun` from the SQLite trace store and emits a self-contained pytest regression file that replays the contract. Safe against hostile payloads (every interpolation site uses `repr()` or comment-quarantine). Recorded episode IDs round-trip into the generated test so ID-paired `forget` / capability-checked recall actually addresses the same row in the regenerated run. Atomic write via temp-file + `os.replace` with default refuse-to-overwrite; `--force` to opt in. Reference adapter is now thread-safe in-process (`threading.Lock` + `check_same_thread=False`, mirroring the round-12 LangGraph pattern). Five rounds of adversarial review.

**v0.2.2** — judge-driven assertion modes: `latest_fact_is`, `must_not_answer_as`, `judge_assertion=Rubric(...)`. `AnthropicJudge` backend with `temperature=0` + model pinning + post-call budget enforcement (per-run AND per-pytest-session caps, with explicit `pytest-xdist` caveat). Fail-loud default — a judge-mode kwarg without configured `[judge]` raises `JudgeUnavailableError` so forgotten CI config doesn't silently skip semantic checks. Rule-first short-circuit means a failing `contains="X"` never spends judge cost. Deterministic prompt assembly with a `blake2s`-derived nonce fence and explicit `<`/`>` escaping. `recalllab record` extended to render judge modes as kwargs and `Rubric(...)` literals; `--optional-judge` flag adds the skip marker (default off). Three rescue rounds + four adversarial rounds against the design doc, plus per-step adversarial review against the implementation.

**v0.2.x roadmap** (see [`CHANGELOG.md`](CHANGELOG.md)):
- Benchmark importers: LongMemEval, LoCoMo, MemoryAgentBench → contract DSL
- pgvector / embedding-based reference adapter
- Verdict caching: skip the judge when ``(query, recall, expected,
  rubric, model, mode, prompt_template_version)`` is already verdicted

**Gates** — 310 passing tests + 1 conditionally-skipped (xdist) · mypy `--strict` clean across 32 source files · ruff clean · CI matrix py3.11/3.12 + wheel build verification.

## Docs

- [`docs/concepts.md`](docs/concepts.md) — failure modes, contract DSL, providers, capability flags, traces.

## License

[Apache 2.0](LICENSE).
