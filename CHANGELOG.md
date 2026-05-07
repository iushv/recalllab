# Changelog

All notable changes to RecallLab are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/iushv/recalllab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/iushv/recalllab/releases/tag/v0.1.0
