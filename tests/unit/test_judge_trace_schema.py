"""Tests for the v0.2.2 step 3 trace-schema additions.

Covers:

- ``AssertionResult.passed`` accepts ``True`` / ``False`` / ``None``
  (Decision #9 short-circuit placeholder). Run-status logic only flips
  to ``FAILED`` on ``passed is False``; ``None`` placeholders do not
  count as failures.
- ``ContractRun.judge_cost_usd`` defaults to ``0.0`` and accepts a
  positive float.
- ``TraceEvent.cost_estimate`` carries the documented judge-call
  payload on judge-mode ASSERT events; rule-based ASSERTs leave it
  ``None``.
- SQLite round-trip preserves all three additions.
- ``[judge]`` config defaults include ``max_cost_usd`` and
  ``max_session_cost_usd`` per the docstring in
  ``docs/judge-assertions.md`` §Cost & budget.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.judge import NoOpJudge
from recalllab.core.pytest_plugin.plugin import _DEFAULT_CONFIG
from recalllab.core.traces.schema import (
    AssertionResult,
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)
from recalllab.core.traces.sqlite_store import TraceStore


def _new_run(**overrides: object) -> ContractRun:
    """Build a minimal ContractRun for schema tests."""
    base = {
        "id": "run-test",
        "contract_id": "test::contract",
        "provider": "reference",
        "started_at": datetime.now(tz=UTC),
        "status": RunStatus.PASSED,
    }
    base.update(overrides)
    return ContractRun(**base)


# -------------------------------------------------------- AssertionResult.passed


def test_assertion_result_accepts_passed_true() -> None:
    """``passed=True`` represents an evaluated, passing assertion."""
    result = AssertionResult(
        passed=True,
        mode="contains",
        expected="X",
        actual="X is here",
        sequence=0,
    )
    assert result.passed is True


def test_assertion_result_accepts_passed_false() -> None:
    """``passed=False`` represents an evaluated, failing assertion."""
    result = AssertionResult(
        passed=False,
        mode="contains",
        expected="X",
        actual="something else",
        sequence=0,
    )
    assert result.passed is False


def test_assertion_result_accepts_passed_none_placeholder() -> None:
    """``passed=None`` is the Decision #9 short-circuit placeholder.

    Used when a combined rule + judge call short-circuits on the
    rule-based side; the judge-mode kwarg's ASSERT is still emitted so
    ``recalllab record`` can regenerate both kwargs faithfully.
    """
    result = AssertionResult(
        passed=None,
        mode="latest_fact_is",
        expected="Mumbai",
        actual="",
        reason="short_circuited: preceding rule-based assertion failed",
        sequence=1,
    )
    assert result.passed is None


def test_assertion_result_passed_is_required() -> None:
    """``passed`` has no default — the writer must pick an explicit value."""
    with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError varies
        AssertionResult(  # type: ignore[call-arg]
            mode="contains",
            expected="X",
            actual="X",
            sequence=0,
        )


# ------------------------------------------------ run status three-valued logic


def test_dsl_short_circuit_placeholder_does_not_flip_status() -> None:
    """``_record_assertion(passed=None, ...)`` must NOT flip run status to FAILED.

    Step 5+ will exercise this for real (placeholder emitted when a
    judge-mode call short-circuits behind a failing rule-based one); for
    step 3 we just verify the run-status logic handles ``passed=None``
    correctly. Calling ``_record_assertion`` directly is the cleanest
    way to test the run-status branching.
    """
    provider = ReferenceMemoryAdapter()
    run = _new_run()
    contract = MemoryContract(provider, run, judge=NoOpJudge())
    # Use the private recorder directly — public DSL doesn't emit
    # placeholders yet (lands in step 5).
    contract._record_assertion(
        passed=None,
        mode="latest_fact_is",
        expected="Mumbai",
        actual="",
        reason="short_circuited: preceding rule-based assertion failed",
    )
    assert run.status == RunStatus.PASSED, (
        "passed=None placeholder must not flip status to FAILED; only "
        "passed is False counts as a true failure"
    )
    # And the assertion IS recorded on both surfaces (assertions list + ASSERT event).
    assert len(run.assertions) == 1
    assert run.assertions[0].passed is None
    assert run.assertions[0].reason is not None
    assert "short_circuited" in run.assertions[0].reason
    assert len(run.events) == 1
    assert run.events[0].kind == EventKind.ASSERT
    assert run.events[0].payload["passed"] is None


def test_dsl_real_failure_still_flips_status() -> None:
    """``passed=False`` must still flip run status to FAILED (regression guard)."""
    provider = ReferenceMemoryAdapter()
    run = _new_run()
    contract = MemoryContract(provider, run, judge=NoOpJudge())
    contract._record_assertion(
        passed=False,
        mode="contains",
        expected="X",
        actual="something else",
        reason="not found",
    )
    assert run.status == RunStatus.FAILED


# ---------------------------------------------------- ContractRun.judge_cost_usd


def test_contract_run_judge_cost_defaults_to_zero() -> None:
    """``judge_cost_usd`` defaults to 0.0 so rule-based-only runs are unchanged."""
    run = _new_run()
    assert run.judge_cost_usd == 0.0


def test_contract_run_judge_cost_accepts_positive_float() -> None:
    """``judge_cost_usd`` is mutable on the run model so the runtime can accumulate."""
    run = _new_run()
    # ContractRun is not frozen — judge accounting accumulates as judge
    # calls complete inside should_recall.
    run.judge_cost_usd = 0.00347
    assert run.judge_cost_usd == pytest.approx(0.00347)


def test_contract_run_judge_cost_rejects_negative() -> None:
    """``judge_cost_usd`` must be non-negative (Field(ge=0.0))."""
    with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError varies
        _new_run(judge_cost_usd=-0.01)


# ----------------------------------------------------- TraceEvent.cost_estimate


def test_trace_event_cost_estimate_judge_payload_shape() -> None:
    """``cost_estimate`` accepts the documented judge-call payload schema."""
    judge_event = TraceEvent(
        sequence=0,
        kind=EventKind.ASSERT,
        payload={"mode": "latest_fact_is", "passed": True},
        timestamp=datetime.now(tz=UTC),
        cost_estimate={
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251022",
            "input_tokens": 420,
            "output_tokens": 18,
            "estimated_usd": 0.00134,
            "attempts": 1,
        },
    )
    assert judge_event.cost_estimate is not None
    assert judge_event.cost_estimate["estimated_usd"] == pytest.approx(0.00134)
    assert judge_event.cost_estimate["attempts"] == 1


def test_trace_event_rule_based_cost_estimate_stays_none() -> None:
    """Rule-based ASSERT events leave ``cost_estimate=None`` as today."""
    rule_event = TraceEvent(
        sequence=0,
        kind=EventKind.ASSERT,
        payload={"mode": "contains", "passed": True},
        timestamp=datetime.now(tz=UTC),
    )
    assert rule_event.cost_estimate is None


# -------------------------------------------------------- SQLite round-trip


def test_sqlite_round_trip_preserves_judge_additions(tmp_path: Path) -> None:
    """Every v0.2.2 schema addition survives a write-then-read cycle.

    Without this, a generated regression after step 4 lands could lose
    judge_cost_usd or the placeholder ASSERT's ``passed=None`` value.
    """
    store = TraceStore(tmp_path / "trace.sqlite")
    judge_cost_payload = {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251022",
        "input_tokens": 100,
        "output_tokens": 12,
        "estimated_usd": 0.0008,
        "attempts": 2,  # 1 valid + 1 malformed-JSON retry
    }
    original = ContractRun(
        id="run-roundtrip",
        contract_id="test::roundtrip",
        provider="reference",
        started_at=datetime.now(tz=UTC),
        finished_at=datetime.now(tz=UTC),
        status=RunStatus.PASSED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.RECALL,
                payload={"query": "Where do I live?", "k": 5},
                timestamp=datetime.now(tz=UTC),
            ),
            # Real passing judge ASSERT carries cost_estimate.
            TraceEvent(
                sequence=1,
                kind=EventKind.ASSERT,
                payload={
                    "mode": "latest_fact_is",
                    "expected": "Mumbai",
                    "passed": True,
                },
                timestamp=datetime.now(tz=UTC),
                cost_estimate=judge_cost_payload,
            ),
            # Decision #9 placeholder: passed=None on a judge mode that
            # was short-circuited behind a failed rule-based assertion.
            TraceEvent(
                sequence=2,
                kind=EventKind.ASSERT,
                payload={
                    "mode": "must_not_answer_as",
                    "expected": ["Delhi"],
                    "passed": None,
                    "reason": "short_circuited",
                },
                timestamp=datetime.now(tz=UTC),
            ),
        ],
        assertions=[
            AssertionResult(
                passed=True,
                mode="latest_fact_is",
                expected="Mumbai",
                actual="ayush lives in Mumbai",
                sequence=1,
            ),
            AssertionResult(
                passed=None,
                mode="must_not_answer_as",
                expected=["Delhi"],
                actual="",
                reason="short_circuited",
                sequence=2,
            ),
        ],
        judge_cost_usd=0.0008,
    )
    store.write_run(original)

    loaded = store.get_run("run-roundtrip")
    assert loaded is not None
    assert loaded.judge_cost_usd == pytest.approx(0.0008)

    # Real judge ASSERT cost_estimate survives.
    loaded_judge_event = loaded.events[1]
    assert loaded_judge_event.cost_estimate == judge_cost_payload

    # Placeholder ASSERT preserves passed=None.
    loaded_placeholder = loaded.events[2]
    assert loaded_placeholder.payload["passed"] is None
    assert loaded_placeholder.payload["reason"] == "short_circuited"

    # AssertionResult three-valued passed survives both rows.
    loaded_real_assert = loaded.assertions[0]
    assert loaded_real_assert.passed is True
    loaded_placeholder_assert = loaded.assertions[1]
    assert loaded_placeholder_assert.passed is None


# -------------------------------------------- judge config defaults


def test_judge_config_defaults_include_cost_caps() -> None:
    """``_DEFAULT_CONFIG['judge']`` ships both caps so step 4 has knobs to read.

    Cost-budget enforcement lands in step 4 (AnthropicJudge); step 3 just
    wires the config defaults so the keys are present.
    """
    judge = _DEFAULT_CONFIG["judge"]
    assert judge["provider"] == "none"
    assert judge["max_cost_usd"] == pytest.approx(0.10)
    assert judge["max_session_cost_usd"] == pytest.approx(1.00)
