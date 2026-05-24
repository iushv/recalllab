"""End-to-end integration tests for the latest_fact_is judge mode.

Tests build a ``MemoryContract`` with a configured fake judge and
verify the full path: recall -> rule-based assertions (if any) ->
judge evaluation -> trace ASSERT row with cost_estimate ->
``ContractRun.judge_cost_usd`` accumulation -> short-circuit
placeholder when combined with a failing rule-based assertion.

The fake judge avoids real Anthropic API calls; AnthropicJudge's
own evaluation logic is covered in ``test_anthropic_judge.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.judge import (
    JudgeBudgetExceededError,
    JudgeCapabilities,
    JudgeCostEstimate,
    JudgePartialFailureError,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
    NoOpJudge,
)
from recalllab.core.traces.schema import ContractRun, EventKind, RunStatus


class _FakeJudge:
    """Configurable JudgeProvider that returns scripted verdicts.

    Mimics the AnthropicJudge surface without any network. Constructor
    takes a list of verdicts (or exceptions) that ``evaluate`` returns
    or raises in order.
    """

    def __init__(
        self,
        scripted: list[JudgeVerdict | Exception] | None = None,
        *,
        model_name: str = "test-model",
        max_cost_usd: float = 0.10,
        available: bool = True,
    ) -> None:
        self._scripted: list[JudgeVerdict | Exception] = list(scripted or [])
        self._model_name = model_name
        self._max_cost_usd = max_cost_usd
        self._available = available
        self.evaluate_calls: list[JudgeRequest] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_cost_usd(self) -> float:
        return self._max_cost_usd

    def capabilities(self) -> JudgeCapabilities:
        return JudgeCapabilities(available=self._available)

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        self.evaluate_calls.append(request)
        if not self._scripted:
            raise RuntimeError(
                "FakeJudge ran out of scripted responses; test setup bug"
            )
        next_response = self._scripted.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


def _cost(usd: float = 0.0005, *, attempts: int = 1) -> JudgeCostEstimate:
    return JudgeCostEstimate(
        provider="anthropic",
        model="test-model",
        input_tokens=100,
        output_tokens=20,
        estimated_usd=usd,
        attempts=attempts,
    )


def _verdict(*, passed: bool, reason: str = "ok", usd: float = 0.0005) -> JudgeVerdict:
    return JudgeVerdict(
        passed=passed,
        reason=reason,
        cost=_cost(usd=usd),
    )


def _new_contract(
    *,
    judge: Any = None,
    judge_optional: bool = False,
    judge_always_run: bool = False,
) -> MemoryContract:
    provider = ReferenceMemoryAdapter()
    run = ContractRun(
        id="run-test",
        contract_id="test::contract",
        provider="reference",
        started_at=datetime.now(tz=UTC),
        status=RunStatus.PASSED,
    )
    return MemoryContract(
        provider,
        run,
        judge=judge,
        judge_optional=judge_optional,
        judge_always_run=judge_always_run,
    )


# -------------------------------------------------- happy path


def test_latest_fact_is_pass_records_cost_and_assert() -> None:
    """Configured judge returns PASS -> ASSERT row with cost_estimate
    populated and ContractRun.judge_cost_usd incremented.

    Also pins the canonical trace ordering documented in
    ``docs/judge-assertions.md``: ``GIVEN_USER → REMEMBER → RECALL →
    ASSERT(judge)``. ``recalllab record`` depends on this ordering to
    regenerate the original call.
    """
    judge = _FakeJudge([_verdict(passed=True, reason="Mumbai is current", usd=0.0008)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai now.")

    contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # One judge call.
    assert len(judge.evaluate_calls) == 1
    request = judge.evaluate_calls[0]
    assert request.mode.value == "latest_fact_is"
    assert request.expected == "Mumbai"
    assert request.model == "test-model"

    # Run accumulator.
    assert contract.run.judge_cost_usd == pytest.approx(0.0008)

    # Canonical event ordering: GIVEN_USER → REMEMBER → RECALL → ASSERT.
    kinds = [e.kind for e in contract.run.events]
    assert kinds == [
        EventKind.GIVEN_USER,
        EventKind.REMEMBER,
        EventKind.RECALL,
        EventKind.ASSERT,
    ]

    # Trace state: one ASSERT row with cost_estimate populated.
    assert_events = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(assert_events) == 1
    judge_event = assert_events[0]
    assert judge_event.payload["mode"] == "latest_fact_is"
    assert judge_event.payload["passed"] is True
    assert judge_event.cost_estimate is not None
    assert judge_event.cost_estimate["estimated_usd"] == pytest.approx(0.0008)
    assert judge_event.cost_estimate["attempts"] == 1
    # raw_responses live on the payload, not inside cost_estimate
    # (docs/judge-assertions.md §Failed-judge ASSERT lifecycle).
    assert "raw_responses" not in judge_event.cost_estimate


def test_latest_fact_is_fail_raises_assertion_and_records_failure() -> None:
    judge = _FakeJudge([_verdict(passed=False, reason="Bangalore still asserted", usd=0.0006)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I still live in Bangalore.")

    with pytest.raises(AssertionError, match="returned FAIL"):
        contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # Cost still billed.
    assert contract.run.judge_cost_usd == pytest.approx(0.0006)
    # Run status flipped to FAILED.
    assert contract.run.status == RunStatus.FAILED


# -------------------------------------------------- combined with rule-based


def test_combined_pass_runs_both_rule_and_judge() -> None:
    judge = _FakeJudge([_verdict(passed=True, usd=0.0002)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    contract.should_recall(
        "Where do I live?",
        contains="Mumbai",
        latest_fact_is="Mumbai",
    )

    assert len(judge.evaluate_calls) == 1  # Judge ran.
    # Two ASSERT events: contains (passed), latest_fact_is (passed).
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[0].payload["mode"] == "contains"
    assert asserts[0].payload["passed"] is True
    assert asserts[0].cost_estimate is None  # Rule-based has no cost.
    assert asserts[1].payload["mode"] == "latest_fact_is"
    assert asserts[1].payload["passed"] is True
    assert asserts[1].cost_estimate is not None


def test_combined_rule_failure_emits_placeholder_and_skips_judge() -> None:
    """Decision #9: failing contains short-circuits BEFORE the judge runs.
    A placeholder ASSERT row (passed=None) is recorded so the trace
    captures the original intent for `recalllab record`."""
    judge = _FakeJudge([_verdict(passed=True)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")

    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",  # Will fail — Mumbai not in recall.
            latest_fact_is="Mumbai",
        )

    # Judge MUST NOT have been called.
    assert len(judge.evaluate_calls) == 0

    # No cost billed.
    assert contract.run.judge_cost_usd == 0.0

    # Two ASSERT rows: failing contains + placeholder for latest_fact_is.
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[0].payload["mode"] == "contains"
    assert asserts[0].payload["passed"] is False  # Real failure.
    assert asserts[1].payload["mode"] == "latest_fact_is"
    assert asserts[1].payload["passed"] is None  # Placeholder.
    assert "short_circuited" in asserts[1].payload["reason"]
    assert asserts[1].cost_estimate is None  # No cost — never ran.

    # Run status: FAILED (rule-based failure). Placeholder does NOT
    # contribute to status.
    assert contract.run.status == RunStatus.FAILED


# -------------------------------------------------- error handling


def test_initial_api_error_propagates_as_judge_unavailable() -> None:
    """JudgeUnavailableError from initial call has no cost — propagate."""
    judge = _FakeJudge([JudgeUnavailableError("Anthropic down")])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    with pytest.raises(JudgeUnavailableError, match="Anthropic down"):
        contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # No cost billed because no call completed.
    assert contract.run.judge_cost_usd == 0.0
    # No judge-mode ASSERT row.
    asserts = [
        e for e in contract.run.events
        if e.kind == EventKind.ASSERT and e.payload.get("mode") == "latest_fact_is"
    ]
    assert len(asserts) == 0


def test_retry_api_error_records_failed_judge_assert_with_partial_cost() -> None:
    """JudgePartialFailureError must land as an ASSERT row carrying the
    realized cost (Codex round-1 step-4 finding). The DSL raises
    AssertionError so pytest reports failure."""
    partial = JudgePartialFailureError(
        "retry API error",
        cost=_cost(usd=0.0003, attempts=2),
        raw_responses=["first malformed reply"],
        underlying_error=RuntimeError("simulated network blip"),
    )
    judge = _FakeJudge([partial])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    with pytest.raises(AssertionError, match="judge_api_error"):
        contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # Cost billed.
    assert contract.run.judge_cost_usd == pytest.approx(0.0003)
    # Failed-judge ASSERT row with cost_estimate AND raw_responses.
    asserts = [
        e for e in contract.run.events
        if e.kind == EventKind.ASSERT and e.payload.get("mode") == "latest_fact_is"
    ]
    assert len(asserts) == 1
    judge_event = asserts[0]
    assert judge_event.payload["passed"] is False
    assert "judge_api_error" in judge_event.payload["reason"]
    assert judge_event.cost_estimate is not None
    assert judge_event.cost_estimate["estimated_usd"] == pytest.approx(0.0003)
    # raw_responses lives on the ASSERT payload, not inside the
    # cost_estimate accounting payload (docs/judge-assertions.md
    # §Failed-judge ASSERT lifecycle).
    assert "raw_responses" not in judge_event.cost_estimate
    assert judge_event.payload["raw_responses"] == ["first malformed reply"]


# -------------------------------------------------- per-run budget cap


def test_per_run_cap_blocks_next_call() -> None:
    """Per-run cap is enforced by the DSL: when ContractRun.judge_cost_usd
    has reached the judge's max_cost_usd, the NEXT call raises before
    invoking the judge."""
    judge = _FakeJudge(
        [
            _verdict(passed=True, usd=0.05),
            _verdict(passed=True, usd=0.05),  # Should never be reached.
        ],
        max_cost_usd=0.05,  # Tight cap.
    )
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    # First call: bills $0.05 (== cap). The next call must refuse.
    contract.should_recall("Where do I live?", latest_fact_is="Mumbai")
    assert contract.run.judge_cost_usd == pytest.approx(0.05)
    assert len(judge.evaluate_calls) == 1

    with pytest.raises(JudgeBudgetExceededError, match="per-run"):
        contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # No additional judge call.
    assert len(judge.evaluate_calls) == 1


# -------------------------------------------------- noop interaction


def test_always_run_invokes_judge_after_rule_failure() -> None:
    """Decision #9 diagnostic mode: ``[judge].always_run = true`` makes
    the judge run even after a rule-based assertion failed. The judge's
    verdict NEVER overrides the rule-based AssertionError for pytest
    reporting, but the cost IS billed and the ASSERT row is real (not
    a placeholder).
    """
    judge = _FakeJudge([_verdict(passed=True, reason="diagnostic ok", usd=0.0004)])
    contract = _new_contract(judge=judge, judge_always_run=True)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")

    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",  # Fails — Mumbai not in recall.
            latest_fact_is="Mumbai",
        )

    # Judge WAS invoked despite the rule-based failure.
    assert len(judge.evaluate_calls) == 1
    # Cost billed.
    assert contract.run.judge_cost_usd == pytest.approx(0.0004)
    # Two ASSERT rows: failing contains + REAL latest_fact_is verdict
    # (not a placeholder).
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[0].payload["passed"] is False  # contains failed.
    assert asserts[1].payload["passed"] is True   # judge passed.
    assert asserts[1].payload["mode"] == "latest_fact_is"
    assert asserts[1].cost_estimate is not None  # Real call, real cost.


def test_always_run_judge_fail_does_not_override_rule_failure() -> None:
    """When the diagnostic judge ALSO returns FAIL, pytest still reports
    the original rule-based failure — the judge verdict only enriches
    the trace, never escalates the failure type."""
    judge = _FakeJudge([_verdict(passed=False, reason="judge also disagrees", usd=0.0003)])
    contract = _new_contract(judge=judge, judge_always_run=True)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")

    with pytest.raises(AssertionError) as exc_info:
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",
            latest_fact_is="Mumbai",
        )

    # The AssertionError that bubbles out is the rule-based one
    # (contains failed), NOT the judge's FAIL verdict.
    assert "contains" in str(exc_info.value)
    # Judge still ran and recorded.
    assert len(judge.evaluate_calls) == 1
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[1].payload["passed"] is False


def test_noop_judge_still_blocks_judge_calls_via_gate() -> None:
    """Regression: the new evaluate path must NOT be reached when the
    judge is NoOpJudge (the fail-loud gate fires first)."""
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    with pytest.raises(JudgeUnavailableError, match="not configured"):
        contract.should_recall("Where do I live?", latest_fact_is="Mumbai")

    # No ASSERT row for the judge mode (gate fired before recall).
    asserts = [
        e for e in contract.run.events
        if e.kind == EventKind.ASSERT and e.payload.get("mode") == "latest_fact_is"
    ]
    assert len(asserts) == 0
