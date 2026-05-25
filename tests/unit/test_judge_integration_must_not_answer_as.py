"""End-to-end integration tests for the must_not_answer_as judge mode.

The dispatch infrastructure that handles must_not_answer_as lives in
``_evaluate_judge_mode`` (shared with latest_fact_is and judge_assertion);
these tests exercise the full DSL path with a single value, a list,
combined rule + judge, short-circuit placeholder, and the per-run cap
to confirm the mode is wired correctly through the prompt builder,
identity tuple, and trace ASSERT shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.judge import (
    JudgeCapabilities,
    JudgeCostEstimate,
    JudgeMode,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
    NoOpJudge,
)
from recalllab.core.traces.schema import ContractRun, EventKind, RunStatus


class _FakeJudge:
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
            raise RuntimeError("FakeJudge ran out of scripted responses")
        nxt = self._scripted.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _verdict(*, passed: bool, reason: str = "ok", usd: float = 0.0005) -> JudgeVerdict:
    return JudgeVerdict(
        passed=passed,
        reason=reason,
        cost=JudgeCostEstimate(
            provider="anthropic",
            model="test-model",
            input_tokens=100,
            output_tokens=20,
            estimated_usd=usd,
            attempts=1,
        ),
    )


def _new_contract(
    *,
    judge: Any = None,
    judge_optional: bool = False,
) -> MemoryContract:
    provider = ReferenceMemoryAdapter()
    run = ContractRun(
        id="run-test",
        contract_id="test::contract",
        provider="reference",
        started_at=datetime.now(tz=UTC),
        status=RunStatus.PASSED,
    )
    return MemoryContract(provider, run, judge=judge, judge_optional=judge_optional)


# -------------------------------------------------- happy path


def test_must_not_answer_as_pass_records_cost_and_assert() -> None:
    """Configured judge returns PASS (none of the forbidden values are
    asserted as current) -> ASSERT row with cost_estimate + run total
    incremented."""
    judge = _FakeJudge(
        [_verdict(passed=True, reason="agent presents Mumbai only", usd=0.0006)]
    )
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai now.")

    contract.should_recall(
        "Where do I live?",
        must_not_answer_as=["Bangalore", "Delhi"],
    )

    assert len(judge.evaluate_calls) == 1
    request = judge.evaluate_calls[0]
    # Mode + expected (list) are correctly propagated to the judge
    # identity tuple.
    assert request.mode == JudgeMode.MUST_NOT_ANSWER_AS
    assert request.expected == ["Bangalore", "Delhi"]
    # The judge's rubric is the one for must_not_answer_as (sanity
    # check the dispatch went through the right branch).
    assert request.rubric is None

    # Run accumulator.
    assert contract.run.judge_cost_usd == pytest.approx(0.0006)

    # ASSERT row records the literal list as expected (trace identity =
    # full kwarg value).
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 1
    judge_event = asserts[0]
    assert judge_event.payload["mode"] == "must_not_answer_as"
    assert judge_event.payload["passed"] is True
    assert judge_event.payload["expected"] == ["Bangalore", "Delhi"]
    assert judge_event.cost_estimate is not None
    assert judge_event.cost_estimate["estimated_usd"] == pytest.approx(0.0006)


def test_must_not_answer_as_fail_raises_assertion() -> None:
    """If the judge says FAIL (a forbidden value IS asserted as
    current), the contract fails and the cost is still billed."""
    judge = _FakeJudge(
        [_verdict(passed=False, reason="agent asserted Bangalore as current", usd=0.0007)]
    )
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I still live in Bangalore.")

    with pytest.raises(AssertionError, match="returned FAIL"):
        contract.should_recall(
            "Where do I live?",
            must_not_answer_as=["Bangalore"],
        )

    assert contract.run.judge_cost_usd == pytest.approx(0.0007)
    assert contract.run.status == RunStatus.FAILED


def test_must_not_answer_as_duplicates_are_deduped_order_preserving() -> None:
    """Codex round-1 step-6/7 finding #6: ``["X", "X", "Y"]`` must
    produce the same prompt-identity tuple as ``["X", "Y"]`` so the
    v0.2.3 cache key doesn't fragment on caller sloppiness. The dedupe
    preserves first-seen order so the trace remains user-readable."""
    judge = _FakeJudge([_verdict(passed=True)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    contract.should_recall(
        "Where do I live?",
        must_not_answer_as=["Bangalore", "Bangalore", "Delhi", "Bangalore"],
    )

    request = judge.evaluate_calls[0]
    # Deduped + order preserved.
    assert request.expected == ["Bangalore", "Delhi"]
    # The ASSERT row's `expected` also stores the deduped list so the
    # trace identity matches the prompt identity.
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert asserts[0].payload["expected"] == ["Bangalore", "Delhi"]


def test_must_not_answer_as_single_string_list() -> None:
    """A list with one entry works exactly like a multi-element list —
    confirms the kwarg type matches its annotation everywhere."""
    judge = _FakeJudge([_verdict(passed=True, usd=0.0004)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    contract.should_recall(
        "Where do I live?",
        must_not_answer_as=["Bangalore"],
    )

    request = judge.evaluate_calls[0]
    assert request.expected == ["Bangalore"]


# -------------------------------------------------- combined with rule-based


def test_combined_excludes_and_must_not_answer_as() -> None:
    """Rule-based excludes + judge must_not_answer_as in the same call.
    Decision #9: rule-based runs first; judge runs only if it passed."""
    judge = _FakeJudge(
        [_verdict(passed=True, reason="judge agrees", usd=0.0003)]
    )
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    # excludes="Bangalore" passes (Bangalore isn't in the recall); then
    # the judge runs.
    contract.should_recall(
        "Where do I live?",
        excludes="Bangalore",
        must_not_answer_as=["Delhi"],
    )

    assert len(judge.evaluate_calls) == 1
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[0].payload["mode"] == "excludes"
    assert asserts[0].payload["passed"] is True
    assert asserts[1].payload["mode"] == "must_not_answer_as"
    assert asserts[1].payload["passed"] is True


def test_combined_failure_short_circuits_with_placeholder() -> None:
    """Decision #9: failing rule-based assertion never spends judge
    cost; the must_not_answer_as kwarg lands as a placeholder ASSERT."""
    judge = _FakeJudge([_verdict(passed=True)])  # Should never be called.
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I still live in Bangalore.")

    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            excludes="Bangalore",  # Fails — Bangalore IS in the recall.
            must_not_answer_as=["Delhi"],
        )

    # Judge never invoked, no cost.
    assert len(judge.evaluate_calls) == 0
    assert contract.run.judge_cost_usd == 0.0

    # Placeholder ASSERT.
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[1].payload["mode"] == "must_not_answer_as"
    assert asserts[1].payload["passed"] is None
    assert asserts[1].payload["expected"] == ["Delhi"]
    assert "short_circuited" in asserts[1].payload["reason"]
    assert asserts[1].cost_estimate is None


# -------------------------------------------------- noop gate


def test_must_not_answer_as_with_noop_judge_raises_unavailable() -> None:
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    with pytest.raises(JudgeUnavailableError, match="not configured"):
        contract.should_recall(
            "Where do I live?",
            must_not_answer_as=["Bangalore"],
        )


def test_must_not_answer_as_with_noop_judge_skips_when_optional() -> None:
    contract = _new_contract(judge=NoOpJudge(), judge_optional=True)
    contract.given_user("ayush")
    contract.remember("I live in Mumbai.")

    with pytest.raises(pytest.skip.Exception):
        contract.should_recall(
            "Where do I live?",
            must_not_answer_as=["Bangalore"],
        )
