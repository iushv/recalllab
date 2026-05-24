"""End-to-end integration tests for the judge_assertion (Rubric) judge mode.

Covers:

- Happy-path PASS / FAIL with a Rubric.
- Rubric labels stay local to the trace (criterion is what enters the
  prompt envelope's rubric field).
- JudgeRequest.expected is None for this mode (per Codex round-1
  step-5 finding #4) — no sentinel string.
- Rubric.criterion / pass_label / fail_label length bounds from the
  Codex round-3 deferred finding #7.
- Combined rule + judge_assertion (Decision #9 dispatch through the
  shared _evaluate_judge_mode).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

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
    Rubric,
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


def _new_contract(*, judge: Any = None, judge_optional: bool = False) -> MemoryContract:
    provider = ReferenceMemoryAdapter()
    run = ContractRun(
        id="run-test",
        contract_id="test::contract",
        provider="reference",
        started_at=datetime.now(tz=UTC),
        status=RunStatus.PASSED,
    )
    return MemoryContract(provider, run, judge=judge, judge_optional=judge_optional)


# -------------------------------------------------- Rubric bounds


def test_rubric_criterion_min_length_one() -> None:
    """An empty criterion is meaningless and rejected at construction."""
    with pytest.raises(ValidationError):
        Rubric(criterion="")


def test_rubric_criterion_max_length_4096() -> None:
    """Bounded so a runaway rubric can't blow through the prompt budget."""
    # Length 4096 is fine; length 4097 fails.
    Rubric(criterion="x" * 4096)
    with pytest.raises(ValidationError):
        Rubric(criterion="x" * 4097)


def test_rubric_label_bounds() -> None:
    """Labels are kept short for trace-display readability."""
    Rubric(criterion="anything", pass_label="x" * 32, fail_label="y" * 32)
    with pytest.raises(ValidationError):
        Rubric(criterion="anything", pass_label="x" * 33)
    with pytest.raises(ValidationError):
        Rubric(criterion="anything", fail_label="y" * 33)
    with pytest.raises(ValidationError):
        Rubric(criterion="anything", pass_label="")
    with pytest.raises(ValidationError):
        Rubric(criterion="anything", fail_label="")


# -------------------------------------------------- happy path


def test_judge_assertion_pass_records_cost_and_assert() -> None:
    """Judge returns PASS for the supplied Rubric. The ASSERT row's
    expected field stores the full Rubric (criterion + both labels)
    so the trace is human-readable; only criterion enters the prompt
    envelope; JudgeRequest.expected is None for this mode."""
    rubric = Rubric(
        criterion="The response must cite the source episode.",
        pass_label="CITED",
        fail_label="UNCITED",
    )
    judge = _FakeJudge([_verdict(passed=True, reason="source cited", usd=0.0009)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("Episode-42: I bought a new phone yesterday.")

    contract.should_recall("What did I buy?", judge_assertion=rubric)

    assert len(judge.evaluate_calls) == 1
    request = judge.evaluate_calls[0]
    assert request.mode == JudgeMode.JUDGE_ASSERTION
    # No expected literal for this mode.
    assert request.expected is None
    # The Rubric's criterion is what enters the prompt envelope's rubric field.
    assert request.rubric == "The response must cite the source episode."

    # Trace identity: the ASSERT row's expected field stores the
    # Rubric.model_dump() (dict, not the live instance) — labels
    # included so the trace shows the user's vocabulary. Codex round-1
    # step-6/7 finding #3 — normalizing to dict at record-time means
    # the SQLite round-trip and trace-to-test emitter consume one
    # canonical shape.
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 1
    judge_event = asserts[0]
    assert judge_event.payload["mode"] == "judge_assertion"
    assert judge_event.payload["passed"] is True
    stored_expected = judge_event.payload["expected"]
    assert isinstance(stored_expected, dict), (
        "judge_assertion expected must be normalized to dict at "
        "trace-record time, not stored as a live Rubric instance"
    )
    assert stored_expected == {
        "criterion": rubric.criterion,
        "pass_label": "CITED",
        "fail_label": "UNCITED",
    }

    assert contract.run.judge_cost_usd == pytest.approx(0.0009)


def test_judge_assertion_fail_raises_assertion() -> None:
    rubric = Rubric(criterion="must cite source")
    judge = _FakeJudge(
        [_verdict(passed=False, reason="no citation in response", usd=0.0005)]
    )
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I bought a phone.")  # No episode citation.

    with pytest.raises(AssertionError, match="returned FAIL"):
        contract.should_recall("What did I buy?", judge_assertion=rubric)

    assert contract.run.judge_cost_usd == pytest.approx(0.0005)
    assert contract.run.status == RunStatus.FAILED


def test_rubric_label_only_change_does_not_change_prompt_identity() -> None:
    """Per Rubric Identity (docs/judge-assertions.md §Rubric class):
    only criterion enters the prompt envelope. Two Rubrics with the
    same criterion but different labels produce the SAME judge prompt
    (system + user message both byte-identical) so the v0.2.3 cache
    key will be identical AND no label text leaks into the prompt.
    """
    from recalllab.core.judge.prompts import build_judge_prompt

    rubric_a = Rubric(criterion="must cite source", pass_label="A_PASS", fail_label="A_FAIL")
    rubric_b = Rubric(criterion="must cite source", pass_label="B_PASS", fail_label="B_FAIL")
    judge_a = _FakeJudge([_verdict(passed=True)])
    judge_b = _FakeJudge([_verdict(passed=True)])

    contract_a = _new_contract(judge=judge_a)
    contract_a.given_user("ayush")
    contract_a.remember("Episode-7: I baked a cake.")
    contract_a.should_recall("What did I make?", judge_assertion=rubric_a)

    contract_b = _new_contract(judge=judge_b)
    contract_b.given_user("ayush")
    contract_b.remember("Episode-7: I baked a cake.")
    contract_b.should_recall("What did I make?", judge_assertion=rubric_b)

    # The JudgeRequest values for the two calls are identical
    # (criterion-only identity).
    req_a = judge_a.evaluate_calls[0]
    req_b = judge_b.evaluate_calls[0]
    assert req_a.rubric == req_b.rubric == "must cite source"
    # And the FULL prompt built from each request is byte-identical —
    # no label text leaks into system or user message.
    sys_a, user_a = build_judge_prompt(req_a)
    sys_b, user_b = build_judge_prompt(req_b)
    assert sys_a == sys_b
    assert user_a == user_b
    for label in ("A_PASS", "A_FAIL", "B_PASS", "B_FAIL"):
        assert label not in sys_a, (
            f"label {label!r} leaked into system prompt; only criterion "
            f"should affect the prompt"
        )
        assert label not in user_a, (
            f"label {label!r} leaked into user message; only criterion "
            f"should affect the prompt"
        )


def test_rubric_labels_appear_in_failure_message() -> None:
    """Codex round-1 step-6/7 finding #2: pass_label / fail_label are
    documented as affecting user-visible reasons. Wire them into the
    AssertionError so the message uses the user's vocabulary."""
    rubric = Rubric(
        criterion="must cite source",
        pass_label="CITED",
        fail_label="UNCITED",
    )
    judge = _FakeJudge([_verdict(passed=False, reason="no citation found")])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I bought a phone.")

    with pytest.raises(AssertionError, match="UNCITED") as exc_info:
        contract.should_recall("What did I buy?", judge_assertion=rubric)
    # Generic "FAIL" must NOT appear when a custom label is configured.
    assert "returned FAIL" not in str(exc_info.value)


# -------------------------------------------------- combined with rule-based


def test_combined_contains_and_judge_assertion() -> None:
    """Decision #9: rule-based runs first; judge_assertion runs only
    if it passed."""
    rubric = Rubric(criterion="must mention Mumbai")
    judge = _FakeJudge([_verdict(passed=True, reason="Mumbai cited", usd=0.0004)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I moved to Mumbai last month.")

    contract.should_recall(
        "Where did I move?",
        contains="Mumbai",
        judge_assertion=rubric,
    )

    assert len(judge.evaluate_calls) == 1
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    assert asserts[0].payload["mode"] == "contains"
    assert asserts[0].payload["passed"] is True
    assert asserts[1].payload["mode"] == "judge_assertion"
    assert asserts[1].payload["passed"] is True


def test_combined_rule_failure_emits_placeholder_for_judge_assertion() -> None:
    rubric = Rubric(criterion="must mention Mumbai")
    judge = _FakeJudge([_verdict(passed=True)])
    contract = _new_contract(judge=judge)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")

    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",  # Fails.
            judge_assertion=rubric,
        )

    assert len(judge.evaluate_calls) == 0
    assert contract.run.judge_cost_usd == 0.0
    asserts = [e for e in contract.run.events if e.kind == EventKind.ASSERT]
    assert len(asserts) == 2
    placeholder = asserts[1]
    assert placeholder.payload["mode"] == "judge_assertion"
    assert placeholder.payload["passed"] is None
    # Placeholder records the Rubric.model_dump() on expected so
    # `recalllab record` (step 8) can regenerate the literal from a
    # single canonical shape.
    stored = placeholder.payload["expected"]
    assert isinstance(stored, dict)
    assert stored == {
        "criterion": "must mention Mumbai",
        "pass_label": "PASS",
        "fail_label": "FAIL",
    }


# -------------------------------------------------- noop gate


def test_judge_assertion_with_noop_judge_raises_unavailable() -> None:
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    contract.remember("anything")

    with pytest.raises(JudgeUnavailableError, match="not configured"):
        contract.should_recall(
            "any query",
            judge_assertion=Rubric(criterion="anything"),
        )


def test_judge_assertion_with_noop_judge_skips_when_optional() -> None:
    contract = _new_contract(judge=NoOpJudge(), judge_optional=True)
    contract.given_user("ayush")
    contract.remember("anything")

    with pytest.raises(pytest.skip.Exception):
        contract.should_recall(
            "any query",
            judge_assertion=Rubric(criterion="anything"),
        )
