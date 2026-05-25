"""Tests for the trace-to-test emitter's judge-mode rendering (step 8).

Covers:

- ``latest_fact_is`` / ``must_not_answer_as`` / ``judge_assertion``
  modes rendered as ``should_recall`` kwargs.
- ``Rubric(...)`` literal rendering with default-value omission.
- Decision #9 short-circuit placeholders still produce the kwarg in
  the regenerated call (regression fidelity).
- ``--optional-judge`` CLI flag adds the
  ``recalllab_optional("judge_configured")`` decorator; default off.
- Generated source includes ``import pytest`` and ``from recalllab
  import Rubric`` only when the test actually needs them.
- Combined rule + judge call regenerates both kwargs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from recalllab.cli.record import (
    _RUBRIC_DEFAULTS,
    _emit_rubric_literal,
    _is_renderable_rubric,
    trace_to_test_source,
)
from recalllab.core.judge import Rubric
from recalllab.core.traces.schema import (
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)


def _ts() -> datetime:
    return datetime(2025, 1, 1, tzinfo=UTC)


def _new_run(events: list[TraceEvent]) -> ContractRun:
    return ContractRun(
        id="run-test",
        contract_id="tests/test_x.py::test_recorded",
        provider="reference",
        started_at=_ts(),
        finished_at=_ts(),
        status=RunStatus.FAILED,
        events=events,
        assertions=[],
    )


# ---------------------------------------------------- Rubric literal


def test_emit_rubric_literal_with_defaults_omits_labels() -> None:
    """Default labels (PASS/FAIL) are not emitted — keeps generated
    files readable for the common case."""
    rendered = _emit_rubric_literal(
        {"criterion": "must cite source", "pass_label": "PASS", "fail_label": "FAIL"}
    )
    assert rendered == "Rubric(criterion='must cite source')"


def test_emit_rubric_literal_with_custom_labels_emits_both() -> None:
    rendered = _emit_rubric_literal(
        {
            "criterion": "must cite source",
            "pass_label": "CITED",
            "fail_label": "UNCITED",
        }
    )
    assert "criterion='must cite source'" in rendered
    assert "pass_label='CITED'" in rendered
    assert "fail_label='UNCITED'" in rendered


def test_emit_rubric_literal_with_partial_defaults_emits_only_overridden() -> None:
    rendered = _emit_rubric_literal(
        {"criterion": "must cite source", "pass_label": "CITED", "fail_label": "FAIL"}
    )
    assert "pass_label='CITED'" in rendered
    assert "fail_label" not in rendered  # Default omitted.


def test_is_renderable_rubric_requires_criterion() -> None:
    """Gate: emitter falls back to a comment + dropped kwarg when the
    stored rubric shape is corrupt (missing criterion). Codex round-1
    step-8 finding #1."""
    assert _is_renderable_rubric(
        {"criterion": "must cite", "pass_label": "P", "fail_label": "F"}
    )
    assert not _is_renderable_rubric({})
    assert not _is_renderable_rubric({"pass_label": "P"})
    assert not _is_renderable_rubric({"criterion": ""})
    assert not _is_renderable_rubric({"criterion": 123})  # wrong type
    assert not _is_renderable_rubric("not a dict")


def test_rubric_defaults_match_model_defaults() -> None:
    """Codex round-1 step-8 finding #4: _RUBRIC_DEFAULTS is hard-coded
    in record.py for dependency-light trace-walking. Pin it against
    Rubric.model_fields so a future default change in the model fails
    this test loudly rather than silently producing redundant kwargs
    in every regenerated regression file."""
    actual = {
        name: Rubric.model_fields[name].default
        for name in _RUBRIC_DEFAULTS
    }
    assert actual == _RUBRIC_DEFAULTS, (
        f"_RUBRIC_DEFAULTS drift from Rubric.model_fields: emitter has "
        f"{_RUBRIC_DEFAULTS}, model has {actual}"
    )


# ---------------------------------------------------- mode rendering


def _build_recall_assert(
    mode: str,
    expected: object,
    *,
    passed: bool | None = True,
    reason: str | None = None,
) -> list[TraceEvent]:
    return [
        TraceEvent(
            sequence=0,
            kind=EventKind.GIVEN_USER,
            payload={"user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=1,
            kind=EventKind.RECALL,
            payload={"query": "Where do I live?", "k": 5, "user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=2,
            kind=EventKind.ASSERT,
            payload={
                "mode": mode,
                "expected": expected,
                "passed": passed,
                "reason": reason,
            },
            timestamp=_ts(),
        ),
    ]


def test_latest_fact_is_rendered_as_kwarg() -> None:
    run = _new_run(_build_recall_assert("latest_fact_is", "Mumbai"))
    source = trace_to_test_source(run)
    assert "latest_fact_is='Mumbai'" in source
    # Whole call appears.
    assert "memory_contract.should_recall('Where do I live?', latest_fact_is='Mumbai')" in source


def test_must_not_answer_as_rendered_as_kwarg() -> None:
    run = _new_run(_build_recall_assert("must_not_answer_as", ["Bangalore", "Delhi"]))
    source = trace_to_test_source(run)
    assert "must_not_answer_as=['Bangalore', 'Delhi']" in source


def test_judge_assertion_rendered_as_rubric_literal() -> None:
    run = _new_run(
        _build_recall_assert(
            "judge_assertion",
            {
                "criterion": "must cite source",
                "pass_label": "PASS",
                "fail_label": "FAIL",
            },
        )
    )
    source = trace_to_test_source(run)
    assert "judge_assertion=Rubric(criterion='must cite source')" in source
    # Rubric import is present.
    assert "from recalllab import Rubric" in source


def test_corrupt_judge_assertion_payload_drops_kwarg_and_falls_back_to_recall() -> None:
    """Codex round-1 step-8 finding #1 + round-2 confirming finding:
    when the stored Rubric payload lacks `criterion`, drop the judge
    kwarg, emit a comment, AND fall through to plain ``recall(...)``
    when no other assertion kwargs remain. Emitting
    ``should_recall(query)`` with no kwargs would crash at replay
    because the DSL rejects it with ValueError."""
    run = _new_run(
        _build_recall_assert(
            "judge_assertion",
            {"pass_label": "P", "fail_label": "F"},  # No criterion!
        )
    )
    source = trace_to_test_source(run)
    compile(source, "<emitted>", "exec")
    # The judge_assertion kwarg must NOT appear in the regenerated call.
    assert "judge_assertion=" not in source
    # And a comment must explain the drop.
    assert "corrupt Rubric payload" in source
    # AND the regenerated call must be plain recall(...), NOT
    # should_recall(query) with no kwargs (which would crash at replay).
    assert "memory_contract.recall(" in source
    assert "memory_contract.should_recall(" not in source


def test_corrupt_judge_assertion_with_other_kwargs_keeps_should_recall() -> None:
    """When the trace mixes a corrupt judge_assertion with a working
    rule-based assertion, drop the corrupt judge kwarg only and KEEP
    should_recall with the surviving kwarg. The recall-side fallback
    only fires when EVERY assertion kwarg got dropped."""
    events = [
        TraceEvent(
            sequence=0,
            kind=EventKind.GIVEN_USER,
            payload={"user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=1,
            kind=EventKind.RECALL,
            payload={"query": "Where?", "k": 5, "user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=2,
            kind=EventKind.ASSERT,
            payload={"mode": "contains", "expected": "Mumbai", "passed": True},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=3,
            kind=EventKind.ASSERT,
            payload={
                "mode": "judge_assertion",
                "expected": {"pass_label": "P", "fail_label": "F"},  # corrupt
                "passed": True,
            },
            timestamp=_ts(),
        ),
    ]
    source = trace_to_test_source(_new_run(events))
    compile(source, "<emitted>", "exec")
    # should_recall is preserved (contains kwarg still applies).
    assert "memory_contract.should_recall('Where?', contains='Mumbai')" in source
    # Corrupt judge_assertion is dropped with a comment.
    assert "corrupt Rubric payload" in source
    # And we did NOT fall through to plain recall. The only ``.recall(``
    # appearance in the source should be inside ``should_recall(``.
    assert source.count("memory_contract.recall(") == 0


def test_judge_assertion_with_custom_labels_renders_full_literal() -> None:
    run = _new_run(
        _build_recall_assert(
            "judge_assertion",
            {
                "criterion": "must cite source",
                "pass_label": "CITED",
                "fail_label": "UNCITED",
            },
        )
    )
    source = trace_to_test_source(run)
    assert "pass_label='CITED'" in source
    assert "fail_label='UNCITED'" in source


# ---------------------------------------------------- combined rule + judge


def test_combined_contains_and_latest_fact_is() -> None:
    """Combined rule + judge call: both kwargs appear in one
    should_recall."""
    events = [
        TraceEvent(
            sequence=0,
            kind=EventKind.GIVEN_USER,
            payload={"user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=1,
            kind=EventKind.RECALL,
            payload={"query": "Where?", "k": 5, "user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=2,
            kind=EventKind.ASSERT,
            payload={"mode": "contains", "expected": "Mumbai", "passed": True},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=3,
            kind=EventKind.ASSERT,
            payload={"mode": "latest_fact_is", "expected": "Mumbai", "passed": True},
            timestamp=_ts(),
        ),
    ]
    source = trace_to_test_source(_new_run(events))
    # Both kwargs appear in the same should_recall call, in canonical order.
    assert (
        "memory_contract.should_recall('Where?', contains='Mumbai', "
        "latest_fact_is='Mumbai')"
    ) in source


def test_short_circuit_placeholder_still_regenerates_kwarg() -> None:
    """Decision #9 regression fidelity: a placeholder ASSERT (passed=None,
    judge never ran) still produces the judge kwarg in the regenerated
    call, so re-running the regression hits the same short-circuit."""
    events = [
        TraceEvent(
            sequence=0,
            kind=EventKind.GIVEN_USER,
            payload={"user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=1,
            kind=EventKind.RECALL,
            payload={"query": "Where?", "k": 5, "user_id": "ayush"},
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=2,
            kind=EventKind.ASSERT,
            payload={
                "mode": "contains",
                "expected": "Mumbai",
                "passed": False,
                "reason": "Mumbai not found",
            },
            timestamp=_ts(),
        ),
        TraceEvent(
            sequence=3,
            kind=EventKind.ASSERT,
            payload={
                "mode": "latest_fact_is",
                "expected": "Mumbai",
                "passed": None,
                "reason": "short_circuited: preceding rule-based assertion failed",
            },
            timestamp=_ts(),
        ),
    ]
    source = trace_to_test_source(_new_run(events))
    # The judge kwarg IS present even though the placeholder shows
    # passed=None.
    assert "latest_fact_is='Mumbai'" in source
    assert "contains='Mumbai'" in source
    # The failed rule-based assertion's reason is rendered as a
    # documenting comment.
    assert "original assertion failed (contains)" in source
    # The placeholder's reason text is NOT rendered as "failed" because
    # passed=None is not a failure.
    assert "original assertion failed (latest_fact_is)" not in source
    # But the placeholder IS surfaced as a short-circuited comment so
    # the reader knows the judge didn't run at record time (Codex
    # round-1 step-8 finding #3).
    assert "original assertion short-circuited (latest_fact_is)" in source


# ---------------------------------------------------- optional-judge flag


def test_optional_judge_off_by_default_no_marker() -> None:
    """Decision #3b: default-off so generated regressions ERROR loudly
    when judge isn't configured in CI."""
    run = _new_run(_build_recall_assert("latest_fact_is", "Mumbai"))
    source = trace_to_test_source(run)  # default optional_judge=False
    assert 'recalllab_optional("judge_configured")' not in source


def test_optional_judge_on_emits_marker() -> None:
    run = _new_run(_build_recall_assert("latest_fact_is", "Mumbai"))
    source = trace_to_test_source(run, optional_judge=True)
    assert '@pytest.mark.recalllab_optional("judge_configured")' in source
    # And the import resolves.
    assert "import pytest" in source


def test_optional_judge_on_but_no_judge_mode_does_not_emit_marker() -> None:
    """When the trace has only rule-based assertions, --optional-judge
    is a no-op (no judge marker to add)."""
    run = _new_run(_build_recall_assert("contains", "Mumbai"))
    source = trace_to_test_source(run, optional_judge=True)
    assert 'recalllab_optional("judge_configured")' not in source


# ---------------------------------------------------- import shape


def test_orphan_judge_assert_does_not_trigger_judge_imports() -> None:
    """Codex round-1 step-8 finding #2: a judge-mode ASSERT not paired
    with a RECALL is rendered only as a comment. _trace_uses_judge /
    _trace_uses_rubric must therefore NOT count it — otherwise the
    generated test gets a spurious import or marker even though it
    never invokes the judge."""
    events = [
        TraceEvent(
            sequence=0,
            kind=EventKind.GIVEN_USER,
            payload={"user_id": "ayush"},
            timestamp=_ts(),
        ),
        # Orphan ASSERT — no preceding RECALL.
        TraceEvent(
            sequence=1,
            kind=EventKind.ASSERT,
            payload={
                "mode": "judge_assertion",
                "expected": {"criterion": "x", "pass_label": "PASS", "fail_label": "FAIL"},
                "passed": True,
            },
            timestamp=_ts(),
        ),
    ]
    source = trace_to_test_source(_new_run(events), optional_judge=True)
    # Orphan-assert comment IS rendered (audit trail).
    assert "orphan assert" in source
    # But no Rubric import — the generated test never calls Rubric().
    assert "from recalllab import Rubric" not in source
    # And no judge_configured marker — the generated test has no
    # judge-mode call to gate.
    assert 'recalllab_optional("judge_configured")' not in source


def test_pytest_not_imported_when_no_markers_apply() -> None:
    """Backward-compatible: rule-based traces with no markers stay
    minimal (no ``import pytest`` in the header)."""
    run = _new_run(_build_recall_assert("contains", "Mumbai"))
    source = trace_to_test_source(run)
    assert "import pytest" not in source


def test_rubric_import_only_when_judge_assertion_used() -> None:
    """The ``from recalllab import Rubric`` line appears only when the
    test actually uses Rubric — not for other judge modes."""
    run_latest = _new_run(_build_recall_assert("latest_fact_is", "Mumbai"))
    assert "from recalllab import Rubric" not in trace_to_test_source(run_latest)
    run_must_not = _new_run(
        _build_recall_assert("must_not_answer_as", ["Bangalore"])
    )
    assert "from recalllab import Rubric" not in trace_to_test_source(run_must_not)
    run_judge = _new_run(
        _build_recall_assert(
            "judge_assertion",
            {"criterion": "must cite", "pass_label": "PASS", "fail_label": "FAIL"},
        )
    )
    assert "from recalllab import Rubric" in trace_to_test_source(run_judge)


# ---------------------------------------------------- byte-stability


def test_same_run_produces_byte_identical_output() -> None:
    """Pure-function guarantee: same input → same bytes."""
    run = _new_run(_build_recall_assert("latest_fact_is", "Mumbai"))
    a = trace_to_test_source(run)
    b = trace_to_test_source(run)
    assert a == b
    # And the optional-judge variant is stable too.
    c = trace_to_test_source(run, optional_judge=True)
    d = trace_to_test_source(run, optional_judge=True)
    assert c == d
    # And the two variants differ exactly by the marker line.
    assert c != a
