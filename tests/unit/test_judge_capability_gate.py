"""Tests for the v0.2.2 step 2 judge capability gate and plumbing.

Covers:

- Decision #3a (one judge mode per call) — DSL-level validation.
- Decision #3b (fail-loud default) — DSL gate behavior with NoOpJudge.
- Decision #9 (rule-based first) — short-circuit semantics (the step-2
  gate currently raises ``NotImplementedError`` after a passing
  rule-based call when a judge mode is also present; step 4 replaces
  that with real evaluation).
- Multi-marker iter_markers — pytest plugin honors every declared gate.
- Collection-time validation — unknown capability names raise
  ``pytest.UsageError`` before any test runs.
- Judge-mode kwargs against an unconfigured judge: ERROR by default,
  SKIPPED with the ``recalllab_optional("judge_configured")`` marker.

The DSL-level tests exercise ``MemoryContract`` directly. The pytest-
plugin-level tests run an inner pytest in a subprocess (the same
pattern ``test_codex_review_regressions.py`` uses) so the plugin
collection / fixture behavior is checked end-to-end.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.judge import (
    JudgeCapabilities,
    JudgeProvider,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
    NoOpJudge,
    Rubric,
)
from recalllab.core.traces.schema import ContractRun, RunStatus


def _new_contract(
    *,
    judge: JudgeProvider | None = None,
    judge_optional: bool = False,
) -> MemoryContract:
    """Build a MemoryContract bound to an empty reference adapter."""
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
    )


# ---------------------------------------------------------------- DSL-level


def test_single_judge_mode_per_call_enforced() -> None:
    """Decision #3a: combining two judge-mode kwargs raises ValueError."""
    contract = _new_contract().given_user("ayush")
    with pytest.raises(ValueError, match="only one judge-mode kwarg"):
        contract.should_recall(
            "any query",
            latest_fact_is="X",
            must_not_answer_as=["Y"],
        )


def test_single_judge_mode_per_call_with_judge_assertion() -> None:
    """All three judge modes are mutually exclusive, not just two of them."""
    contract = _new_contract().given_user("ayush")
    with pytest.raises(ValueError, match="only one judge-mode kwarg"):
        contract.should_recall(
            "any query",
            latest_fact_is="X",
            judge_assertion=Rubric(criterion="any"),
        )


def test_no_assertion_kwarg_raises() -> None:
    """Calling should_recall with no contains/excludes/judge kwarg is an error."""
    contract = _new_contract().given_user("ayush")
    with pytest.raises(ValueError, match="needs at least one of"):
        contract.should_recall("any query")


def test_judge_mode_without_judge_raises_fail_loud_by_default() -> None:
    """Decision #3b: NoOpJudge + judge_optional=False → JudgeUnavailableError."""
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    with pytest.raises(JudgeUnavailableError, match="not configured"):
        contract.should_recall("any query", latest_fact_is="Mumbai")


def test_judge_mode_without_judge_skips_when_marker_present() -> None:
    """Decision #3b: NoOpJudge + judge_optional=True → pytest.skip."""
    contract = _new_contract(judge=NoOpJudge(), judge_optional=True)
    contract.given_user("ayush")
    with pytest.raises(pytest.skip.Exception):
        contract.should_recall("any query", latest_fact_is="Mumbai")


def test_judge_assertion_kwarg_also_triggers_gate() -> None:
    """The fail-loud gate fires for all three judge-mode kwargs, not just latest_fact_is."""
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    with pytest.raises(JudgeUnavailableError):
        contract.should_recall(
            "any query",
            judge_assertion=Rubric(criterion="must cite source"),
        )
    contract2 = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract2.given_user("ayush")
    with pytest.raises(JudgeUnavailableError):
        contract2.should_recall(
            "any query",
            must_not_answer_as=["X"],
        )


def test_rule_based_only_calls_still_work_with_no_judge_provided() -> None:
    """Backward compatibility: contracts with no judge kwarg should never touch the judge.

    A MemoryContract built without explicit judge=... gets a NoOpJudge by
    default. Pure rule-based calls must never reach the gate; the gate
    only fires on judge-mode kwargs.
    """
    contract = _new_contract()
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")
    # contains= is satisfied by the in-memory recall; this should pass
    # without ever consulting the judge.
    contract.should_recall("Where do I live?", contains="Bangalore")


def test_combined_rule_judge_short_circuits_against_default_noop() -> None:
    """Decision #9 must hold against the DEFAULT NoOpJudge, not just a fake.

    Codex round-3 adversarial finding (HIGH): the original step-2
    implementation placed the fail-loud gate BEFORE rule-based
    assertions, so ``should_recall(query, contains="missing",
    latest_fact_is="X")`` against the default NoOpJudge raised
    ``JudgeUnavailableError`` instead of the cheap ``AssertionError``
    the user expected. That contradicted Decision #9 (rule-first
    short-circuit) because the fail-loud gate fired even though the
    judge would never have been invoked.

    The current implementation runs rule-based assertions first; the
    gate only fires if all rule-based pass. This test pins that
    ordering against the default ``NoOpJudge`` so the regression cannot
    return.
    """
    contract = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")
    # contains="Mumbai" will fail — Mumbai is nowhere in the recall.
    # The gate should NOT fire because the rule-based assertion fails
    # first; the user must see AssertionError, not JudgeUnavailableError.
    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",
            latest_fact_is="Mumbai",
        )
    # And specifically NOT a JudgeUnavailableError — pytest.raises would
    # accept a subclass, so guard explicitly.
    contract2 = _new_contract(judge=NoOpJudge(), judge_optional=False)
    contract2.given_user("ayush")
    contract2.remember("I live in Bangalore.")
    try:
        contract2.should_recall(
            "Where do I live?",
            contains="Mumbai",
            latest_fact_is="Mumbai",
        )
    except JudgeUnavailableError:  # pragma: no cover — regression guard
        pytest.fail(
            "fail-loud gate ran before rule-based assertion; Decision #9 "
            "violation: combined call should have raised AssertionError "
            "from the failing contains= check"
        )
    except AssertionError:
        pass  # Expected.


def test_combined_rule_judge_short_circuits_on_rule_failure() -> None:
    """Decision #9: failing rule-based assertion never spends judge cost.

    The gate is configured with a NoOpJudge that would raise on any
    evaluate() call. If the rule-based assertion is evaluated FIRST and
    fails fast (Decision #9), the judge gate must never fire. This is
    the budget-protection property the design promises.
    """

    class TrackingJudge(NoOpJudge):
        """A NoOpJudge that records whether evaluate() was ever called."""

        def __init__(self) -> None:
            self.evaluate_calls = 0

        def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
            self.evaluate_calls += 1
            return super().evaluate(request)

    judge = TrackingJudge()

    # Pretend the judge is configured for this scenario so the gate
    # doesn't short-circuit on availability; we want to confirm
    # Decision #9 short-circuits on the rule-based failure first.
    class AlwaysAvailableNoOp(TrackingJudge):
        def capabilities(self) -> JudgeCapabilities:
            return JudgeCapabilities(available=True)

    judge = AlwaysAvailableNoOp()
    contract = _new_contract(judge=judge, judge_optional=False)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")
    # contains="Mumbai" will fail (the recall returns "Bangalore" only),
    # and Decision #9 says the judge must never be invoked in that case.
    with pytest.raises(AssertionError):
        contract.should_recall(
            "Where do I live?",
            contains="Mumbai",
            latest_fact_is="Mumbai",
        )
    assert judge.evaluate_calls == 0, (
        "Decision #9 violation: judge.evaluate() was called even though "
        "the rule-based assertion failed first."
    )


# ---------------------------------------------------------------- Plugin-level


def _run_inner_pytest(
    testfile: Path,
    *,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run pytest in a subprocess against ``testfile``; cwd is the file's parent."""
    args = [sys.executable, "-m", "pytest", "-q", str(testfile)]
    if extra_args is not None:
        args.extend(extra_args)
    return subprocess.run(
        args,
        cwd=testfile.parent,
        capture_output=True,
        text=True,
        check=False,
    )


def test_unknown_capability_name_raises_at_collection(
    tmp_path: Path,
) -> None:
    """Collection-time validation: typos surface before any test runs."""
    testfile = tmp_path / "test_typo.py"
    testfile.write_text(
        textwrap.dedent(
            '''
            import pytest

            @pytest.mark.recalllab_optional("supprots_forget")
            def test_typo(memory_contract):
                memory_contract.given_user("ayush")
            '''
        ).lstrip()
    )
    result = _run_inner_pytest(testfile)
    # pytest.UsageError exits with code 4 (collection error).
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "unknown recalllab_optional capability" in combined, combined
    assert "supprots_forget" in combined, combined
    # Make sure the known-names hint is rendered so users can fix their typo
    # without consulting docs.
    assert "supports_forget" in combined, combined
    assert "judge_configured" in combined, combined


def test_judge_configured_marker_is_recognized(tmp_path: Path) -> None:
    """A contract marked recalllab_optional('judge_configured') must skip cleanly,
    not error, when [judge] is set to the default 'none'.
    """
    testfile = tmp_path / "test_optional.py"
    testfile.write_text(
        textwrap.dedent(
            '''
            import pytest


            @pytest.mark.recalllab_optional("judge_configured")
            def test_optional_judge(memory_contract):
                memory_contract.given_user("ayush")
                memory_contract.remember("I live in Bangalore.")
                memory_contract.should_recall(
                    "Where do I live?", latest_fact_is="Bangalore"
                )
            '''
        ).lstrip()
    )
    result = _run_inner_pytest(testfile)
    combined = result.stdout + result.stderr
    # pytest exits 0 when only-skipped tests are present.
    assert result.returncode == 0, combined
    assert "1 skipped" in combined, combined


def test_unmarked_judge_call_fails_loudly(tmp_path: Path) -> None:
    """Decision #3b end-to-end: unmarked judge call against default [judge]
    must ERROR (raise JudgeUnavailableError), not silently skip.
    """
    testfile = tmp_path / "test_unmarked.py"
    testfile.write_text(
        textwrap.dedent(
            '''
            def test_unmarked_judge(memory_contract):
                memory_contract.given_user("ayush")
                memory_contract.remember("I live in Bangalore.")
                memory_contract.should_recall(
                    "Where do I live?", latest_fact_is="Bangalore"
                )
            '''
        ).lstrip()
    )
    result = _run_inner_pytest(testfile)
    combined = result.stdout + result.stderr
    # Failure or error, NOT a skip.
    assert result.returncode != 0, combined
    assert "1 skipped" not in combined, combined
    assert "JudgeUnavailableError" in combined, combined


def test_xdist_session_cap_warning_is_emitted_when_judge_configured(
    tmp_path: Path,
) -> None:
    """Codex round-1 step-4 finding: the doc promises a session-start
    warning when pytest-xdist is detected because the per-session judge
    budget cap is enforced per-worker, not per-pytest-invocation. The
    warning fires only when judge is actually configured — rule-based-
    only suites have no judge cost to bound, so we don't pester users
    who never opted into judge mode.

    Skipped when pytest-xdist isn't installed (it's not a test
    dependency); the warning code path is still unit-tested above via
    the plugin's _build_judge no-op path.
    """
    pytest.importorskip("xdist")

    testfile = tmp_path / "test_warns.py"
    testfile.write_text("def test_noop(memory_contract): pass\n")
    toml = tmp_path / "recalllab.toml"
    toml.write_text(
        '[provider]\ntype = "reference"\n'
        '[trace]\npath = ".recalllab/traces.sqlite"\n'
        '[judge]\nprovider = "anthropic"\n'
    )

    result = _run_inner_pytest(testfile, extra_args=["-n", "0"])
    combined = result.stdout + result.stderr
    # We just need the warning string to appear; the test outcome
    # doesn't matter (the inner test will error on missing
    # ANTHROPIC_API_KEY but that's OK for this assertion).
    assert "pytest-xdist detected" in combined, combined


def test_multi_marker_honors_every_gate(tmp_path: Path) -> None:
    """A contract stacking two recalllab_optional markers honors BOTH.

    The reference adapter declares ``supports_forget=True`` and the
    default judge declares ``judge_configured=False`` (i.e. NoOpJudge has
    ``available=False``). Stacking both markers should produce a SKIP via
    the ``judge_configured`` marker — the multi-marker iterator must
    walk both, not just the closest.
    """
    testfile = tmp_path / "test_multimarker.py"
    testfile.write_text(
        textwrap.dedent(
            '''
            import pytest


            @pytest.mark.recalllab_optional("supports_forget")
            @pytest.mark.recalllab_optional("judge_configured")
            def test_two_gates(memory_contract):
                memory_contract.given_user("ayush")
            '''
        ).lstrip()
    )
    result = _run_inner_pytest(testfile)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "1 skipped" in combined, combined
