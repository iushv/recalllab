"""``recalllab record`` — turn a recorded ``ContractRun`` into a pytest regression.

The trace-to-test generator is a *pure transform* over the persisted
``ContractRun`` schema (no LLM, no provider calls, no I/O beyond reading
the trace store and writing the output file). Given a real production
failure recorded as a trace, the user gets a checked-in pytest test that
reproduces the failure the next time CI runs.

Design constraints (the v0.2.0-lessons checklist):

- **Protocol promises.** Inputs are ``TraceStore.get_run`` and the
  ``ContractRun`` Pydantic schema (both RecallLab-owned). Output is
  Python source. No external API or protocol assumption.
- **Identity audit.** Output is byte-stable for the same input run.
  Timestamps, latencies, and the run UUID do *not* appear in the
  generated source (they would change across replays of the same
  logical contract).
- **Cross-feature matrix.** ``GIVEN_USER`` / ``REMEMBER`` / ``RECALL`` +
  paired ``ASSERT`` / unpaired ``RECALL`` / ``FORGET`` / ``MUTATION``
  (distractors and stale_repeats, including partial_failed and
  unsupported statuses) are all rendered. Unknown event kinds emit a
  comment line so the regenerated test stays valid Python.
- **Adversarial scenarios.** Text with quotes / newlines / unicode is
  emitted via ``repr()``. Empty traces emit ``pass`` with an
  explanation comment.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recalllab.core.traces.schema import (
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)
from recalllab.core.traces.sqlite_store import TraceStore

# Sentinel for "this assertion mode was not present in the trace". Using
# ``None`` would conflate "absent" with "expected=None", which is a valid
# (if unusual) payload shape.
_UNSET: Any = object()

# Modes the emitter renders into ``should_recall`` kwargs. Rule-based
# modes are unconditional; judge modes only render when the configured
# judge produced a verdict OR a Decision #9 placeholder (we don't drop
# the kwarg just because the judge didn't run — that would degrade the
# regression's fidelity to the original call).
_RULE_BASED_MODES: frozenset[str] = frozenset({"contains", "excludes"})
_JUDGE_MODES: frozenset[str] = frozenset(
    {"latest_fact_is", "must_not_answer_as", "judge_assertion"}
)
_KNOWN_ASSERTION_MODES: frozenset[str] = _RULE_BASED_MODES | _JUDGE_MODES

# Rubric defaults — values that match these are omitted from the
# regenerated ``Rubric(...)`` literal for readability. Mirrors the
# defaults in ``recalllab.core.judge.rubric.Rubric``; kept inline here
# so the emitter never imports Rubric (the emitter must stay
# dependency-light so it can run against a trace from any RecallLab
# version).
_RUBRIC_DEFAULTS: dict[str, str] = {
    "pass_label": "PASS",
    "fail_label": "FAIL",
}


@dataclass(frozen=True)
class _FailedAssertion:
    """One failed assertion harvested from a multi-assert ``should_recall``.

    Kept as a tiny record so the emitter can render one documenting
    comment per failure and so a multi-failure case can label each by
    mode without losing any reason text.
    """

    mode: str
    expected: Any
    reason: str | None


# --------------------------------------------------------------------- helpers
def _py(value: Any) -> str:
    """Render a Python literal for ``value`` using ``repr``.

    ``repr`` round-trips for ``str`` / ``int`` / ``float`` / ``bool`` /
    ``None`` / ``list`` / ``dict`` of those — all the payload field
    types the trace stores. Strings get proper escaping for quotes,
    newlines, backslashes, and unicode.
    """
    return repr(value)


def _safe_comment_lines(prefix: str, body: str) -> Iterable[str]:
    """Yield one or more comment lines so ``body`` cannot escape comment context.

    The naive form ``yield f"    # {prefix}{body}"`` is unsafe when
    ``body`` comes from trace payload data: an embedded newline ends
    the comment and the text after it becomes executable Python in the
    generated regression. Splitting on ``\\n`` (and normalising ``\\r``
    line breaks) before prefixing each line keeps the comment context
    closed no matter what the trace contained. Continuation lines are
    indented under the first so a multi-line assertion reason reads as
    one logical comment block.
    """
    normalised = body.replace("\r\n", "\n").replace("\r", "\n")
    body_lines = normalised.split("\n")
    yield f"    # {prefix}{body_lines[0]}"
    for line in body_lines[1:]:
        yield f"    #   {line}"


# ------------------------------------------------------------------ emitter
def trace_to_test_source(
    run: ContractRun,
    *,
    optional_judge: bool = False,
) -> str:
    """Render ``run`` as a self-contained pytest regression file.

    Pure function: the same ``(ContractRun, optional_judge)`` pair
    always produces the same bytes. Timestamps, latencies, and the run
    UUID are intentionally omitted from the output so re-recording the
    same logical contract yields identical source.

    ``optional_judge`` (default ``False``) controls whether the
    generated test is decorated with
    ``@pytest.mark.recalllab_optional("judge_configured")`` when the
    trace contains a judge-mode assertion. Off by default per
    Decision #3b: a judge-mode regression that lives in CI should
    ERROR loudly when ``[judge]`` is unconfigured, not silently skip.
    Pass ``--optional-judge`` on the CLI when you specifically want the
    skip-with-marker behavior (e.g. running the regression locally
    without an API key).

    When the trace contains a ``REMEMBER`` with a recorded
    ``episode_id``, the generated test is decorated with
    ``@pytest.mark.recalllab_optional("supports_custom_episode_ids")``
    so the pytest plugin auto-skips it cleanly against providers that
    don't honour custom episode IDs. Without that guard, the
    regenerated ``remember`` could silently degrade to a
    provider-assigned id and a later ``forget(episode_id=X)`` from the
    same trace would no-op — the exact failure mode the round-5
    episode-id round-trip fix was supposed to close, just at the
    provider boundary instead of the emitter boundary.

    **Reproduction-fidelity caveat.** The emitter is deterministic but
    the *run* of the generated test against a fresh adapter is not a
    perfect replay:

    - ``status="partial_failed"`` mutations replay the call against a
      clean adapter; the original failure may have been provider-
      state-dependent (mid-call disconnect, race) and may not
      reproduce. The generated file documents this via an inline
      ``# NOTE: ... results may diverge ...`` comment.
    - Recall result *ranking* depends on the provider's retrieval
      backend (BM25 vs keyword overlap vs embeddings). The
      assertions are rule-based (``contains`` / ``excludes``), so
      they survive ranking differences, but the trace's recorded
      ``results`` array is not replayed against — the regenerated
      test re-runs the recall against the fresh adapter.

    Both are documented trade-offs, not bugs. v0.3 may add a
    ``--strict`` mode that fails when reproduction can't be
    guaranteed.
    """
    needs_custom_ids = _trace_needs_custom_episode_ids(run)
    has_mutations = _trace_has_mutations(run)
    uses_judge = _trace_uses_judge(run)
    uses_rubric = _trace_uses_rubric(run)
    # Any marker on the test function needs ``import pytest`` in the
    # header. Custom-id marker is unconditional when needs_custom_ids is
    # true; the judge-configured marker only fires when the user passed
    # --optional-judge (or its programmatic equivalent).
    will_emit_marker = needs_custom_ids or (uses_judge and optional_judge)
    lines: list[str] = []
    lines.extend(
        _emit_header(
            run,
            needs_pytest_import=will_emit_marker,
            uses_rubric=uses_rubric,
        )
    )
    lines.append("")
    if needs_custom_ids:
        lines.append(
            '@pytest.mark.recalllab_optional("supports_custom_episode_ids")'
        )
    if uses_judge and optional_judge:
        lines.append('@pytest.mark.recalllab_optional("judge_configured")')
    lines.append("def test_recorded_failure(memory_contract) -> None:")
    body = list(_emit_body(run))
    if has_mutations:
        # Pin the contract_id so deterministic mutation episode IDs
        # (``mut-{type}-{sha256[:16]}-{index:04d}``, hashed from
        # contract_id + user + invocation + ... by ``MemoryContract.
        # _mutation_episode_id``) reproduce the recorded trace's IDs.
        # Without this pin the regenerated test's pytest nodeid becomes
        # the contract_id, mutations write under different IDs, and any
        # later ``forget(episode_id=<mut-...>)`` from the trace silently
        # deletes nothing — round-8 Codex finding.
        prelude = [
            "    # Pin to the original contract_id so deterministic mutation",
            "    # episode IDs reproduce the recorded trace's IDs. Without this,",
            "    # mutation replay computes different ``mut-*`` IDs than what",
            "    # landed at trace time and later ID-paired operations no-op.",
            f"    memory_contract.run.contract_id = {_py(run.contract_id)}",
        ]
        body = prelude + body
    if not body:
        body.append("    # No recorded events in this run; nothing to replay.")
        body.append("    pass")
    elif not _body_has_statement(body):
        # Every body line is a comment (e.g. the trace contains only
        # unknown event kinds, unsupported mutations, or an orphan
        # ASSERT). Python requires at least one statement in a function
        # body, so append ``pass`` after the comments so the generated
        # file still parses.
        body.append("    pass")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


def _trace_needs_custom_episode_ids(run: ContractRun) -> bool:
    """Return True iff the generated test should gate on
    ``supports_custom_episode_ids``.

    Two trigger conditions, both backed by the DSL's runtime gates:

    1. A ``REMEMBER`` event carries a non-empty ``episode_id`` — the
       emitter renders ``remember(text, episode_id=...)`` and the
       round-6 fix routes that through the capability check.
    2. A non-``unsupported`` ``MUTATION`` event with a positive write
       count — the emitter renders ``with_distractors`` /
       ``with_stale_repeats``, which the DSL gates on the same
       capability. Round-9 Codex finding: ignoring mutation events
       here meant a trace recorded on a capable provider would
       generate a regression that ``RuntimeError``-fails on an
       incapable provider rather than cleanly skipping. The
       ``status="unsupported"`` case is excluded because the emitter
       already renders those as comments (no DSL call). Write counts
       of zero are excluded because the DSL's ``_require_custom_
       episode_ids_for_mutation`` short-circuits when ``write_count
       == 0`` and doesn't hit the capability gate.
    """
    for event in run.events:
        if event.kind == EventKind.REMEMBER:
            ep_id = event.payload.get("episode_id")
            if isinstance(ep_id, str) and ep_id:
                return True
        elif event.kind == EventKind.MUTATION:
            status = event.payload.get("status", "completed")
            if status == "unsupported":
                continue
            mtype = event.payload.get("type")
            count: object = 0
            if mtype == "distractors":
                count = event.payload.get("requested", 0)
            elif mtype == "stale_repeats":
                count = event.payload.get("times", 0)
            else:
                # Unknown mutation type — emitter renders only a
                # comment, no DSL call, no capability gate hit.
                continue
            if isinstance(count, int) and count > 0:
                return True
    return False


def _iter_rendered_judge_assert_modes(run: ContractRun) -> Iterable[str]:
    """Yield the ``mode`` string of every ASSERT that will be RENDERED.

    An ASSERT is "rendered" iff it's part of a contiguous group
    immediately following a RECALL event — exactly the set
    ``_emit_body`` hands to ``_emit_should_recall``. Orphan ASSERTs
    (no preceding RECALL) are emitted as comments only, so flagging
    judge usage off them would add a spurious ``import pytest`` or
    ``recalllab_optional("judge_configured")`` marker even though the
    generated test never invokes a judge — Codex round-1 step-8
    finding #2.
    """
    events = list(run.events)
    i = 0
    while i < len(events):
        if events[i].kind != EventKind.RECALL:
            i += 1
            continue
        j = i + 1
        while j < len(events) and events[j].kind == EventKind.ASSERT:
            mode = events[j].payload.get("mode")
            if isinstance(mode, str):
                yield mode
            j += 1
        i = max(j, i + 1)


def _trace_uses_judge(run: ContractRun) -> bool:
    """Return True iff a RENDERED ASSERT in the trace is judge-mode.

    Used to decide whether the optional-judge marker applies and
    whether judge-mode kwargs need to be rendered. See
    ``_iter_rendered_judge_assert_modes`` for the pairing rationale.
    """
    return any(mode in _JUDGE_MODES for mode in _iter_rendered_judge_assert_modes(run))


def _trace_uses_rubric(run: ContractRun) -> bool:
    """Return True iff a RENDERED ASSERT in the trace is judge_assertion.

    Used to decide whether ``from recalllab import Rubric`` belongs in
    the generated test's imports.
    """
    return any(
        mode == "judge_assertion" for mode in _iter_rendered_judge_assert_modes(run)
    )


def _trace_has_mutations(run: ContractRun) -> bool:
    """Return True iff the trace has any ``MUTATION`` event whose status
    isn't ``unsupported``.

    Unsupported mutations are emitted as comments (no DSL call), so
    they don't trigger the contract_id-dependent ID derivation. Only
    ``completed`` / ``partial_failed`` mutations actually invoke
    ``with_distractors`` / ``with_stale_repeats``, which is the path
    that depends on ``self._run.contract_id``.
    """
    for event in run.events:
        if event.kind != EventKind.MUTATION:
            continue
        status = event.payload.get("status", "completed")
        if status != "unsupported":
            return True
    return False


def _body_has_statement(body: list[str]) -> bool:
    """Return True iff any line in ``body`` is a non-comment, non-blank statement."""
    for line in body:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        return True
    return False


def _emit_header(
    run: ContractRun,
    *,
    needs_pytest_import: bool,
    uses_rubric: bool,
) -> Iterable[str]:
    """Module-level docstring + imports.

    User-supplied metadata (``contract_id``, ``status``) goes through
    ``repr()`` and is placed on dedicated ``#`` comment lines. A naive
    docstring interpolation would let a ``contract_id`` containing
    ``\"\"\"`` or newlines terminate the docstring and inject top-level
    code into the generated file — a real failure mode because pytest
    node IDs can include arbitrary parametrized id strings.

    ``needs_pytest_import=True`` adds ``import pytest`` so any
    decorator (``recalllab_optional`` for custom ids or
    judge_configured) resolves. The caller decides based on which
    markers will fire so this header stays minimal when no markers
    apply. ``uses_rubric=True`` adds ``from recalllab import Rubric``
    so the rendered ``judge_assertion=Rubric(...)`` literal resolves.
    """
    yield '"""Recorded regression generated by `recalllab record`.'
    yield ""
    yield (
        "Do not edit by hand. Regenerate with `recalllab record --run-id "
        "<id>` if the trace changes."
    )
    yield '"""'
    yield ""
    yield f"# Source contract: {run.contract_id!r}"
    yield f"# Original status:  {run.status.value!r}"
    yield ""
    yield "from __future__ import annotations"
    if needs_pytest_import:
        yield ""
        yield "import pytest"
    if uses_rubric:
        yield ""
        yield "from recalllab import Rubric"


# Event kinds whose DSL replay requires an active user. A
# ``MemoryContract.{remember,recall,forget,with_*}`` call without a
# prior ``given_user`` raises ``RuntimeError`` from ``_require_user``,
# so when these arrive before any GIVEN_USER event we synthesise one.
_USER_DEPENDENT_KINDS = frozenset({
    EventKind.REMEMBER,
    EventKind.RECALL,
    EventKind.FORGET,
    EventKind.MUTATION,
})


def _emit_body(run: ContractRun) -> Iterable[str]:
    """Walk the trace's events and emit DSL calls in order.

    Round-7 Codex finding: the emitter preserved trace order strictly,
    so a trace that started with a user-dependent event (REMEMBER /
    RECALL / FORGET / MUTATION) without a preceding GIVEN_USER produced
    a regression that raised ``RuntimeError("no active user...")`` from
    the DSL before reaching the recorded assertion — a generator-
    induced setup failure that masked the real bug. RecallLab-produced
    traces can't have this shape (``_require_user`` would have raised
    at trace time), but external exports, partial dumps, and schema
    migrations can. We synthesise a ``given_user`` from the event's
    payload ``user_id`` when needed, with an explicit comment marking
    that the row didn't come from the trace.
    """
    events = list(run.events)
    i = 0
    # Tracks the most recent ``given_user`` we've emitted (real OR
    # synthesised). ``None`` until the first user is established.
    active_user: str | None = None
    while i < len(events):
        event = events[i]
        kind = event.kind

        # Pre-dispatch: synthesise a given_user if the payload's
        # user_id doesn't match the active one. Two trigger shapes:
        #
        # 1. ``active_user is None`` — round-7 case: trace opens with a
        #    user-dependent event (no preceding GIVEN_USER). Payload's
        #    user_id seeds the active user.
        # 2. ``active_user != payload_user`` — round-10 case: the
        #    trace switches users mid-stream without an intervening
        #    GIVEN_USER event. Without this branch, the second user's
        #    REMEMBER / RECALL / FORGET / MUTATION replays under the
        #    first user's tenant — the generated regression silently
        #    exercises the wrong namespace and can pass while hiding
        #    the original failure.
        #
        # In both cases, the payload's user_id is authoritative because
        # the DSL records it at every user-dependent call.
        if kind in _USER_DEPENDENT_KINDS:
            payload_user = event.payload.get("user_id")
            if (
                isinstance(payload_user, str)
                and payload_user
                and payload_user != active_user
            ):
                reason = "initial" if active_user is None else "switch"
                yield from _emit_synthesized_given_user(
                    payload_user, event.sequence, reason=reason
                )
                active_user = payload_user

        if kind == EventKind.GIVEN_USER:
            payload_user = event.payload.get("user_id")
            if isinstance(payload_user, str):
                active_user = payload_user
            yield from _emit_given_user(event)
            i += 1
        elif kind == EventKind.REMEMBER:
            yield from _emit_remember(event)
            i += 1
        elif kind == EventKind.RECALL:
            # The DSL's ``should_recall(query, contains=X, excludes=Y)``
            # records ONE recall followed by ONE ASSERT per assertion
            # mode (so up to two ASSERTs back-to-back). Pairing only the
            # next ASSERT would silently drop the second mode — and if
            # the second mode was the failing assertion, the regenerated
            # regression would pass while the original run failed.
            assert_group, next_i = _collect_asserts(events, i + 1)
            if assert_group:
                yield from _emit_should_recall(event, assert_group)
                i = next_i
            else:
                yield from _emit_recall(event)
                i += 1
        elif kind == EventKind.FORGET:
            yield from _emit_forget(event)
            i += 1
        elif kind == EventKind.MUTATION:
            yield from _emit_mutation(event)
            i += 1
        elif kind == EventKind.ASSERT:
            # An ASSERT without a preceding RECALL is unusual; we surface
            # it as a comment so the trace is documented but no
            # malformed DSL call is emitted.
            yield (
                f"    # orphan assert at sequence {event.sequence} "
                f"(mode={event.payload.get('mode')!r}, "
                f"expected={event.payload.get('expected')!r})"
            )
            i += 1
        else:
            yield f"    # unsupported event kind: {kind!r}"
            i += 1


def _emit_given_user(event: TraceEvent) -> Iterable[str]:
    user_id = event.payload.get("user_id", "unknown")
    yield f"    memory_contract.given_user({_py(user_id)})"


def _emit_synthesized_given_user(
    user_id: str,
    before_sequence: int,
    *,
    reason: str = "initial",
) -> Iterable[str]:
    """Emit a synthesised ``given_user`` call when the trace omits one.

    Two reasons fire the synthesis:

    - ``reason="initial"`` — the trace opens with a user-dependent
      event and never set an active user. RecallLab-produced traces
      always have a preceding GIVEN_USER (``_require_user`` would have
      raised otherwise), so this only fires for non-canonical sources
      (schema migration, partial dump, third-party export).
    - ``reason="switch"`` — the trace's payload ``user_id`` changed
      from the active one without an intervening GIVEN_USER event.
      Without synthesis, the regenerated test would replay under the
      previous user's tenant — round-10 Codex finding.

    The comment above the synthesised call documents which case fired
    so the developer reading the file can audit what the generator
    inferred.
    """
    if reason == "switch":
        yield (
            f"    # given_user synthesized by `recalllab record` from "
            f"event sequence {before_sequence} payload (trace switched "
            f"user_id without an intervening GIVEN_USER event)"
        )
    else:
        yield (
            f"    # given_user synthesized by `recalllab record` from "
            f"event sequence {before_sequence} payload "
            f"(no GIVEN_USER recorded earlier in trace)"
        )
    yield f"    memory_contract.given_user({_py(user_id)})"


def _emit_remember(event: TraceEvent) -> Iterable[str]:
    """Render a ``REMEMBER`` event as a DSL ``remember`` call.

    Forwards the recorded ``episode_id`` to ``remember(text, episode_id=...)``
    so later ID-paired operations in the same trace (``forget(episode_id=X)``,
    capability-checked recall, custom-id idempotency assertions) actually
    address the same row in the regenerated run. Round-5 Codex finding:
    dropping the recorded id let the emitter produce regressions where a
    later ``forget(episode_id=X)`` silently deleted nothing because the
    regenerated ``remember`` had been assigned a fresh provider-side uuid.
    """
    text = event.payload.get("text", "")
    episode_id = event.payload.get("episode_id")
    if isinstance(episode_id, str) and episode_id:
        yield (
            f"    memory_contract.remember({_py(text)}, "
            f"episode_id={_py(episode_id)})"
        )
    else:
        yield f"    memory_contract.remember({_py(text)})"


def _emit_recall(event: TraceEvent) -> Iterable[str]:
    query = event.payload.get("query", "")
    k = event.payload.get("k", 5)
    if k == 5:
        yield f"    memory_contract.recall({_py(query)})"
    else:
        yield f"    memory_contract.recall({_py(query)}, k={int(k)})"


def _emit_should_recall(
    recall_event: TraceEvent,
    assert_events: list[TraceEvent],
) -> Iterable[str]:
    """Render a ``should_recall`` call from a recall + one or more asserts.

    The DSL's ``should_recall(query, contains=X, excludes=Y,
    latest_fact_is=Z, must_not_answer_as=[...], judge_assertion=R)``
    records a single ``RECALL`` followed by up to two ``ASSERT`` events
    (a rule-based assertion plus at most one judge-mode assertion;
    Decision #3a forbids combining two judge modes in one call). The
    emitter gathers every contiguous ASSERT and rebuilds a single
    ``should_recall`` call with all the kwargs the original used —
    including the judge-mode kwarg even when the trace's ASSERT is a
    Decision #9 short-circuit placeholder, because regenerating the
    call WITHOUT the judge kwarg would silently degrade the
    regression's fidelity to the original.
    """
    query = recall_event.payload.get("query", "")
    k = recall_event.payload.get("k", 5)
    args: list[str] = [_py(query)]
    if k != 5:
        args.append(f"k={int(k)}")

    # Group asserts by mode. The DSL never emits two asserts with the
    # same mode in one should_recall (each mode is checked once), but
    # if the trace contains duplicates we keep the *last* — the most
    # recently recorded value.
    rendered_kwargs: dict[str, Any] = {}
    unknown_modes: list[tuple[str, Any]] = []
    failed_assertions: list[_FailedAssertion] = []

    for assert_event in assert_events:
        mode = assert_event.payload.get("mode", "")
        expected = assert_event.payload.get("expected")
        # ``passed`` is three-valued from v0.2.2: True / False / None.
        # ``None`` is a short-circuit placeholder (Decision #9) and must NOT
        # be reported as a failure. A missing ``passed`` key (legacy traces)
        # defaults to True; anything that round-trips a literal ``None``
        # stays ``None``. Don't ``bool()`` here — that would coerce ``None``
        # into ``False`` and falsely flag placeholders as failures.
        passed_raw = assert_event.payload.get("passed", True)
        passed: bool | None = None if passed_raw is None else bool(passed_raw)
        reason = assert_event.payload.get("reason")
        reason_text = reason if isinstance(reason, str) else None
        if mode in _KNOWN_ASSERTION_MODES:
            rendered_kwargs[mode] = expected
        else:
            unknown_modes.append((mode, expected))
            # Don't track unknown-mode pass/fail as a "failed" assertion
            # because the emitter doesn't render the call anyway.
            continue
        if passed is False:
            failed_assertions.append(_FailedAssertion(mode, expected, reason_text))

    if not rendered_kwargs:
        # Every assert was an unsupported mode. Emit a plain recall
        # plus a comment per skipped assert so the developer knows
        # what didn't get replayed.
        for mode, expected in unknown_modes:
            yield (
                f"    # original assertion mode {mode!r} not yet supported "
                f"by the v0.2.2 emitter (expected={_py(expected)})"
            )
        yield from _emit_recall(recall_event)
        return

    # Emit kwargs in a canonical order so byte-stability holds across
    # trace re-records: rule-based first (matching their declaration
    # order in the DSL signature), then judge modes. Track separately
    # how many *assertion* kwargs make it into the call so the
    # corrupt-Rubric fall-through can detect "nothing left to assert"
    # and degrade to a plain recall rather than a should_recall with
    # only the query (which the DSL rejects with ValueError).
    assertion_kwarg_count = 0
    for kwarg in ("contains", "excludes", "latest_fact_is", "must_not_answer_as"):
        if kwarg in rendered_kwargs:
            args.append(f"{kwarg}={_py(rendered_kwargs[kwarg])}")
            assertion_kwarg_count += 1
    judge_assertion_corrupt_comment: str | None = None
    if "judge_assertion" in rendered_kwargs:
        # judge_assertion stores Rubric.model_dump() (dict) on the
        # trace's ``expected`` field; render it as a Rubric literal so
        # the regenerated test reads naturally. When the stored shape
        # is corrupt (no criterion), drop the kwarg entirely and
        # surface a comment — emitting Rubric(**{...}) would compile
        # but raise ValidationError at runtime, masking the original
        # regression behind a misleading failure.
        stored_rubric = rendered_kwargs["judge_assertion"]
        if _is_renderable_rubric(stored_rubric):
            args.append(
                f"judge_assertion={_emit_rubric_literal(stored_rubric)}"
            )
            assertion_kwarg_count += 1
        else:
            judge_assertion_corrupt_comment = (
                "    # judge_assertion ASSERT had a corrupt Rubric "
                f"payload (no 'criterion' field): {_py(stored_rubric)}. "
                "Kwarg dropped from regenerated call so the regression "
                "still runs — fix the source trace if you need the "
                "judge assertion to replay."
            )

    if assertion_kwarg_count == 0:
        # Every assertion kwarg ended up dropped (e.g. trace has only
        # a corrupt judge_assertion). The DSL rejects
        # ``should_recall(query)`` with no assertion kwargs as
        # ValueError("needs at least one of..."), so a regenerated
        # ``should_recall(query)`` would crash on replay. Fall through
        # to plain ``recall(...)`` + the corrupt-Rubric / unknown-mode
        # comments so the regression preserves the recall side of the
        # call without tripping the DSL validation gate. (Round-2
        # confirming review on the corrupt-Rubric fallback path.)
        if judge_assertion_corrupt_comment is not None:
            yield judge_assertion_corrupt_comment
        for mode, expected in unknown_modes:
            yield (
                f"    # original assertion mode {mode!r} not yet supported "
                f"by the v0.2.2 emitter (expected={_py(expected)})"
            )
        yield from _emit_recall(recall_event)
        return

    # Documenting comments for each failed assertion. Including the mode
    # in the label keeps multi-failure cases readable.
    for failure in failed_assertions:
        prefix = f"original assertion failed ({failure.mode}): "
        if failure.reason:
            yield from _safe_comment_lines(prefix, failure.reason)
        else:
            yield f"    # original assertion failed ({failure.mode})"
    # And any unsupported-mode asserts get a comment so the developer
    # sees the regenerated test doesn't replay them.
    for mode, expected in unknown_modes:
        yield (
            f"    # original assertion mode {mode!r} skipped by emitter "
            f"(expected={_py(expected)})"
        )

    # Decision #9 audit comment: when a judge mode was short-circuited
    # at record time (passed=None), surface a neutral comment so a
    # reader knows the regenerated judge kwarg won't necessarily
    # invoke the judge (the same short-circuit will reproduce on
    # replay). Separate from the "failed" comment above so the
    # semantic distinction reads cleanly in the regenerated file.
    for assert_event in assert_events:
        if assert_event.payload.get("passed") is not None:
            continue
        mode = assert_event.payload.get("mode", "")
        if mode not in _JUDGE_MODES:
            continue
        reason = assert_event.payload.get("reason")
        prefix = f"original assertion short-circuited ({mode}): "
        if isinstance(reason, str) and reason:
            yield from _safe_comment_lines(prefix, reason)
        else:
            yield f"    # original assertion short-circuited ({mode})"

    if judge_assertion_corrupt_comment is not None:
        yield judge_assertion_corrupt_comment

    rendered = ", ".join(args)
    yield f"    memory_contract.should_recall({rendered})"


def _is_renderable_rubric(value: Any) -> bool:
    """Return True iff ``value`` is a dict with a non-empty ``criterion``.

    Used to decide whether ``_emit_rubric_literal`` can produce a
    valid call. A trace with a corrupt or legacy shape (no criterion)
    triggers the fallback "drop the kwarg + emit a comment" path so
    the regenerated test still runs, instead of compiling but raising
    at execution time inside ``Rubric(**{...})``.
    """
    if not isinstance(value, dict):
        return False
    criterion = value.get("criterion")
    return isinstance(criterion, str) and len(criterion) > 0


def _emit_rubric_literal(expected: dict[str, Any]) -> str:
    """Render a ``Rubric(...)`` literal from a trace's stored value.

    ``expected`` is ``Rubric.model_dump()`` (a dict of criterion +
    pass_label + fail_label) per the step-6+7 trace normalization.
    Default-valued fields are omitted so the generated file stays
    readable for the common case
    ``judge_assertion=Rubric(criterion="...")``.

    Callers must gate on :func:`_is_renderable_rubric` first; a dict
    missing ``criterion`` raises ``KeyError`` here rather than
    silently emitting an invalid ``Rubric(**...)`` literal that
    would compile but crash at runtime.
    """
    kwargs: list[str] = [f"criterion={_py(expected['criterion'])}"]
    for field, default in _RUBRIC_DEFAULTS.items():
        value = expected.get(field, default)
        if value != default:
            kwargs.append(f"{field}={_py(value)}")
    return f"Rubric({', '.join(kwargs)})"


def _emit_forget(event: TraceEvent) -> Iterable[str]:
    matching = event.payload.get("matching")
    episode_id = event.payload.get("episode_id")
    if isinstance(episode_id, str):
        yield f"    memory_contract.forget(episode_id={_py(episode_id)})"
    elif isinstance(matching, str):
        yield f"    memory_contract.forget(matching={_py(matching)})"
    else:
        yield "    # forget event missing both matching and episode_id"


def _emit_mutation(event: TraceEvent) -> Iterable[str]:
    mutation_type = event.payload.get("type")
    status = event.payload.get("status", "completed")
    if status == "unsupported":
        yield (
            "    # MUTATION was skipped at record time because the "
            "provider did not declare supports_custom_episode_ids; "
            "regenerated test would fail the same gate."
        )
        return
    # Round-11 code-review note: a partial-failed mutation may not
    # reproduce against a fresh adapter — the original failure can have
    # been provider-state-dependent (mid-call disconnect, transient
    # disk-full, race against another writer), and the regenerated
    # adapter starts clean. We replay the call so the test exists in
    # the regression file, but the developer needs to know the
    # regression may pass OR fail and neither matches the original
    # deterministically. The fix is documented; the comment text is
    # the user-facing signal.
    partial_comment = (
        "    # NOTE: original mutation recorded "
        f"status={status!r}. The regenerated test replays the call "
        "against a fresh adapter; results may diverge from the "
        "original (the failure may not reproduce, or may reproduce "
        "with a different shape)."
        if status != "completed"
        else None
    )
    if mutation_type == "distractors":
        n = int(event.payload.get("requested", 0))
        seed = int(event.payload.get("seed", 0))
        if partial_comment is not None:
            yield partial_comment
        if seed == 0:
            yield f"    memory_contract.with_distractors({n})"
        else:
            yield f"    memory_contract.with_distractors({n}, seed={seed})"
    elif mutation_type == "stale_repeats":
        times = int(event.payload.get("times", 0))
        if partial_comment is not None:
            yield partial_comment
        yield f"    memory_contract.with_stale_repeats(times={times})"
    else:
        yield f"    # unsupported mutation type: {mutation_type!r}"


def _collect_asserts(
    events: list[TraceEvent], start: int
) -> tuple[list[TraceEvent], int]:
    """Collect contiguous ``ASSERT`` events starting at index ``start``.

    Returns ``(asserts, next_index)`` where ``next_index`` is the index
    of the first non-ASSERT event (or ``len(events)`` if the scan ran
    off the end). The DSL records one ``ASSERT`` per assertion mode in
    a single ``should_recall`` call, so a recall is followed by zero
    (plain ``recall``), one (``should_recall`` with one mode), or two
    (``should_recall`` with both ``contains`` and ``excludes``)
    immediately-contiguous ASSERT events. Earlier "next assert" logic
    consumed only the first, silently dropping the second mode — and
    if the second was the failing assertion, the regenerated regression
    would pass while the original run failed.
    """
    j = start
    asserts: list[TraceEvent] = []
    while j < len(events) and events[j].kind == EventKind.ASSERT:
        asserts.append(events[j])
        j += 1
    return asserts, j


# ------------------------------------------------------------------- CLI glue
def cmd_record(
    *,
    trace_path: Path,
    run_id: str | None,
    latest_failure: bool,
    out_path: Path,
    force: bool = False,
    optional_judge: bool = False,
) -> int:
    """Read one ``ContractRun`` from ``trace_path`` and write a regression test.

    Either ``run_id`` or ``latest_failure`` must be supplied. The
    selected run is rendered via :func:`trace_to_test_source` and
    written to ``out_path``.

    Refuses to overwrite an existing file unless ``force=True``: a typo
    in ``--out``, or re-running the command with the same path as a
    previously-checked-in regression, would otherwise silently destroy
    real code. With ``force=True`` the write goes through a temp file
    in the same directory and an atomic rename, so a crash mid-write
    cannot leave a half-written test file in the user's tree.
    """
    if not trace_path.exists():
        print(f"Trace store not found at {trace_path}.")
        print()
        print(
            "Run your contracts at least once to populate it, then re-run "
            "`recalllab record`."
        )
        return 1

    store = TraceStore(trace_path)
    run = _select_run(store, run_id=run_id, latest_failure=latest_failure)
    if run is None:
        if run_id is not None:
            print(f"No run found with id {run_id!r} in {trace_path}.")
        else:
            print(
                f"No failed runs found in {trace_path}. Pass --run-id "
                "<id> to record a specific run, or run a failing "
                "contract first."
            )
        return 1

    # Refuse to clobber an existing checked-in test by default. Round-4
    # Codex finding: an unconditional ``write_text`` makes a typo in
    # ``--out`` (or a reused path across reruns) silently destroy real
    # code in the user's tree — an outsized risk for a command that's
    # marketed as the fast path from CI failure to committed
    # regression. The user must opt in with ``--force`` to overwrite.
    if out_path.exists() and not force:
        print(f"Refusing to overwrite existing file at {out_path}.")
        print()
        print(
            "Re-run with --force to replace it, or pass a different "
            "--out path. (The generated source is byte-stable for the "
            "same input run, so an atomic --force overwrite is safe.)"
        )
        return 1

    source = trace_to_test_source(run, optional_judge=optional_judge)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_path, source)
    print(f"Wrote {out_path} ({len(source.splitlines())} lines).")
    print(f"  source run:    {run.id}")
    print(f"  contract_id:   {run.contract_id}")
    print(f"  original:      {run.status.value}")
    return 0


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via temp file + ``os.replace``.

    ``Path.write_text`` opens the destination, truncates it, then writes
    — so a crash, signal, or disk-full event between truncate and the
    final write leaves a half-written file in place of the original
    user data. Writing to a sibling temp file first and then renaming
    means the destination either holds the old content or the complete
    new content, never a half-written merge. ``os.replace`` is atomic
    on POSIX (single-rename within a filesystem) and replaces
    existing files on Windows too.

    The temp file is created in the same directory as the destination
    so the rename stays on the same filesystem (cross-filesystem
    renames degrade to copy+delete, which is no longer atomic).
    """
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(content)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file if anything went wrong before the
        # rename. ``missing_ok=True`` covers the case where the rename
        # already moved it.
        tmp_path.unlink(missing_ok=True)
        raise


def _select_run(
    store: TraceStore,
    *,
    run_id: str | None,
    latest_failure: bool,
) -> ContractRun | None:
    if run_id is not None:
        return store.get_run(run_id)
    if latest_failure:
        # Indexed lookup — finds the latest failed run regardless of how
        # many passed runs have accumulated since.
        return store.get_latest_run_by_status(RunStatus.FAILED)
    return None
