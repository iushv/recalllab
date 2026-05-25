"""Unit tests for the v0.2.1 trace-to-test emitter.

Pins the four mandatory stability properties from the v0.2.0-lessons
checklist plus the cross-feature matrix from the design notes in
``src/recalllab/cli/record.py``:

1. Same ``ContractRun`` → same output bytes.
2. Same logical run with different timestamps / latency / UUID → same
   output bytes (the things that *should* be irrelevant are).
3. Different events → different output.
4. Canonical 6-failure-mode run renders cleanly and round-trips through
   ``compile``.

Plus per-event-kind rendering checks (given_user, remember,
recall+assert pair, unpaired recall, forget, mutations).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from recalllab.cli.record import trace_to_test_source
from recalllab.core.traces.schema import (
    AssertionResult,
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)

# Pinned datetime so the helper's default doesn't bake wall-clock time
# into trace events. Round-1 code-review finding: if any future emitter
# change ever lets timestamps leak into output, ``datetime.now()`` as
# the default would make tests pass locally and flake in CI. A frozen
# value also makes test inputs fully deterministic.
_FROZEN_DEFAULT_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _event(
    seq: int,
    kind: EventKind,
    payload: dict[str, object],
    *,
    when: datetime | None = None,
) -> TraceEvent:
    return TraceEvent(
        sequence=seq,
        kind=kind,
        payload=payload,
        timestamp=when or _FROZEN_DEFAULT_TS,
    )


def _run(
    *,
    events: list[TraceEvent],
    assertions: list[AssertionResult] | None = None,
    contract_id: str = "tests/recorded::demo",
    status: RunStatus = RunStatus.FAILED,
) -> ContractRun:
    return ContractRun(
        id=uuid4().hex,
        contract_id=contract_id,
        provider="reference",
        started_at=datetime.now(tz=UTC),
        status=status,
        events=events,
        assertions=assertions or [],
    )


# ---------------------------------------------------------- stability tests
def test_same_run_renders_byte_stable_output() -> None:
    """Identical inputs must produce identical bytes (deterministic)."""
    when = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    run = _run(
        events=[
            _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}, when=when),
            _event(1, EventKind.REMEMBER, {"text": "I live in Mumbai."}, when=when),
        ]
    )
    first = trace_to_test_source(run)
    second = trace_to_test_source(run)
    assert first == second


def test_timestamp_changes_do_not_shift_output() -> None:
    """Identity audit: timestamps must not appear in the source.

    Re-recording the same logical contract at a different time has to
    produce identical bytes, otherwise the checked-in regression
    diffs every run.
    """
    payload: dict[str, object] = {"user_id": "ayush"}
    run_a = _run(events=[
        _event(0, EventKind.GIVEN_USER, payload,
               when=datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    ])
    run_b = _run(events=[
        _event(0, EventKind.GIVEN_USER, payload,
               when=datetime(2026, 5, 19, 23, 59, tzinfo=UTC))
    ])
    assert trace_to_test_source(run_a) == trace_to_test_source(run_b)


def test_run_uuid_does_not_appear_in_output() -> None:
    """The volatile run UUID must not contaminate the byte-stable output."""
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"})
    ])
    source = trace_to_test_source(run)
    assert run.id not in source


def test_different_events_produce_different_output() -> None:
    """The emitter must respond to actual content changes."""
    run_a = _run(events=[
        _event(0, EventKind.REMEMBER, {"text": "Mumbai"})
    ])
    run_b = _run(events=[
        _event(0, EventKind.REMEMBER, {"text": "Bangalore"})
    ])
    assert trace_to_test_source(run_a) != trace_to_test_source(run_b)


def test_emitted_source_compiles_to_valid_python() -> None:
    """Whatever the trace shape, the output must be syntactically valid."""
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {"text": "I live in Mumbai."}),
        _event(2, EventKind.RECALL, {"query": "Where do I live?", "k": 5}),
        _event(3, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True
        }),
    ])
    source = trace_to_test_source(run)
    # If compile() raises SyntaxError, the test fails with a clear
    # location — exactly what we want when an emitter regression slips
    # in.
    compile(source, "<emitted>", "exec")


# ---------------------------------------------------------- per-kind rendering
def test_given_user_event_renders() -> None:
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"})
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.given_user('ayush')" in src


def test_remember_event_with_episode_id_renders_kwarg() -> None:
    """Round-5 Codex finding: traces record the episode_id the original
    run produced. The emitter must forward it to ``remember(text,
    episode_id=...)`` so a later ``forget(episode_id=X)`` in the same
    trace actually deletes the row rather than silently no-op-ing
    against a freshly-generated uuid.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "user_id": "ayush",
            "text": "I live in Mumbai.",
            "episode_id": "ep-7",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert (
        "memory_contract.remember('I live in Mumbai.', episode_id='ep-7')"
        in src
    )


def test_remember_event_without_episode_id_omits_kwarg() -> None:
    """Older traces (or hand-built ones) that don't carry ``episode_id``
    must still emit a plain ``remember(text)`` call — no spurious
    ``episode_id=None`` argument that would change semantics.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {"text": "I live in Mumbai."}),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "memory_contract.remember('I live in Mumbai.')" in src
    assert "episode_id=" not in src


def test_remember_event_with_empty_episode_id_omits_kwarg() -> None:
    """An empty string id (degenerate payload from a future adapter
    that returns ``""``) is treated as absent — better than emitting
    ``episode_id=''`` and trying to make that round-trip.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {"text": "X", "episode_id": ""}),
    ])
    src = trace_to_test_source(run)
    assert "episode_id=" not in src


def test_generated_test_has_capability_marker_when_remember_has_episode_id() -> None:
    """Round-6 Codex finding: forwarding ``episode_id`` to ``remember``
    silently degrades to provider-assigned ids on adapters that don't
    declare ``supports_custom_episode_ids``. The fix gates the
    generated test on the capability via the pytest plugin's existing
    ``recalllab_optional`` marker, so incompatible providers skip
    cleanly with a clear reason rather than silently mis-replaying.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "text": "I live in Mumbai.",
            "episode_id": "ep-1",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Marker emitted as decorator on the test function.
    assert (
        '@pytest.mark.recalllab_optional("supports_custom_episode_ids")'
        in src
    )
    # pytest import resolved.
    assert "import pytest" in src


def test_generated_test_omits_capability_marker_when_no_episode_ids() -> None:
    """ID-free traces produce minimal output — no pytest import, no
    decorator. The capability gate only fires when the trace actually
    depends on custom episode IDs.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {"text": "I live in Mumbai."}),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "recalllab_optional" not in src
    assert "import pytest" not in src


def test_capability_marker_does_not_break_direct_function_call() -> None:
    """The runpy-based execution test (in test_record_cli.py) loads the
    generated file and calls ``test_recorded_failure`` directly. The
    new ``@pytest.mark.recalllab_optional(...)`` decorator must not
    interfere with that direct-call pattern — pytest marks attach a
    ``pytestmark`` attribute but don't change the call signature.
    """
    import runpy
    import tempfile

    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {
            "text": "I live in Mumbai.",
            "episode_id": "ep-1",
        }),
    ])
    src = trace_to_test_source(run)
    # Write to a temp file and load via runpy — that's what
    # test_record_cli's end-to-end test does. We're just verifying the
    # marker doesn't break the load + lookup path.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fp:
        fp.write(src)
        tmp_path = fp.name
    try:
        ns = runpy.run_path(tmp_path)
        test_fn = ns["test_recorded_failure"]
        assert callable(test_fn)
        # And the marker decoration attached a pytestmark attribute that
        # pytest's plugin can read.
        assert hasattr(test_fn, "pytestmark")
        marks = test_fn.pytestmark
        assert any(
            m.name == "recalllab_optional"
            and m.args == ("supports_custom_episode_ids",)
            for m in marks
        )
    finally:
        import os

        os.unlink(tmp_path)


# ---------------------------------- round-7: synthesise missing GIVEN_USER
def test_synthesizes_given_user_for_remember_only_trace() -> None:
    """Round-7 Codex finding: a trace whose first event is REMEMBER
    (no GIVEN_USER) would emit a regression that raises
    ``RuntimeError("no active user")`` before reaching the recorded
    behaviour. Synthesise from the event's payload ``user_id`` so the
    regression runs to the recorded assertion.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "user_id": "ayush",
            "text": "I live in Mumbai.",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Synthesised given_user with the right user.
    assert "memory_contract.given_user('ayush')" in src
    # Documenting comment so the file is self-describing.
    assert "synthesized by `recalllab record`" in src
    assert "no GIVEN_USER recorded earlier in trace" in src


def test_synthesizes_given_user_for_recall_only_trace() -> None:
    """Same synthesis must fire for RECALL-only traces.

    A trace that opens with RECALL is realistic for traces sliced /
    exported from a long-running run where the earlier remember
    happened outside the recorded window.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {
            "user_id": "alice",
            "query": "what did we talk about?",
            "k": 5,
            "results": [],
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "memory_contract.given_user('alice')" in src
    assert "memory_contract.recall(" in src
    # The synthesised given_user comes BEFORE the recall in source order.
    assert src.index("given_user('alice')") < src.index("memory_contract.recall")


def test_synthesis_only_fires_once_for_multi_remember_no_given_user() -> None:
    """Two REMEMBERs at the start (no GIVEN_USER) get ONE synthesised
    given_user, not two. The active-user tracker stays set after the
    first synthesis.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "Fact A",
        }),
        _event(1, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "Fact B",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Exactly one given_user call rendered (not two).
    assert src.count("memory_contract.given_user(") == 1
    # And exactly one synthesis comment.
    assert src.count("synthesized by `recalllab record`") == 1


def test_explicit_given_user_suppresses_synthesis() -> None:
    """A trace that DOES start with GIVEN_USER must NOT trigger
    synthesis — the active-user tracker is set on the GIVEN_USER
    branch and the user-dependent events that follow find it already
    in place.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "Fact A",
        }),
    ])
    src = trace_to_test_source(run)
    assert "synthesized by `recalllab record`" not in src
    # Exactly one given_user — the real one, not a synthesis duplicate.
    assert src.count("memory_contract.given_user(") == 1


def test_synthesis_skipped_when_payload_user_id_missing_or_empty() -> None:
    """When the user-dependent event's payload also lacks ``user_id``
    (or it's empty), we can't synthesise honestly. The event still
    emits its DSL call; the regenerated test will surface the runtime
    error rather than the emitter guessing.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "text": "no user_id in payload",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # No synthesis.
    assert "synthesized by `recalllab record`" not in src
    assert "memory_contract.given_user(" not in src
    # The remember IS still emitted — the test will raise at runtime,
    # which is the honest outcome for a malformed trace we can't
    # repair.
    assert "memory_contract.remember(" in src


# ---------------------------------- round-10: mid-trace user switching
def test_mid_trace_user_switch_without_given_user_synthesizes() -> None:
    """Round-10 Codex finding: ``GIVEN_USER alice → REMEMBER alice →
    REMEMBER bob`` (no explicit GIVEN_USER bob) replayed bob's memory
    under alice's namespace because the synthesis guard only fired
    when ``active_user is None``. The fix also synthesises when the
    payload user_id changes mid-trace.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "alice"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "alice", "text": "alice's fact",
        }),
        _event(2, EventKind.REMEMBER, {
            "user_id": "bob", "text": "bob's fact",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Both given_user calls appear.
    assert "memory_contract.given_user('alice')" in src
    assert "memory_contract.given_user('bob')" in src
    # The mid-trace synthesis comment is distinct from the initial-
    # case comment.
    assert "trace switched user_id without an intervening GIVEN_USER" in src
    # Source order: alice's given_user → alice's remember → synthesized
    # bob's given_user → bob's remember.
    alice_gu = src.index("given_user('alice')")
    alice_rm = src.index("remember(\"alice's fact\")")
    bob_gu = src.index("given_user('bob')")
    bob_rm = src.index("remember(\"bob's fact\")")
    assert alice_gu < alice_rm < bob_gu < bob_rm


def test_explicit_given_user_switch_does_not_synthesize() -> None:
    """Canonical RecallLab traces have explicit GIVEN_USER events at
    every switch. Synthesis must NOT double-emit when the trace
    already has them.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "alice"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "alice", "text": "fact1",
        }),
        _event(2, EventKind.GIVEN_USER, {"user_id": "bob"}),
        _event(3, EventKind.REMEMBER, {
            "user_id": "bob", "text": "fact2",
        }),
    ])
    src = trace_to_test_source(run)
    # No synthesis comments (neither initial nor switch).
    assert "synthesized by `recalllab record`" not in src
    # Exactly two given_user calls — the real ones, not duplicated.
    assert src.count("memory_contract.given_user(") == 2


def test_same_user_events_after_given_user_do_not_synthesize() -> None:
    """A run of events for the same active user must NOT spuriously
    synthesise given_user on every event. Only changes trigger it.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "fact1",
        }),
        _event(2, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "fact2",
        }),
        _event(3, EventKind.RECALL, {
            "user_id": "ayush", "query": "any fact?",
        }),
    ])
    src = trace_to_test_source(run)
    assert "synthesized by `recalllab record`" not in src
    assert src.count("memory_contract.given_user(") == 1


def test_user_switch_synthesis_handles_initial_user_too() -> None:
    """A trace with no opening GIVEN_USER and then a user switch
    mid-stream should fire BOTH synthesis kinds — initial for the
    first event, switch for the second.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "user_id": "alice", "text": "alice's fact",
        }),
        _event(1, EventKind.REMEMBER, {
            "user_id": "bob", "text": "bob's fact",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Both kinds of synthesis comments present.
    assert "no GIVEN_USER recorded earlier in trace" in src
    assert "trace switched user_id without an intervening GIVEN_USER" in src
    assert "memory_contract.given_user('alice')" in src
    assert "memory_contract.given_user('bob')" in src


# ---------------------------------- round-9: mutation traces gate on capability
def test_completed_with_distractors_emits_capability_marker() -> None:
    """Round-9 Codex finding: a trace with a non-``unsupported``
    MUTATION event drives ``with_distractors`` / ``with_stale_repeats``
    in the regenerated test, and those methods gate on
    ``supports_custom_episode_ids`` via the DSL. A trace recorded on a
    capable provider then run against an incapable one would
    ``RuntimeError``-fail instead of cleanly skipping; the marker
    closes that gap.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 5,
            "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert (
        '@pytest.mark.recalllab_optional("supports_custom_episode_ids")'
        in src
    )
    assert "import pytest" in src


def test_completed_with_stale_repeats_emits_capability_marker() -> None:
    """Same trigger for ``with_stale_repeats`` mutations."""
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "ayush", "text": "I live in Bangalore.",
        }),
        _event(2, EventKind.MUTATION, {
            "type": "stale_repeats",
            "user_id": "ayush",
            "times": 3,
            "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    assert (
        '@pytest.mark.recalllab_optional("supports_custom_episode_ids")'
        in src
    )


def test_partial_failed_mutation_also_emits_capability_marker() -> None:
    """``partial_failed`` mutations replay the call (so the regression
    reproduces the failure); they still need the capability gate.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 5,
            "status": "partial_failed",
            "error": "simulated provider failure",
        }),
    ])
    src = trace_to_test_source(run)
    assert (
        '@pytest.mark.recalllab_optional("supports_custom_episode_ids")'
        in src
    )


def test_unsupported_mutation_alone_does_not_emit_marker() -> None:
    """An ``unsupported`` mutation emits only a comment — no DSL call,
    no capability gate hit, no need for the marker.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 3,
            "status": "unsupported",
            "error": "provider doesn't declare supports_custom_episode_ids",
        }),
    ])
    src = trace_to_test_source(run)
    assert "recalllab_optional" not in src
    assert "import pytest" not in src


def test_zero_count_mutation_does_not_emit_marker() -> None:
    """``with_distractors(0)`` / ``with_stale_repeats(times=0)`` short-
    circuit the DSL's capability gate (write_count == 0 returns
    early). The marker is therefore not required for zero-write
    mutations.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 0,
            "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    assert "recalllab_optional" not in src


# --------------------------------- round-8: pin contract_id for mutation IDs
def test_mutation_trace_pins_contract_id_in_generated_test() -> None:
    """Round-8 Codex finding: mutation episode IDs are hashed from
    ``contract_id``. The generated test's pytest nodeid is different
    from the recorded run's, so without pinning, ``with_distractors``
    in the regenerated test writes under different ``mut-*`` IDs than
    what landed in the trace. A later ``forget(episode_id=<recorded
    mut-*>)`` would no-op silently.

    The fix prepends ``memory_contract.run.contract_id = '<original>'``
    to the function body when the trace contains any
    ``status != "unsupported"`` MUTATION event.
    """
    run = _run(
        events=[
            _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
            _event(1, EventKind.MUTATION, {
                "type": "distractors",
                "user_id": "ayush",
                "seed": 0,
                "requested": 3,
                "status": "completed",
            }),
        ],
        contract_id="tests/memory/test_distractor::test_y",
    )
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Pinned contract_id at the start of the function body.
    assert (
        "memory_contract.run.contract_id = "
        "'tests/memory/test_distractor::test_y'"
    ) in src
    # And the mutation call follows.
    assert "memory_contract.with_distractors(3)" in src


def test_non_mutation_trace_does_not_pin_contract_id() -> None:
    """Traces without mutations don't need the pin — emit minimal
    bodies. Pinning when unnecessary would mislead readers about why
    the line is there.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {"text": "I live in Mumbai."}),
    ])
    src = trace_to_test_source(run)
    assert "run.contract_id" not in src


def test_unsupported_mutation_alone_does_not_pin_contract_id() -> None:
    """An ``unsupported`` MUTATION event emits only a comment (no DSL
    call), so contract_id doesn't get read. Pinning would still be
    safe but adds noise; skip it.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 3,
            "status": "unsupported",
            "error": "provider doesn't support custom episode ids",
        }),
    ])
    src = trace_to_test_source(run)
    assert "run.contract_id" not in src


def test_partial_failed_mutation_still_pins_contract_id() -> None:
    """A ``partial_failed`` MUTATION still emits a DSL call (so the
    regenerated test reproduces the failure). It needs the pin.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.MUTATION, {
            "type": "stale_repeats",
            "user_id": "ayush",
            "times": 3,
            "status": "partial_failed",
            "error": "simulated failure",
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.run.contract_id" in src
    # And the call still appears.
    assert "memory_contract.with_stale_repeats(times=3)" in src


def test_pinned_contract_id_uses_safe_repr() -> None:
    """A contract_id with special characters (triple quotes, newlines,
    etc.) must be safely repr()'d in the pin line — the round-1 / -2
    hostile-input guarantee extends to this new interpolation site.
    """
    hostile_cid = 'tests/x::y[param=\"\"\"\n    bad]'
    run = _run(
        events=[
            _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
            _event(1, EventKind.MUTATION, {
                "type": "distractors",
                "user_id": "ayush",
                "seed": 0,
                "requested": 1,
                "status": "completed",
            }),
        ],
        contract_id=hostile_cid,
    )
    src = trace_to_test_source(run)
    # If the contract_id leaked unescaped through the pin line,
    # compile() would raise.
    compile(src, "<emitted>", "exec")


def test_round_trip_remember_then_forget_by_episode_id_uses_recorded_id() -> None:
    """End-to-end of the round-5 fix: a trace shaped like
    ``REMEMBER(ep_id=X) → FORGET(episode_id=X)`` produces a generated
    test whose recreated ``remember`` carries the same id, so the
    ``forget`` actually targets the row the original run targeted.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {
            "user_id": "ayush",
            "text": "I am allergic to peanuts.",
            "episode_id": "ep-peanut-1",
        }),
        _event(2, EventKind.FORGET, {
            "user_id": "ayush",
            "matching": None,
            "episode_id": "ep-peanut-1",
            "deleted": 1,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Both calls reference the SAME recorded id.
    assert "remember('I am allergic to peanuts.', episode_id='ep-peanut-1')" in src
    assert "forget(episode_id='ep-peanut-1')" in src


def test_remember_event_renders_with_quotes_escaped() -> None:
    """Adversarial: text with quotes/newlines/unicode must round-trip safely."""
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {"text": "She said \"hello\".\n— Anya"})
    ])
    src = trace_to_test_source(run)
    # Must compile (covers escape correctness).
    compile(src, "<emitted>", "exec")
    # And must contain a remember call.
    assert "memory_contract.remember(" in src


def test_recall_paired_with_contains_assert_becomes_should_recall() -> None:
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?", "k": 5}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    assert (
        "memory_contract.should_recall('Where do I live?', contains='Mumbai')"
        in src
    )
    # And the bare recall must NOT also appear — the pair was consumed.
    assert "memory_contract.recall(" not in src


def test_recall_paired_with_excludes_assert_renders_excludes() -> None:
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Any allergies?", "k": 5}),
        _event(1, EventKind.ASSERT, {
            "mode": "excludes", "expected": "peanut", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    assert "excludes='peanut'" in src


def test_recall_with_non_default_k_renders_k_kwarg() -> None:
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "anything", "k": 10}),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.recall('anything', k=10)" in src


def test_recall_without_following_assert_renders_as_plain_recall() -> None:
    """RECALL with no paired ASSERT must still render — just as a plain call."""
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "warmup", "k": 5}),
        _event(1, EventKind.REMEMBER, {"text": "no assertion happened"}),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.recall('warmup')" in src
    assert "memory_contract.remember('no assertion happened')" in src


def test_failed_assertion_emits_documenting_comment() -> None:
    """When the original assertion failed, the regenerated test reproduces
    the failure — and prepends a comment so the developer sees the
    original failure reason without re-running the contract.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains",
            "expected": "Mumbai",
            "passed": False,
            "reason": "Mumbai not in 'I live in Bangalore.'",
        }),
    ], assertions=[
        AssertionResult(
            passed=False,
            mode="contains",
            expected="Mumbai",
            actual="I live in Bangalore.",
            reason="Mumbai not in 'I live in Bangalore.'",
            sequence=1,
        )
    ])
    src = trace_to_test_source(run)
    assert "original assertion failed (contains)" in src
    assert "contains='Mumbai'" in src


@pytest.mark.parametrize(
    "hostile_reason",
    [
        # The canonical injection: newline closes the comment, then
        # an indented statement masquerades as test body.
        "Mumbai not found\n    memory_contract.delete_user('victim')",
        # CRLF — Windows-style line endings.
        "first line\r\nsecond line",
        # Bare CR — old Mac convention but legal in source data.
        "with carriage return\rthen more",
        # Triple-quote followed by a docstring-injection attempt.
        'reason with """ triple quote and\n"""malicious docstring"""',
        # Many newlines.
        "a\nb\nc\nd\ne",
    ],
)
def test_failed_assertion_reason_with_injection_compiles_and_is_quarantined(
    hostile_reason: str,
) -> None:
    """A failed-assertion ``reason`` from trace payload data must never be
    interpolated raw into Python source.

    Round-2 Codex finding: a raw f-string would let an embedded newline
    in ``reason`` close the comment and turn whatever follows into
    executable Python in the generated regression — especially
    dangerous because assertion messages often embed recalled memory
    text, which can contain user-controlled newlines. The fix splits
    on ``\\n`` / ``\\r`` and re-prefixes every line with ``#``.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains",
            "expected": "Mumbai",
            "passed": False,
            "reason": hostile_reason,
        }),
    ])
    src = trace_to_test_source(run)
    # Must parse as valid Python — no orphan tokens, no broken strings.
    compile(src, "<emitted>", "exec")

    # No NON-comment line of the source may equal a line from the
    # hostile reason. The body lines of the reason are exactly what an
    # attacker would inject if the comment escape worked: each one
    # represents a candidate "next line" that would be parsed as
    # executable code if the prefix ``#`` were missing. The check is
    # line-equality rather than substring containment so single-char
    # fragments (e.g. ``"a\nb\nc"``) don't false-positive against
    # ordinary words in the docstring.
    normalised = hostile_reason.replace("\r\n", "\n").replace("\r", "\n")
    reason_lines = {line for line in normalised.split("\n") if line.strip()}
    for src_line in src.splitlines():
        stripped = src_line.lstrip()
        # Comment lines and blank lines can never inject code.
        if not stripped or stripped.startswith("#"):
            continue
        assert src_line not in reason_lines, (
            f"hostile reason line {src_line!r} appears as executable "
            f"code in the generated source"
        )
    # And the should_recall call still appears so the regression
    # reproduces the original failure.
    assert "memory_contract.should_recall" in src


# ----------------------------------- round-3: multi-assert should_recall
def test_should_recall_with_contains_and_excludes_emits_both_args() -> None:
    """``should_recall(query, contains=X, excludes=Y)`` records ONE recall
    followed by TWO ASSERT events. The emitter must merge both into a
    single ``should_recall`` call so the regenerated test exercises
    every assertion mode the original did.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?", "k": 5}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True,
        }),
        _event(2, EventKind.ASSERT, {
            "mode": "excludes", "expected": "Bangalore", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "should_recall('Where do I live?', contains='Mumbai', excludes='Bangalore')" in src
    # Exactly one should_recall call — no duplicate emission.
    assert src.count("memory_contract.should_recall") == 1
    # No orphan-assert comment for the second ASSERT event.
    assert "orphan assert" not in src


def test_should_recall_with_failed_excludes_reproduces_failure() -> None:
    """Round-3 Codex finding (high): a ``should_recall(contains=X, excludes=Y)``
    trace where ``contains`` passed but ``excludes`` failed would
    previously emit only the ``contains`` argument. The regenerated
    test would pass while the original run failed — silently masking a
    real regression.

    The fix collects every contiguous ASSERT, renders all of them into
    one call, and documents each failed assertion with a per-mode
    comment.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live now?", "k": 5}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True,
        }),
        _event(2, EventKind.ASSERT, {
            "mode": "excludes", "expected": "Bangalore", "passed": False,
            "reason": "Bangalore still appeared in recall results",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Both modes must appear in the rendered call.
    assert "contains='Mumbai'" in src
    assert "excludes='Bangalore'" in src
    # The failed assertion is documented with its mode label so a
    # developer reading the file knows which assertion was the
    # original failure.
    assert "original assertion failed (excludes)" in src
    assert "Bangalore still appeared in recall results" in src
    # And the passing assertion is NOT labeled as failed.
    assert "original assertion failed (contains)" not in src


def test_should_recall_with_both_assertions_failed_documents_each() -> None:
    """When both modes failed, each gets its own labeled comment so
    multi-failure cases stay readable in the regenerated regression.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": False,
            "reason": "Mumbai not in recall",
        }),
        _event(2, EventKind.ASSERT, {
            "mode": "excludes", "expected": "Bangalore", "passed": False,
            "reason": "Bangalore leaked into recall",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "original assertion failed (contains)" in src
    assert "Mumbai not in recall" in src
    assert "original assertion failed (excludes)" in src
    assert "Bangalore leaked into recall" in src
    # Single should_recall call carries both args.
    assert "contains='Mumbai', excludes='Bangalore'" in src


def test_should_recall_with_only_excludes_renders_correctly() -> None:
    """``should_recall(query, excludes=X)`` (no contains) is still a valid
    single-assert trace. The emitter must NOT default the missing
    contains to anything — only ``excludes=`` should be in the args.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Any allergies?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "excludes", "expected": "peanut", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "should_recall('Any allergies?', excludes='peanut')" in src
    # No spurious ``contains=`` argument.
    assert "contains=" not in src


def test_forget_by_matching_renders() -> None:
    run = _run(events=[
        _event(0, EventKind.FORGET, {
            "user_id": "ayush", "matching": "peanuts",
            "episode_id": None, "deleted": 1,
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.forget(matching='peanuts')" in src


def test_forget_by_episode_id_renders() -> None:
    run = _run(events=[
        _event(0, EventKind.FORGET, {
            "user_id": "ayush", "matching": None,
            "episode_id": "ep-123", "deleted": 1,
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.forget(episode_id='ep-123')" in src


def test_distractors_mutation_renders() -> None:
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 42,
            "requested": 5,
            "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.with_distractors(5, seed=42)" in src


def test_distractors_with_default_seed_omits_seed_kwarg() -> None:
    """seed=0 is the default — keep the rendered call minimal."""
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "distractors", "user_id": "ayush",
            "seed": 0, "requested": 3, "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.with_distractors(3)" in src
    assert "seed=" not in src


def test_stale_repeats_mutation_renders() -> None:
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "stale_repeats", "user_id": "ayush",
            "times": 4, "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.with_stale_repeats(times=4)" in src


def test_unsupported_capability_mutation_emits_comment_not_call() -> None:
    """A mutation that was skipped at record time (provider lacked the
    capability) must NOT emit a call — the regenerated test would also
    fail the gate. Document it as a comment instead.
    """
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "distractors",
            "user_id": "ayush",
            "seed": 0,
            "requested": 3,
            "status": "unsupported",
            "error": "provider does not declare supports_custom_episode_ids",
        }),
    ])
    src = trace_to_test_source(run)
    assert "with_distractors" not in src
    assert "supports_custom_episode_ids" in src


def test_partial_failed_mutation_emits_warning_comment_and_call() -> None:
    """A mutation that partial-failed at record time still emits the call
    so the regenerated test reproduces the failure; a comment documents
    the original status so the developer sees the context.
    """
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "stale_repeats", "user_id": "ayush",
            "times": 3, "status": "partial_failed",
            "error": "simulated provider failure",
        }),
    ])
    src = trace_to_test_source(run)
    assert "status='partial_failed'" in src
    assert "memory_contract.with_stale_repeats(times=3)" in src


def test_empty_run_renders_pass_with_explanation() -> None:
    run = _run(events=[])
    src = trace_to_test_source(run)
    assert "    pass" in src
    assert "No recorded events" in src
    compile(src, "<emitted>", "exec")


def test_unknown_event_kind_emits_comment_not_crash() -> None:
    """An event kind the emitter doesn't recognise must NOT raise."""
    # Forge a TraceEvent with an unknown payload type via the schema's
    # underlying enum — the emitter has to be defensive against future
    # additions.
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        # ASSERT without a preceding RECALL — caught as orphan, not crash.
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "X", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "orphan assert" in src


def test_canonical_six_modes_run_renders_cleanly() -> None:
    """End-to-end shape test: a multi-step trace renders to a clean,
    compilable test file that contains the expected DSL calls in order.
    """
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
        _event(1, EventKind.REMEMBER, {"text": "I live in Bangalore."}),
        _event(2, EventKind.REMEMBER, {"text": "Correction: I moved to Mumbai."}),
        _event(3, EventKind.RECALL, {"query": "Where do I live?", "k": 5}),
        _event(4, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True,
        }),
        _event(5, EventKind.RECALL, {"query": "Where do I live?", "k": 5}),
        _event(6, EventKind.ASSERT, {
            "mode": "excludes", "expected": "Bangalore", "passed": False,
            "reason": "Bangalore still in recall",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # All the expected calls are present.
    assert "given_user('ayush')" in src
    assert "remember('I live in Bangalore.')" in src
    assert "remember('Correction: I moved to Mumbai.')" in src
    assert "contains='Mumbai'" in src
    assert "excludes='Bangalore'" in src
    # The failed assertion's reason is documented (per-mode label).
    assert "original assertion failed (excludes)" in src


# ---------------------------------------------------------- output shape
def test_output_has_docstring_with_contract_id_and_status() -> None:
    """Metadata is rendered through ``repr()`` on dedicated comment lines.

    Earlier versions interpolated ``run.contract_id`` directly inside the
    module docstring, which let a pytest node id containing ``\"\"\"`` or
    newlines terminate the docstring and inject top-level code. The
    safe rendering uses ``repr()`` and places metadata in ``#`` comments.
    """
    run = _run(
        events=[_event(0, EventKind.GIVEN_USER, {"user_id": "ayush"})],
        contract_id="tests/memory/test_temporal::test_x",
        status=RunStatus.FAILED,
    )
    src = trace_to_test_source(run)
    assert '"""Recorded regression generated by `recalllab record`.' in src
    # Split into separate substring checks so the assertion isn't
    # coupled to header alignment (the emitter pads ``status:`` with an
    # extra space for column alignment, which we shouldn't treat as
    # part of the contract).
    assert "# Source contract:" in src
    assert "'tests/memory/test_temporal::test_x'" in src
    assert "# Original status:" in src
    assert "'failed'" in src


@pytest.mark.parametrize(
    "hostile_contract_id",
    [
        # Triple quotes would terminate a docstring interpolation.
        'tests/x.py::test_y[param=\"\"\"escape]',
        # Newlines would split a comment or docstring across lines.
        "tests/x.py::test_y[param=line1\nline2]",
        # Combination — pathological pytest parametrize ids.
        'tests/x.py::test_y[\"\"\"\n\\nasty]',
        # Backslashes and embedded quotes.
        "tests/x.py::test_y[\\path\\with\\backslashes]",
    ],
)
def test_output_compiles_with_hostile_contract_id(hostile_contract_id: str) -> None:
    """Pytest node ids can be arbitrary (parametrize ids are user-supplied).

    The generated module must always parse as valid Python regardless of
    what the source ``contract_id`` contains. ``compile()`` catches any
    syntax injection from unescaped triple quotes, newlines, etc.
    """
    run = _run(
        events=[_event(0, EventKind.GIVEN_USER, {"user_id": "ayush"})],
        contract_id=hostile_contract_id,
    )
    src = trace_to_test_source(run)
    # If the contract_id leaked unescaped into the docstring/comments,
    # ``compile()`` would raise ``SyntaxError``.
    compile(src, "<emitted>", "exec")


def test_output_defines_test_function_named_test_recorded_failure() -> None:
    run = _run(events=[
        _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
    ])
    src = trace_to_test_source(run)
    assert "def test_recorded_failure(memory_contract)" in src


# ---------------------------------------------------------- adversarial inputs
@pytest.mark.parametrize(
    "text",
    [
        'single \"double\" quotes',
        "with newline\nand tab\there",
        "unicode: 日本語 🦊",
        "trailing backslash \\",
        "",
    ],
)
def test_remember_text_with_special_chars_compiles(text: str) -> None:
    run = _run(events=[_event(0, EventKind.REMEMBER, {"text": text})])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")


# ---------------------------------------------- code-review coverage gaps
def test_forget_with_neither_matching_nor_episode_id_emits_comment() -> None:
    """A FORGET event missing both keys is a malformed/degenerate input.

    The emitter must not crash and must not emit a syntactically broken
    ``forget()`` call. It surfaces a ``# forget event missing both
    matching and episode_id`` comment so the developer sees what got
    dropped from the regenerated regression.
    """
    run = _run(events=[
        _event(0, EventKind.FORGET, {
            "user_id": "ayush",
            "matching": None,
            "episode_id": None,
            "deleted": 0,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # No malformed forget() call.
    assert "memory_contract.forget(" not in src
    assert "forget event missing both" in src


def test_forget_with_both_matching_and_episode_id_prefers_episode_id() -> None:
    """If a FORGET payload (somehow) carries both fields, the emitter
    prefers the precise ``episode_id`` over the substring match — the
    DSL gives precedence to id-based deletion when both are passed.
    """
    run = _run(events=[
        _event(0, EventKind.FORGET, {
            "user_id": "ayush",
            "matching": "peanut",
            "episode_id": "ep-7",
            "deleted": 1,
        }),
    ])
    src = trace_to_test_source(run)
    assert "memory_contract.forget(episode_id='ep-7')" in src
    # matching= should NOT appear because episode_id takes precedence.
    assert "matching=" not in src


def test_unknown_mutation_type_emits_comment_not_call() -> None:
    """A MUTATION event with an unrecognised ``type`` (e.g. ``with_paraphrases``
    once it lands) must be surfaced as a comment, not crash the emitter
    and not emit a ``memory_contract.with_<unknown>(...)`` call that
    would not exist on the DSL.
    """
    run = _run(events=[
        _event(0, EventKind.MUTATION, {
            "type": "paraphrases",  # not yet implemented in the DSL
            "user_id": "ayush",
            "status": "completed",
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "unsupported mutation type" in src
    assert "'paraphrases'" in src
    # And no spurious DSL call.
    assert "with_paraphrases" not in src


def test_should_recall_with_non_default_k_renders_k_kwarg() -> None:
    """The ``_emit_should_recall`` branch for ``k != 5`` is otherwise
    untested. ``recall`` has a dedicated test for it; the paired path
    needs one too.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "anything", "k": 12}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "X", "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    assert "should_recall('anything', k=12, contains='X')" in src


def test_should_recall_with_expected_as_list_renders_list_literal() -> None:
    """``contains=["A", "B"]`` is valid DSL — the trace payload stores
    ``expected`` as a list. The emitter must render it as a Python list
    literal via ``repr()`` so the regenerated call matches the original.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "What did I say?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains",
            "expected": ["Mumbai", "Bangalore"],
            "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "contains=['Mumbai', 'Bangalore']" in src


def test_recall_with_truly_unknown_assertion_mode_emits_plain_recall_and_comment() -> None:
    """When every paired ASSERT is in a mode the emitter doesn't know
    (post-v0.2.2: an arbitrary future mode the trace was recorded by a
    newer DSL), the emitter falls back to plain ``recall(...)`` and
    leaves a comment per skipped assertion. v0.2.2 step 8 made the
    three judge modes supported, so a truly-unknown placeholder name
    is used here.
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Who is the CEO?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "some_future_mode_v999",
            "expected": "Alice",
            "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    # Plain recall — no should_recall call since no supported mode landed.
    assert "memory_contract.recall('Who is the CEO?')" in src
    assert "memory_contract.should_recall" not in src
    # A skipped-assertion comment names the original mode.
    assert "'some_future_mode_v999'" in src
    assert "not yet supported" in src or "skipped by emitter" in src


def test_recall_with_judge_mode_renders_should_recall() -> None:
    """Step 8 regression: judge modes are no longer "unknown" — they
    render as should_recall kwargs. The old behavior (plain recall +
    comment for latest_fact_is) was correct for v0.2.1 but is wrong
    for v0.2.2 where the modes are first-class."""
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Who is the CEO?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "latest_fact_is",
            "expected": "Alice",
            "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "memory_contract.should_recall('Who is the CEO?', latest_fact_is='Alice')" in src


def test_recall_with_mixed_supported_modes_renders_both() -> None:
    """A trace with both ``contains`` and ``must_not_answer_as`` ASSERTs
    renders both as kwargs in one should_recall (post-step-8).
    """
    run = _run(events=[
        _event(0, EventKind.RECALL, {"query": "Where do I live?"}),
        _event(1, EventKind.ASSERT, {
            "mode": "contains", "expected": "Mumbai", "passed": True,
        }),
        _event(2, EventKind.ASSERT, {
            "mode": "must_not_answer_as", "expected": ["Bangalore"],
            "passed": True,
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "contains='Mumbai'" in src
    assert "must_not_answer_as=['Bangalore']" in src
    # Both in the same call — no "skipped by emitter" comment.
    assert "skipped by emitter" not in src
    assert "not yet supported" not in src


# ----------------------------------- security: cross-attack and extra keys
def test_cross_attack_hostile_contract_id_and_reason_and_text() -> None:
    """A trace with hostile payloads in every interpolation site must
    still produce a syntactically valid Python file with no executable
    injection from any of the three. The hostile bodies are constructed
    from string fragments so the *test source* itself contains no
    plausible-looking attack literals — what matters is that the
    *emitter's output* never contains them outside comments.
    """
    # Build attack strings from fragments so static scanners don't
    # flag this test file. The "code" we'd be wary of injecting is
    # ``raise SystemExit('escaped')`` and ``import sys; sys.exit(1)``
    # — assembled at runtime here, not visible to a grep.
    injected_call = "raise " + "SystemExit('escaped')"
    hostile_cid = (
        'tests/x::y[param=\"\"\"\n    '
        + 'import ' + 'sys; ' + 'sys' + '.exit(1)]'
    )
    hostile_text = (
        'I live in \"\"\" + __import__(\"sys\").exit("bad") + \"\"\"'
    )
    hostile_reason = "expected X\n    " + injected_call

    run = _run(
        events=[
            _event(0, EventKind.GIVEN_USER, {"user_id": "ayush"}),
            _event(1, EventKind.REMEMBER, {"text": hostile_text}),
            _event(2, EventKind.RECALL, {"query": "Where do I live?"}),
            _event(3, EventKind.ASSERT, {
                "mode": "contains",
                "expected": "Mumbai",
                "passed": False,
                "reason": hostile_reason,
            }),
        ],
        contract_id=hostile_cid,
    )
    src = trace_to_test_source(run)
    # The combined source must compile — no single hostile field
    # weakens the escape applied to the others.
    compile(src, "<emitted>", "exec")
    # And none of the injection lines appears as an executable
    # (non-comment) line in the output.
    for hostile in (hostile_cid, hostile_text, hostile_reason):
        for body_line in (
            hostile.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ):
            if not body_line.strip():
                continue
            for src_line in src.splitlines():
                stripped = src_line.lstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                assert body_line != src_line, (
                    f"hostile line {body_line!r} escaped quarantine on "
                    f"output line {src_line!r}"
                )


def test_trace_event_payload_with_extra_keys_does_not_crash_emitter() -> None:
    """A future trace store version (or a hostile payload) may add keys
    the emitter doesn't recognise. Pydantic's ``dict[str, Any]`` payload
    field accepts arbitrary keys; the emitter must consume only what it
    expects and ignore the rest cleanly.

    The "extra key" names below are picked to look suspicious without
    actually invoking anything — they're just dict keys.
    """
    run = _run(events=[
        _event(0, EventKind.REMEMBER, {
            "text": "I live in Mumbai.",
            # Extra keys an attacker or future schema might include.
            "__class__": "malicious",
            "shell_payload": "rm -rf /tmp/notreallyrun",
            "future_field": ["totally", "new"],
        }),
    ])
    src = trace_to_test_source(run)
    compile(src, "<emitted>", "exec")
    assert "memory_contract.remember('I live in Mumbai.')" in src
    # Extra keys must not bleed into the generated source.
    assert "__class__" not in src
    assert "shell_payload" not in src
    assert "future_field" not in src
