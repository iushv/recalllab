"""Integration tests for ``recalllab record``.

Drives the CLI command end-to-end:

1. Build a ``ContractRun`` in code (the kind a real failed pytest would
   produce) and write it through ``TraceStore``.
2. Invoke ``cmd_record`` with ``--latest-failure`` and ``--run-id``.
3. Read the emitted file and verify it compiles to valid Python *and*
   contains the expected DSL calls.

Also covers the error paths: missing trace store, missing run id, no
failed runs to select.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from recalllab.cli.record import cmd_record
from recalllab.core.traces.schema import (
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)
from recalllab.core.traces.sqlite_store import TraceStore


def _failed_temporal_run() -> ContractRun:
    """A realistic failed-temporal-update run, like
    ``test_updated_location_overrides_stale_memory`` would produce when
    the agent forgot to apply the correction.
    """
    return ContractRun(
        id=uuid4().hex,
        contract_id="tests/memory/test_temporal::test_updated_location",
        provider="reference",
        started_at=datetime(2026, 5, 18, 12, 0, tzinfo=UTC),
        status=RunStatus.FAILED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.GIVEN_USER,
                payload={"user_id": "ayush"},
                timestamp=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "ayush",
                    "text": "I live in Bangalore.",
                    "episode_id": "ep-1",
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 1, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=2,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "ayush",
                    "text": "Correction: I moved to Mumbai.",
                    "episode_id": "ep-2",
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 2, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=3,
                kind=EventKind.RECALL,
                payload={
                    "user_id": "ayush",
                    "query": "Where do I live?",
                    "k": 5,
                    "results": [],
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 3, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=4,
                kind=EventKind.ASSERT,
                payload={
                    "mode": "contains",
                    "expected": "Mumbai",
                    "passed": False,
                    "reason": "Mumbai not in 'I live in Bangalore.'",
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 4, tzinfo=UTC),
            ),
        ],
    )


def _passed_run() -> ContractRun:
    return ContractRun(
        id=uuid4().hex,
        contract_id="tests/memory/test_cross_session::test_birthday",
        provider="reference",
        started_at=datetime(2026, 5, 18, 11, 0, tzinfo=UTC),
        status=RunStatus.PASSED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.GIVEN_USER,
                payload={"user_id": "alice"},
                timestamp=datetime(2026, 5, 18, 11, 0, tzinfo=UTC),
            ),
        ],
    )


def test_record_run_by_id_writes_compilable_file(tmp_path: Path) -> None:
    """Happy path: --run-id picks the requested run and writes a regression."""
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    out = tmp_path / "tests" / "regressions" / "test_recorded.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    assert out.exists()
    src = out.read_text()
    compile(src, str(out), "exec")
    assert "test_recorded_failure" in src
    assert "I live in Bangalore." in src
    assert "Correction: I moved to Mumbai." in src
    assert "contains='Mumbai'" in src
    assert "original assertion failed" in src


def test_record_latest_failure_picks_failed_run_over_passed(
    tmp_path: Path,
) -> None:
    """--latest-failure must skip recently-passed runs and pick the most
    recent FAILED one.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    failed = _failed_temporal_run()
    passed = _passed_run()
    store.write_run(failed)
    # passed.started_at is earlier; list_runs orders by started_at DESC,
    # so failed sits first regardless. Write both anyway so the test
    # documents the filter behavior, not the ordering.
    store.write_run(passed)

    out = tmp_path / "out.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=None,
        latest_failure=True,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # The output must reference the FAILED contract, not the passed one.
    assert "test_updated_location" in src
    assert "test_birthday" not in src


def test_record_latest_failure_finds_old_failed_run_buried_under_many_passes(
    tmp_path: Path,
) -> None:
    """``--latest-failure`` must work regardless of how many passed runs
    have accumulated since the failure.

    Earlier this paginate-and-filter implementation only scanned
    ``list_runs(limit=200)``: if a trace store had >200 newer passed
    runs after the most recent failure, the command reported "no failed
    runs" while the failure was still present in the database. CI
    environments easily reach that scale. The fix is a dedicated
    indexed query.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)

    # The failed run is *oldest* — started_at far in the past.
    failed = _failed_temporal_run().model_copy(
        update={"started_at": datetime(2020, 1, 1, tzinfo=UTC)}
    )
    store.write_run(failed)

    # Then 250 passed runs, all newer. Each needs a unique id and a
    # progressively newer timestamp so the indexed ORDER BY is honest.
    for i in range(250):
        passed = _passed_run().model_copy(
            update={
                "id": uuid4().hex,
                "started_at": datetime(2026, 5, 18, 12, i // 60, i % 60, tzinfo=UTC),
            }
        )
        store.write_run(passed)

    out = tmp_path / "out.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=None,
        latest_failure=True,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # The buried failure was still found.
    assert "test_updated_location" in src
    assert "test_birthday" not in src


def test_record_unknown_run_id_returns_error(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.sqlite"
    TraceStore(trace_path)  # initialise
    out = tmp_path / "out.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id="nonexistent",
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 1
    assert not out.exists()


def test_record_no_failures_in_store_returns_error(tmp_path: Path) -> None:
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    store.write_run(_passed_run())
    out = tmp_path / "out.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=None,
        latest_failure=True,
        out_path=out,
    )
    assert exit_code == 1
    assert not out.exists()


def test_record_missing_trace_store_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.py"
    exit_code = cmd_record(
        trace_path=tmp_path / "does-not-exist.sqlite",
        run_id="anything",
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Trace store not found" in captured.out


def test_record_output_is_byte_stable_across_repeated_calls(
    tmp_path: Path,
) -> None:
    """Round-trip stability: re-running the same record command produces
    identical bytes (this is the property that makes the regression
    file safe to commit and re-record on every CI failure).
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)
    out = tmp_path / "out.py"

    cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    first = out.read_text()
    # Second invocation needs --force because the output now exists;
    # the byte-stability guarantee only matters for legitimate re-record
    # workflows where the user opts in to overwrite.
    cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
        force=True,
    )
    second = out.read_text()
    assert first == second


# ---------------------------------------------------- round-4: overwrite guard
def test_record_refuses_to_overwrite_existing_file_by_default(
    tmp_path: Path,
) -> None:
    """A typo in ``--out`` or a reused path must NOT silently destroy
    checked-in code.

    Round-4 Codex finding: ``cmd_record`` unconditionally wrote to the
    user-supplied path, so any existing file at that path got
    overwritten without warning. For a command marketed as "fast path
    from CI failure to committed regression", that's a real
    foot-cannon. The fix refuses by default and requires ``--force``.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    out = tmp_path / "existing_test.py"
    original_content = "# this is a real checked-in test\nimport pytest\n"
    out.write_text(original_content)

    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 1
    # Original file must be untouched.
    assert out.read_text() == original_content


def test_record_force_overwrites_existing_file(tmp_path: Path) -> None:
    """With ``--force``, the user opts in to overwriting the destination.

    The overwrite goes through ``_atomic_write_text`` so a crash
    mid-write cannot leave a half-written file behind.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    out = tmp_path / "regression.py"
    out.write_text("# stale content\n")

    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
        force=True,
    )
    assert exit_code == 0
    assert "stale content" not in out.read_text()
    assert "test_recorded_failure" in out.read_text()


def test_record_writes_fresh_file_without_force_when_target_does_not_exist(
    tmp_path: Path,
) -> None:
    """The guard only fires when the target exists. Fresh paths still
    work without ``--force`` — the happy path is preserved.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    out = tmp_path / "fresh" / "regression.py"  # parent doesn't exist either
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    assert out.exists()
    assert "test_recorded_failure" in out.read_text()


def test_record_atomic_write_leaves_no_temp_file_on_success(
    tmp_path: Path,
) -> None:
    """After a successful write, no ``.tmp`` siblings should remain.

    The atomic-write helper creates a temp file via ``mkstemp`` in the
    same directory, then ``os.replace`` renames it onto the
    destination. On success the temp file no longer exists; on failure
    the ``except`` branch unlinks it. Either way the directory is
    clean.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    out = tmp_path / "regression.py"
    cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    siblings = list(tmp_path.iterdir())
    tmp_files = [p for p in siblings if p.name.startswith(".") and ".tmp" in p.name]
    assert tmp_files == [], (
        f"atomic-write left temp file(s) behind: {tmp_files}"
    )


def test_record_atomic_write_cleans_up_temp_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``os.replace`` fails mid-write, the temp file must be unlinked
    so the directory stays clean.

    Code-review finding (item 10): the success path is covered by the
    sibling test, but the ``except`` branch in ``_atomic_write_text``
    (which calls ``tmp_path.unlink(missing_ok=True)`` on any exception
    before re-raising) is otherwise unexercised. Patching ``os.replace``
    to raise simulates a real failure mode: disk full, permission
    denied between mkstemp and rename, kernel signal interruption.
    """
    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)
    run = _failed_temporal_run()
    store.write_run(run)

    # Force the rename step to fail. The write to the temp file should
    # succeed; the replace step raises. The except branch must clean up
    # the temp file before propagating the exception.
    def _explode(src: object, dst: object) -> None:
        del src, dst  # signature compat, not used
        raise OSError("simulated disk-full at rename")

    monkeypatch.setattr("recalllab.cli.record.os.replace", _explode)

    out = tmp_path / "regression.py"
    with pytest.raises(OSError, match="simulated disk-full"):
        cmd_record(
            trace_path=trace_path,
            run_id=run.id,
            latest_failure=False,
            out_path=out,
        )

    # Output file was never created (replace failed before it could
    # land).
    assert not out.exists()
    # Crucially: no orphan ``.tmp`` siblings left behind.
    siblings = list(tmp_path.iterdir())
    tmp_files = [
        p for p in siblings
        if p.name.startswith(".") and ".tmp" in p.name
    ]
    assert tmp_files == [], (
        f"atomic-write FAILURE path left temp file(s): {tmp_files}"
    )


def test_record_generated_test_mutation_then_forget_by_recorded_id_actually_deletes(
    tmp_path: Path,
) -> None:
    """Round-8 Codex finding (runtime): a trace with
    ``with_distractors(...)`` followed by ``forget(episode_id=<recorded
    mut-*>)`` must produce a regression where the forget actually
    deletes the row in the regenerated run.

    Pre-fix the regenerated mutation hashed the generated test's
    pytest nodeid into the mutation IDs, so the forget targeted IDs
    that never existed in the regenerated run — silent no-op. The
    contract_id pin makes mutation IDs deterministic across the
    original and regenerated runs.
    """
    import runpy

    from recalllab.adapters.reference import ReferenceMemoryAdapter
    from recalllab.core.contract.dsl import MemoryContract

    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)

    # Build the trace synthetically: given_user, with_distractors,
    # forget by the recorded mut-* id. We have to compute the recorded
    # id by running with_distractors against a real adapter first so
    # we know what mutation IDs would have been written at trace time.
    seed_adapter = ReferenceMemoryAdapter()
    try:
        seed_run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/memory/test_mutation::test_x",
            provider="reference",
            started_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        )
        seed_contract = MemoryContract(seed_adapter, seed_run)
        seed_contract.given_user("ayush")
        seed_contract.with_distractors(2, seed=0)
        # Capture the mutation IDs that landed at "trace time".
        seed_episodes = seed_adapter.list_episodes("ayush")
        recorded_mut_ids = sorted(
            ep.id for ep in seed_episodes if ep.id.startswith("mut-")
        )
        assert len(recorded_mut_ids) == 2
        target_id = recorded_mut_ids[0]
    finally:
        seed_adapter.close()

    # Now build the trace as if a real RecallLab run had produced it:
    # given_user, mutation, forget(episode_id=target_id).
    real_run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/memory/test_mutation::test_x",
        provider="reference",
        started_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        status=RunStatus.FAILED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.GIVEN_USER,
                payload={"user_id": "ayush"},
                timestamp=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.MUTATION,
                payload={
                    "type": "distractors",
                    "user_id": "ayush",
                    "seed": 0,
                    "requested": 2,
                    "invocation": 1,
                    "requested_episode_ids": recorded_mut_ids,
                    "inserted_episode_ids": recorded_mut_ids,
                    "unconfirmed_writes": [],
                    "abandoned_in_flight": [],
                    "status": "completed",
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 1, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=2,
                kind=EventKind.FORGET,
                payload={
                    "user_id": "ayush",
                    "matching": None,
                    "episode_id": target_id,
                    "deleted": 1,
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 2, tzinfo=UTC),
            ),
        ],
    )
    store.write_run(real_run)

    out = tmp_path / "tests" / "test_recorded.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=real_run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # The pin line is present.
    assert (
        "memory_contract.run.contract_id = "
        "'tests/memory/test_mutation::test_x'"
    ) in src
    # The forget references the recorded mutation id.
    assert f"forget(episode_id={target_id!r})" in src

    # Now actually run the generated file against a fresh adapter and
    # assert the forget removed the targeted row. Pre-fix this would
    # leave 2 distractor rows because the mutation wrote under
    # different IDs than the forget targeted.
    ns = runpy.run_path(str(out))
    test_fn = ns["test_recorded_failure"]
    assert callable(test_fn)

    adapter = ReferenceMemoryAdapter()
    try:
        run_again = ContractRun(
            id=uuid4().hex,
            contract_id="exec::pinned",  # deliberately different from recorded
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run_again)
        test_fn(contract)

        # End state: 1 distractor remains (we forgot one of the 2 that
        # landed). The forget targeted the recorded id and actually
        # found a matching row because the contract_id pin made the
        # regenerated mutation write under the same id.
        remaining = adapter.list_episodes("ayush")
        assert len(remaining) == 1, (
            f"expected 1 remaining distractor after forget, got "
            f"{len(remaining)} — forget did not target the recorded "
            f"mutation id (pin failed)"
        )
        # And the surviving row is NOT the one we forgot.
        assert remaining[0].id != target_id
        assert remaining[0].id.startswith("mut-distractors-")
    finally:
        adapter.close()


def test_record_generated_test_handles_mid_trace_user_switch(
    tmp_path: Path,
) -> None:
    """Round-10 Codex finding (runtime): a trace shaped like
    ``GIVEN_USER alice → REMEMBER alice → REMEMBER bob`` (no
    intervening GIVEN_USER for bob) must produce a regression where
    bob's fact lands in bob's namespace, not alice's.

    Pre-fix this would have stored bob's text under alice and bob's
    namespace would have been empty — the regression silently
    exercised the wrong tenant.
    """
    import runpy

    from recalllab.adapters.reference import ReferenceMemoryAdapter
    from recalllab.core.contract.dsl import MemoryContract

    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)

    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/exported::test_multi_user_partial",
        provider="reference",
        started_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        status=RunStatus.PASSED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.GIVEN_USER,
                payload={"user_id": "alice"},
                timestamp=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "alice",
                    "text": "Project codename: Aurora.",
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 1, tzinfo=UTC),
            ),
            # No GIVEN_USER bob in the trace — but bob's REMEMBER lands.
            TraceEvent(
                sequence=2,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "bob",
                    "text": "I prefer espresso in the afternoon.",
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 2, tzinfo=UTC),
            ),
        ],
    )
    store.write_run(run)

    out = tmp_path / "tests" / "test_recorded.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # Synthesised given_user for bob's switch is present.
    assert "memory_contract.given_user('bob')" in src
    assert "trace switched user_id without an intervening GIVEN_USER" in src

    # And running the generated test against a fresh adapter must land
    # bob's fact in bob's namespace, not alice's. Pre-fix this would
    # have stored under alice.
    ns = runpy.run_path(str(out))
    test_fn = ns["test_recorded_failure"]

    adapter = ReferenceMemoryAdapter()
    try:
        run_again = ContractRun(
            id=uuid4().hex,
            contract_id="exec::multi_user",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run_again)
        test_fn(contract)

        # Alice has only her own fact.
        alice_eps = adapter.list_episodes("alice")
        assert len(alice_eps) == 1
        assert alice_eps[0].text == "Project codename: Aurora."

        # Bob has only his own fact — landed correctly under bob's
        # tenant. Pre-fix this would have been empty (bob's text
        # stored under alice).
        bob_eps = adapter.list_episodes("bob")
        assert len(bob_eps) == 1, (
            f"bob's namespace has {len(bob_eps)} episode(s); pre-fix "
            "the regenerated test would have stored bob's text under "
            "alice, leaving bob empty"
        )
        assert bob_eps[0].text == "I prefer espresso in the afternoon."
    finally:
        adapter.close()


def test_record_generated_test_runs_for_trace_missing_given_user(
    tmp_path: Path,
) -> None:
    """Round-7 Codex finding (runtime): a trace whose first event is
    REMEMBER (no GIVEN_USER) must produce a generated test that, when
    loaded via runpy and executed against a real adapter, actually
    runs to the recorded behaviour — not crash with
    ``RuntimeError("no active user")`` before reaching the assertion.

    The synthesised ``given_user(payload['user_id'])`` is the load-
    bearing line. Without it the regression is unusable for the entire
    class of externally-sourced / partial-dump traces this finding
    flagged.
    """
    import runpy

    from recalllab.adapters.reference import ReferenceMemoryAdapter
    from recalllab.core.contract.dsl import MemoryContract

    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)

    # Hand-build a non-canonical trace: REMEMBER + RECALL + ASSERT, no
    # GIVEN_USER. A RecallLab-produced trace can't have this shape
    # (the DSL would have raised at trace time), but exports / migrations
    # can.
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/exported::test_no_given_user",
        provider="reference",
        started_at=datetime(2026, 5, 19, 12, 0, tzinfo=UTC),
        status=RunStatus.PASSED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "ayush",
                    "text": "I live in Mumbai.",
                    "episode_id": "ep-1",
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.RECALL,
                payload={
                    "user_id": "ayush",
                    "query": "Where do I live?",
                    "k": 5,
                    "results": [
                        {"text": "I live in Mumbai.", "episode_id": "ep-1"}
                    ],
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 1, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=2,
                kind=EventKind.ASSERT,
                payload={
                    "mode": "contains",
                    "expected": "Mumbai",
                    "passed": True,
                },
                timestamp=datetime(2026, 5, 19, 12, 0, 2, tzinfo=UTC),
            ),
        ],
    )
    store.write_run(run)

    out = tmp_path / "tests" / "test_recorded.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # The synthesised given_user line is present.
    assert "memory_contract.given_user('ayush')" in src
    assert "synthesized by `recalllab record`" in src

    # And the generated test actually runs against a fresh adapter
    # without crashing on _require_user. The recall must find the
    # remembered text — that's the recorded contract behaviour.
    ns = runpy.run_path(str(out))
    test_fn = ns["test_recorded_failure"]
    assert callable(test_fn)

    adapter = ReferenceMemoryAdapter()
    try:
        run_again = ContractRun(
            id=uuid4().hex,
            contract_id="exec::synthesized_user",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run_again)
        # If synthesis is broken this raises RuntimeError("no active
        # user"). If synthesis works, the test runs to completion.
        test_fn(contract)

        # The remember landed under the correct user.
        eps = adapter.list_episodes("ayush")
        assert len(eps) == 1
        assert eps[0].text == "I live in Mumbai."
        assert eps[0].id == "ep-1"
    finally:
        adapter.close()


def test_record_generated_test_id_paired_forget_actually_deletes(
    tmp_path: Path,
) -> None:
    """End-to-end of round-5: a trace where ``forget`` references the
    ``remember``'s recorded episode_id must produce a generated test
    that, when loaded and executed against a fresh adapter, deletes
    the same row.

    The bug was: the emitter dropped ``episode_id`` from the
    REMEMBER payload, so the regenerated ``remember`` got a fresh
    uuid and the subsequent ``forget(episode_id=X)`` silently deleted
    nothing. The regression here builds a real trace, runs
    ``recalllab record``, then loads the generated file via
    ``runpy.run_path`` (stdlib) and runs its test function against a
    real ``ReferenceMemoryAdapter``, asserting the forget actually
    swept.
    """
    import runpy

    from recalllab.adapters.reference import ReferenceMemoryAdapter
    from recalllab.core.contract.dsl import MemoryContract

    trace_path = tmp_path / "traces.sqlite"
    store = TraceStore(trace_path)

    # Build a synthetic trace shaped like a real allergy-then-forget run.
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/memory/test_forget::test_allergy",
        provider="reference",
        started_at=datetime(2026, 5, 18, 12, 0, tzinfo=UTC),
        status=RunStatus.FAILED,
        events=[
            TraceEvent(
                sequence=0,
                kind=EventKind.GIVEN_USER,
                payload={"user_id": "ayush"},
                timestamp=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=1,
                kind=EventKind.REMEMBER,
                payload={
                    "user_id": "ayush",
                    "text": "I am allergic to peanuts.",
                    "episode_id": "ep-peanut-1",
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 1, tzinfo=UTC),
            ),
            TraceEvent(
                sequence=2,
                kind=EventKind.FORGET,
                payload={
                    "user_id": "ayush",
                    "matching": None,
                    "episode_id": "ep-peanut-1",
                    "deleted": 1,
                },
                timestamp=datetime(2026, 5, 18, 12, 0, 2, tzinfo=UTC),
            ),
        ],
    )
    store.write_run(run)

    out = tmp_path / "tests" / "test_recorded.py"
    exit_code = cmd_record(
        trace_path=trace_path,
        run_id=run.id,
        latest_failure=False,
        out_path=out,
    )
    assert exit_code == 0
    src = out.read_text()
    # The emitter must include the recorded id in the remember call.
    assert "episode_id='ep-peanut-1'" in src

    # Load the generated module via runpy (stdlib's safe path) and drive
    # its test function against a fresh adapter. Confirm the forget
    # actually removes the row — pre-fix this would have left the
    # allergy memory in place because the regenerated remember had a
    # fresh uuid and the forget targeted 'ep-peanut-1' (which never
    # existed in the regenerated run).
    ns = runpy.run_path(str(out))
    test_fn = ns["test_recorded_failure"]
    assert callable(test_fn)

    adapter = ReferenceMemoryAdapter()
    try:
        run_again = ContractRun(
            id=uuid4().hex,
            contract_id="exec::generated",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run_again)
        test_fn(contract)

        # End-state assertion: zero episodes remain for ayush.
        remaining = adapter.list_episodes("ayush")
        assert remaining == [], (
            f"regenerated test left {len(remaining)} row(s) behind — "
            "forget did not target the same id as the remember"
        )
    finally:
        adapter.close()
