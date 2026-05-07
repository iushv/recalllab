"""Mutation tests for v0.2.0.

Pins seven properties of the new mutations (the original four plus three
that lock in the Codex adversarial-review fixes for seed-default
determinism, deterministic episode IDs, and partial-failure tracing):

1. ``sample_distractors(n, seed=...)`` is deterministic — same pair always
   produces the same list, in the same order. Default ``seed=0`` is
   reproducible across calls.
2. ``with_distractors`` and ``with_stale_repeats`` each emit a ``MUTATION``
   trace event with the v0.2.0 payload schema (``type``, ``status``,
   ``inserted_episode_ids``, mutation-specific keys).
3. The existing six example contracts still pass (regression check).
4. Mutations are scoped to the active user.
5. Mutation episode IDs are deterministic from ``contract_id`` + mutation
   type + key + index, so retries against hosted providers are idempotent.
6. Mid-mutation provider exceptions record ``status="partial_failed"`` plus
   the partial ``inserted_episode_ids`` and ``error`` repr, then re-raise
   so the test fails loudly.
7. ``with_distractors`` defaults to ``seed=0`` (no silent system-entropy
   nondeterminism).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from recalllab.adapters.base import CapabilityFlags, Episode
from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.mutations import DISTRACTOR_POOL, sample_distractors
from recalllab.core.traces.schema import ContractRun, EventKind


# ---------------------------------------------------------------- property #1
def test_sample_distractors_is_deterministic() -> None:
    a = sample_distractors(10, seed=42)
    b = sample_distractors(10, seed=42)
    assert a == b
    assert len(a) == 10
    assert all(item in DISTRACTOR_POOL for item in a)


def test_sample_distractors_different_seeds_disagree() -> None:
    assert sample_distractors(10, seed=1) != sample_distractors(10, seed=2)


def test_sample_distractors_pads_with_replacement_when_n_exceeds_pool() -> None:
    n = len(DISTRACTOR_POOL) + 5
    sampled = sample_distractors(n, seed=7)
    assert len(sampled) == n


def test_sample_distractors_zero_returns_empty() -> None:
    assert sample_distractors(0, seed=0) == []


def test_sample_distractors_rejects_negative_n() -> None:
    with pytest.raises(ValueError):
        sample_distractors(-1)


def test_sample_distractors_default_seed_is_reproducible() -> None:
    """Default seed=0 must yield the same list on every call (no system entropy)."""
    a = sample_distractors(8)
    b = sample_distractors(8)
    assert a == b
    # Must also equal the explicit seed=0 case.
    assert a == sample_distractors(8, seed=0)


# ---------------------------------------------------------------- helpers
def _new_contract(
    contract_id: str = "unit::mutation",
) -> tuple[MemoryContract, ReferenceMemoryAdapter, ContractRun]:
    adapter = ReferenceMemoryAdapter()
    run = ContractRun(
        id=uuid4().hex,
        contract_id=contract_id,
        provider="reference",
        started_at=datetime.now(tz=UTC),
    )
    return MemoryContract(adapter, run), adapter, run


# ---------------------------------------------------------------- property #2
def test_with_distractors_records_mutation_trace_event() -> None:
    contract, adapter, run = _new_contract()
    try:
        contract.given_user("ayush")
        contract.with_distractors(5, seed=42)
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 1
        ev = mutation_events[0]
        assert ev.payload["type"] == "distractors"
        assert ev.payload["requested"] == 5
        assert ev.payload["seed"] == 42
        assert ev.payload["user_id"] == "ayush"
        assert ev.payload["status"] == "completed"
        assert isinstance(ev.payload["inserted_episode_ids"], list)
        assert len(ev.payload["inserted_episode_ids"]) == 5
        assert "error" not in ev.payload
    finally:
        adapter.close()


def test_with_stale_repeats_records_mutation_trace_event() -> None:
    contract, adapter, run = _new_contract()
    try:
        contract.given_user("ayush")
        seeded = contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=3)
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 1
        ev = mutation_events[0]
        assert ev.payload["type"] == "stale_repeats"
        assert ev.payload["times"] == 3
        assert ev.payload["user_id"] == "ayush"
        assert ev.payload["status"] == "completed"
        assert ev.payload["source_episode_id"] == seeded.id
        assert len(ev.payload["inserted_episode_ids"]) == 3
        assert "error" not in ev.payload
    finally:
        adapter.close()


def test_with_stale_repeats_actually_duplicates_in_namespace() -> None:
    contract, adapter, _run = _new_contract()
    try:
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=4)
        episodes = adapter.list_episodes("ayush")
        bangalore_count = sum(
            1 for ep in episodes if ep.text == "I live in Bangalore."
        )
        # 1 original + 4 stale repeats == 5 copies
        assert bangalore_count == 5
    finally:
        adapter.close()


def test_with_stale_repeats_without_prior_remember_raises() -> None:
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        with pytest.raises(RuntimeError, match=r"prior remember"):
            contract.with_stale_repeats(times=2)
    finally:
        adapter.close()


# ---------------------------------------------------------------- property #3
def test_existing_six_example_contracts_still_pass() -> None:
    """Smoke regression: run the six v0.1 examples and confirm green."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples" / "tests"
    assert examples_dir.exists(), f"examples not found at {examples_dir}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(examples_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"v0.1 examples regressed under v0.2 changes:\n"
        f"stdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    # Sanity: confirm we actually ran the six contracts
    assert "6 passed" in result.stdout, result.stdout


# ---------------------------------------------------------------- property #4
def test_with_distractors_does_not_leak_across_users() -> None:
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("alice")
        contract.with_distractors(20, seed=42)
        # alice has 20 distractors; bob has none
        alice_eps = adapter.list_episodes("alice")
        bob_eps = adapter.list_episodes("bob")
        assert len(alice_eps) == 20
        assert bob_eps == []
        # And recall for bob returns nothing
        bob_recalls = adapter.recall("bob", "coffee", k=10)
        assert bob_recalls == []
    finally:
        adapter.close()


def test_with_stale_repeats_does_not_leak_across_users() -> None:
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("alice")
        contract.remember("Project codename: Aurora.")
        contract.with_stale_repeats(times=5)
        # Switch to bob — must not see any of alice's repeats
        contract.given_user("bob")
        bob_eps = adapter.list_episodes("bob")
        assert bob_eps == []
        # And alice must have 1 + 5 = 6 copies of the seeded text
        alice_eps = adapter.list_episodes("alice")
        assert len(alice_eps) == 6
        assert all(ep.text == "Project codename: Aurora." for ep in alice_eps)
    finally:
        adapter.close()


# ---------------------------------------------------------------- property #5
def test_with_distractors_episode_ids_are_deterministic_across_runs() -> None:
    """Same contract_id + seed must request the same episode IDs every run.

    The reference adapter honours the requested ``episode_id``, so this is
    a real idempotency guarantee for retried contracts against any adapter
    that respects custom IDs.
    """
    ids_run_a: list[str] = []
    ids_run_b: list[str] = []
    for ids in (ids_run_a, ids_run_b):
        contract, adapter, _ = _new_contract(contract_id="tests/x.py::test_y")
        try:
            contract.given_user("ayush")
            contract.with_distractors(5, seed=42)
            ids[:] = [ep.id for ep in adapter.list_episodes("ayush")]
        finally:
            adapter.close()
    assert ids_run_a == ids_run_b
    assert all(eid.startswith("mut-distractors-") for eid in ids_run_a)


def test_with_stale_repeats_episode_ids_are_deterministic_across_runs() -> None:
    """Same contract_id + source text must request the same repeat IDs every run."""
    ids_a: list[str] = []
    ids_b: list[str] = []
    for ids in (ids_a, ids_b):
        contract, adapter, _ = _new_contract(contract_id="tests/x.py::test_temporal")
        try:
            contract.given_user("ayush")
            contract.remember("I live in Bangalore.")
            contract.with_stale_repeats(times=3)
            # Filter to the deterministic repeat IDs only (the original
            # ``remember`` got a uuid-based id).
            repeat_ids = [
                ep.id
                for ep in adapter.list_episodes("ayush")
                if ep.id.startswith("mut-stale_repeats-")
            ]
            ids[:] = repeat_ids
        finally:
            adapter.close()
    assert ids_a == ids_b
    assert len(ids_a) == 3


# ---------------------------------------------------------------- property #6
class _FailingAfterNProvider:
    """Stub MemoryProvider that errors on the (n+1)-th remember call."""

    def __init__(self, fail_after: int) -> None:
        self._fail_after = fail_after
        self._calls = 0
        self._episodes: list[Episode] = []

    def capabilities(self) -> CapabilityFlags:
        # Declare custom-id support so the mutation capability gate doesn't
        # short-circuit before we get to exercise the partial-failure path
        # this stub exists to test.
        return CapabilityFlags(
            supports_forget=True,
            supports_provenance=True,
            supports_custom_episode_ids=True,
        )

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        if self._calls >= self._fail_after:
            raise RuntimeError("simulated provider failure")
        eid = episode_id or f"stub-{self._calls}"
        episode = Episode(
            id=eid,
            user_id=user_id,
            text=text,
            created_at=datetime.now(tz=UTC),
            metadata=metadata,
        )
        self._episodes.append(episode)
        self._calls += 1
        return episode

    # Unused parts of the protocol for this test.
    def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Any]:
        return []

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        return 0

    def list_episodes(self, user_id: str) -> list[Episode]:
        return [ep for ep in self._episodes if ep.user_id == user_id]

    def delete_user(self, user_id: str) -> None:
        self._episodes = [ep for ep in self._episodes if ep.user_id != user_id]


def test_with_distractors_partial_failure_records_status_and_reraises() -> None:
    provider = _FailingAfterNProvider(fail_after=3)
    run = ContractRun(
        id=uuid4().hex,
        contract_id="unit::partial_fail",
        provider="stub",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    with pytest.raises(RuntimeError, match=r"simulated provider failure"):
        contract.with_distractors(10, seed=0)
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    payload = mutation_events[0].payload
    assert payload["status"] == "partial_failed"
    assert payload["requested"] == 10
    # Provider succeeded on the first 3 inserts, then raised on the 4th.
    assert len(payload["inserted_episode_ids"]) == 3
    assert "error" in payload
    assert "simulated provider failure" in payload["error"]


def test_with_stale_repeats_partial_failure_records_status_and_reraises() -> None:
    provider = _FailingAfterNProvider(fail_after=2)
    run = ContractRun(
        id=uuid4().hex,
        contract_id="unit::partial_fail_repeats",
        provider="stub",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    # Seed remember (consumes 1 of fail_after=2).
    contract.remember("seed text")
    # Now with_stale_repeats(times=5) should succeed once (fill fail_after to 2)
    # then fail on the second iteration.
    with pytest.raises(RuntimeError, match=r"simulated provider failure"):
        contract.with_stale_repeats(times=5)
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    payload = mutation_events[0].payload
    assert payload["status"] == "partial_failed"
    assert len(payload["inserted_episode_ids"]) == 1
    assert "error" in payload


# ---------------------------------------------------------------- property #7
def test_with_distractors_default_seed_is_zero() -> None:
    """Two contracts with no explicit seed must request identical IDs and texts."""
    seen_a: list[tuple[str, str]] = []
    seen_b: list[tuple[str, str]] = []
    for seen in (seen_a, seen_b):
        contract, adapter, _ = _new_contract(contract_id="tests/x.py::test_default_seed")
        try:
            contract.given_user("ayush")
            contract.with_distractors(7)
            seen[:] = [(ep.id, ep.text) for ep in adapter.list_episodes("ayush")]
        finally:
            adapter.close()
    assert seen_a == seen_b


# ------- round-2 P1 #1: same seed under two users must NOT collide
def test_with_distractors_same_seed_two_users_in_one_contract_does_not_collide() -> None:
    """Two users invoking the same mutation with the same seed must get distinct IDs."""
    contract, adapter, _ = _new_contract(contract_id="tests/x.py::test_two_users")
    try:
        contract.given_user("alice").with_distractors(1, seed=0)
        contract.given_user("bob").with_distractors(1, seed=0)
        alice_eps = adapter.list_episodes("alice")
        bob_eps = adapter.list_episodes("bob")
        assert len(alice_eps) == 1
        assert len(bob_eps) == 1
        # Distinct IDs, even though the seed is identical.
        assert alice_eps[0].id != bob_eps[0].id
    finally:
        adapter.close()


def test_repeated_mutation_calls_for_same_user_do_not_collide() -> None:
    """Calling the same mutation twice for one user must invocate distinct IDs."""
    contract, adapter, _ = _new_contract(contract_id="tests/x.py::test_repeated")
    try:
        contract.given_user("ayush")
        contract.with_distractors(2, seed=0)  # invocation=1
        contract.with_distractors(2, seed=0)  # invocation=2 — must NOT collide
        eps = adapter.list_episodes("ayush")
        # 2 + 2 = 4 distractors, all distinct ids
        assert len(eps) == 4
        assert len({ep.id for ep in eps}) == 4
    finally:
        adapter.close()


# ------- round-2 P1 #2: reference adapter custom-id idempotency
def test_reference_remember_with_existing_id_same_content_is_idempotent() -> None:
    """Re-remembering with the same id+user+text returns the existing episode, no error."""
    adapter = ReferenceMemoryAdapter()
    try:
        first = adapter.remember(
            "ayush", "I live in Mumbai.", episode_id="custom-id-1"
        )
        second = adapter.remember(
            "ayush", "I live in Mumbai.", episode_id="custom-id-1"
        )
        # Same id, same created_at — second call returned the existing row.
        assert first.id == second.id == "custom-id-1"
        assert first.created_at == second.created_at
        # Only one row in the namespace.
        assert len(adapter.list_episodes("ayush")) == 1
    finally:
        adapter.close()


def test_reference_remember_with_existing_id_different_text_raises() -> None:
    """A real collision (same id, different content) must raise rather than overwrite."""
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember("ayush", "I live in Mumbai.", episode_id="custom-id-1")
        with pytest.raises(ValueError, match=r"already exists"):
            adapter.remember(
                "ayush", "I live in Bangalore.", episode_id="custom-id-1"
            )
    finally:
        adapter.close()


def test_reference_remember_with_existing_id_different_user_raises() -> None:
    """Same id under a different user is also a collision and must raise."""
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember("alice", "Project codename: Aurora.", episode_id="cust-1")
        with pytest.raises(ValueError, match=r"already exists"):
            adapter.remember("bob", "Project codename: Aurora.", episode_id="cust-1")
    finally:
        adapter.close()


def test_mutation_retry_against_same_persisted_provider_is_idempotent() -> None:
    """Re-running the same mutation under the same contract id against the same
    persistent reference adapter must NOT raise UNIQUE-constraint errors and
    must NOT add duplicate episodes."""
    adapter = ReferenceMemoryAdapter()
    try:
        for _ in range(2):
            run = ContractRun(
                id=uuid4().hex,
                contract_id="tests/x.py::test_retry_idempotent",
                provider="reference",
                started_at=datetime.now(tz=UTC),
            )
            contract = MemoryContract(adapter, run)
            contract.given_user("ayush")
            contract.with_distractors(5, seed=0)
        eps = adapter.list_episodes("ayush")
        # First call inserted 5; second call idempotently returned existing 5
        # without raising and without duplicating.
        assert len(eps) == 5
    finally:
        adapter.close()


# ---- round-3: seed-type validation ---------------------------------------
def test_sample_distractors_rejects_none_seed() -> None:
    """``seed=None`` would silently read system entropy — must raise instead."""
    from recalllab.core.mutations import validate_seed

    with pytest.raises(TypeError, match=r"seed must be an int"):
        validate_seed(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"seed must be an int"):
        sample_distractors(3, seed=None)  # type: ignore[arg-type]


def test_sample_distractors_rejects_string_seed() -> None:
    """A config-provided string seed must not be coerced — raise loudly."""
    with pytest.raises(TypeError, match=r"seed must be an int"):
        sample_distractors(3, seed="abc")  # type: ignore[arg-type]


def test_sample_distractors_rejects_bool_seed() -> None:
    """``True``/``False`` are int subclasses in Python — reject them anyway.

    A boolean ``seed`` is almost always a bug at the call site (e.g. someone
    passed a flag where an integer was meant) and ``random.Random(True)``
    behaves like ``random.Random(1)``, which is a confusing silent coercion.
    """
    with pytest.raises(TypeError, match=r"seed must be an int"):
        sample_distractors(3, seed=True)  # type: ignore[arg-type]


def test_with_distractors_rejects_none_seed() -> None:
    """End-to-end through the DSL: ``with_distractors(seed=None)`` raises."""
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        with pytest.raises(TypeError, match=r"seed must be an int"):
            contract.with_distractors(5, seed=None)  # type: ignore[arg-type]
        # And no MUTATION trace event was recorded — seed validation runs
        # before any provider write.
        assert adapter.list_episodes("ayush") == []
    finally:
        adapter.close()


# ---- round-3: custom-id capability gate ----------------------------------
class _NoCustomIdProvider:
    """Stub provider that does NOT declare ``supports_custom_episode_ids``.

    Tracks remembers in-memory so ``list_episodes`` returns them — this is
    important because ``with_stale_repeats`` now does a resurrection guard
    that asks the provider for the live episode set before the capability
    gate fires. A provider that lost its episodes between remember and
    stale_repeats would trip the resurrection guard first; this stub
    represents a realistic provider that DOES retain its writes but
    happens to not honour custom episode IDs.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._episodes: list[Episode] = []

    def capabilities(self) -> CapabilityFlags:
        return CapabilityFlags(
            supports_forget=True,
            supports_provenance=True,
            supports_custom_episode_ids=False,
        )

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        self.calls += 1
        ep = Episode(
            id=episode_id or f"stub-{self.calls}",
            user_id=user_id,
            text=text,
            created_at=datetime.now(tz=UTC),
            metadata=metadata,
        )
        self._episodes.append(ep)
        return ep

    def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Any]:
        return []

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        return 0

    def list_episodes(self, user_id: str) -> list[Episode]:
        return [ep for ep in self._episodes if ep.user_id == user_id]

    def delete_user(self, user_id: str) -> None:
        self._episodes = [ep for ep in self._episodes if ep.user_id != user_id]


def test_with_distractors_raises_on_provider_without_custom_id_capability() -> None:
    """A provider that can't honour requested IDs must not silently mutate.

    Mutation retry idempotency relies on writing at deterministic IDs. If
    the provider doesn't declare ``supports_custom_episode_ids``, we have
    no guarantee that retrying a contract addresses the same records, so
    the mutation must fail with a clear error and a traced
    ``status="unsupported"`` event rather than scattering ghost writes.
    """
    provider = _NoCustomIdProvider()
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_capability_gate",
        provider="stub-no-custom-id",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    with pytest.raises(RuntimeError, match=r"supports_custom_episode_ids"):
        contract.with_distractors(3, seed=0)
    # The provider must not have been called.
    assert provider.calls == 0
    # And the trace must record the unsupported attempt with the
    # capability-gate status.
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    payload = mutation_events[0].payload
    assert payload["status"] == "unsupported"
    assert payload["inserted_episode_ids"] == []
    assert "supports_custom_episode_ids" in payload["error"]


def test_with_stale_repeats_raises_on_provider_without_custom_id_capability() -> None:
    """Same capability gate must protect ``with_stale_repeats``."""
    provider = _NoCustomIdProvider()
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_capability_gate_repeats",
        provider="stub-no-custom-id",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    # We need a prior remember for stale_repeats to find a source event.
    # _NoCustomIdProvider.remember accepts a write through plain ``remember``
    # (not the mutation path), so this seeds without triggering the gate.
    contract.remember("I live in Bangalore.")
    with pytest.raises(RuntimeError, match=r"supports_custom_episode_ids"):
        contract.with_stale_repeats(times=3)
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    assert mutation_events[0].payload["status"] == "unsupported"


# ---- round-3: provider returning a mismatched ID -------------------------
class _LyingCustomIdProvider:
    """Declares custom-id support but actually rewrites the requested ID.

    Modelled on the failure mode where a hosted MCP server accepts an
    ``episode_id`` argument but writes under its own server-generated id.
    The mutation pipeline must catch this rather than recording a trace
    that claims success.
    """

    def __init__(self, *, override_after: int = 0) -> None:
        self._override_after = override_after
        self.calls = 0

    def capabilities(self) -> CapabilityFlags:
        return CapabilityFlags(
            supports_forget=True,
            supports_provenance=True,
            supports_custom_episode_ids=True,
        )

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        self.calls += 1
        eid = (
            f"server-generated-{self.calls}"
            if self.calls > self._override_after
            else (episode_id or f"echo-{self.calls}")
        )
        return Episode(
            id=eid,
            user_id=user_id,
            text=text,
            created_at=datetime.now(tz=UTC),
            metadata=metadata,
        )

    def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Any]:
        return []

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        return 0

    def list_episodes(self, user_id: str) -> list[Episode]:
        return []

    def delete_user(self, user_id: str) -> None:
        return None


def test_with_distractors_raises_when_provider_returns_mismatched_id() -> None:
    """A provider that returns a different ID than requested breaks idempotency.

    The mutation pipeline must detect the mismatch and raise — recording
    ``status="partial_failed"`` plus the partial ``inserted_episode_ids``
    so the trace explains what actually landed at the provider.
    """
    provider = _LyingCustomIdProvider(override_after=1)
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_lying_provider",
        provider="lying-stub",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    with pytest.raises(RuntimeError, match=r"returned episode_id"):
        contract.with_distractors(3, seed=0)
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    payload = mutation_events[0].payload
    assert payload["status"] == "partial_failed"
    # First write echoed the requested id; second was rewritten — both
    # land in inserted_episode_ids before the mismatch raises.
    assert len(payload["inserted_episode_ids"]) == 2
    assert payload["inserted_episode_ids"][-1].startswith("server-generated-")
    assert "returned episode_id" in payload["error"]


# ---- round-3: reference adapter metadata-aware idempotency ---------------
def test_reference_remember_with_existing_id_same_metadata_is_idempotent() -> None:
    """Same id+user+text+metadata is a safe replay — return existing row."""
    adapter = ReferenceMemoryAdapter()
    try:
        first = adapter.remember(
            "ayush",
            "I live in Mumbai.",
            episode_id="cust-1",
            metadata={"source": "chat-2026-05-08", "confidence": 0.9},
        )
        second = adapter.remember(
            "ayush",
            "I live in Mumbai.",
            episode_id="cust-1",
            metadata={"source": "chat-2026-05-08", "confidence": 0.9},
        )
        assert first.id == second.id == "cust-1"
        assert first.created_at == second.created_at
        assert first.metadata == second.metadata == {
            "source": "chat-2026-05-08",
            "confidence": 0.9,
        }
        # Only one row.
        assert len(adapter.list_episodes("ayush")) == 1
    finally:
        adapter.close()


def test_reference_remember_with_existing_id_metadata_key_order_is_idempotent() -> None:
    """Metadata equality must be order-insensitive — canonical JSON compares."""
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember(
            "ayush",
            "Same text.",
            episode_id="cust-1",
            metadata={"a": 1, "b": 2},
        )
        # Same content, different insertion order in the dict literal.
        echoed = adapter.remember(
            "ayush",
            "Same text.",
            episode_id="cust-1",
            metadata={"b": 2, "a": 1},
        )
        assert echoed.id == "cust-1"
        assert len(adapter.list_episodes("ayush")) == 1
    finally:
        adapter.close()


def test_reference_remember_with_existing_id_different_metadata_raises() -> None:
    """Stale-metadata replay is a silent data-corruption path — must raise.

    A real-world version: the first write stored ``confidence=0.9``; the
    caller fixes the provenance and tries to re-write with
    ``confidence=0.95``. Returning the existing row would leave the stale
    metadata in place and mask the discrepancy. RecallLab refuses.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember(
            "ayush",
            "I live in Mumbai.",
            episode_id="cust-1",
            metadata={"confidence": 0.9},
        )
        with pytest.raises(ValueError, match=r"already exists"):
            adapter.remember(
                "ayush",
                "I live in Mumbai.",
                episode_id="cust-1",
                metadata={"confidence": 0.95},
            )
    finally:
        adapter.close()


def test_reference_remember_with_existing_id_metadata_none_vs_empty_distinct() -> None:
    """``metadata=None`` and ``metadata={}`` are distinct states.

    The first stores SQL NULL; the second stores the JSON string ``{}``.
    Replaying one against the other must not silently succeed.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember("ayush", "text", episode_id="cust-1", metadata=None)
        with pytest.raises(ValueError, match=r"already exists"):
            adapter.remember("ayush", "text", episode_id="cust-1", metadata={})
    finally:
        adapter.close()


# ---- round-5: same-contract retry after partial failure resumes idempotently
class _MidLoopFailingWrapper:
    """Wraps a real ``MemoryProvider``, raises on the (fail_after+1)-th
    ``remember`` call. ``heal()`` removes the failure cap so a retry can run.

    Used to simulate the canonical "hosted provider partial-fails mid-batch"
    scenario without giving up the reference adapter's real idempotency.
    """

    def __init__(self, inner: Any, fail_after: int) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self._calls = 0

    def heal(self) -> None:
        self._fail_after = 10**9

    # ---------------------------------------------------------- protocol passthrough
    def capabilities(self) -> CapabilityFlags:
        caps: CapabilityFlags = self._inner.capabilities()
        return caps

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("simulated mid-loop failure")
        episode: Episode = self._inner.remember(
            user_id, text, episode_id=episode_id, metadata=metadata
        )
        return episode

    def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Any]:
        results: list[Any] = self._inner.recall(user_id, query, k=k)
        return results

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        deleted: int = self._inner.forget(
            user_id, matching=matching, episode_id=episode_id
        )
        return deleted

    def list_episodes(self, user_id: str) -> list[Episode]:
        episodes: list[Episode] = self._inner.list_episodes(user_id)
        return episodes

    def delete_user(self, user_id: str) -> None:
        self._inner.delete_user(user_id)


def test_with_distractors_retry_after_partial_failure_reuses_invocation() -> None:
    """A partial-failed mutation, when retried on the same contract+provider,
    must reuse the original invocation so the deterministic episode IDs
    match the orphan rows from the failed attempt.

    Without the in-flight bookkeeping, the second call would advance the
    invocation counter and request a fresh ID set; the reference adapter
    would see no idempotency hit, and the retry would double-write the
    distractors that already landed.
    """
    inner = ReferenceMemoryAdapter()
    try:
        wrapper = _MidLoopFailingWrapper(inner, fail_after=3)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_retry_resume",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")

        # First attempt: 3 writes succeed, 4th raises.
        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_distractors(10, seed=0)
        assert len(inner.list_episodes("ayush")) == 3

        # Heal and retry on the SAME contract — must complete with 10 total
        # rows (not 13 — the 3 orphan rows must be reused, not duplicated).
        wrapper.heal()
        contract.with_distractors(10, seed=0)
        eps = inner.list_episodes("ayush")
        assert len(eps) == 10
        # All ids must be the deterministic mutation ids.
        assert all(ep.id.startswith("mut-distractors-") for ep in eps)

        # Trace integrity: two MUTATION events, first partial_failed, second
        # completed, SAME invocation number on both.
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        assert mutation_events[0].payload["status"] == "partial_failed"
        assert mutation_events[1].payload["status"] == "completed"
        assert (
            mutation_events[0].payload["invocation"]
            == mutation_events[1].payload["invocation"]
        )
    finally:
        inner.close()


def test_with_stale_repeats_retry_after_partial_failure_reuses_invocation() -> None:
    """Same retry-resume property must hold for ``with_stale_repeats``.

    The text_key is derived from the source remember; on retry it is the
    same so the fingerprint resolves to the same in-flight invocation.
    """
    inner = ReferenceMemoryAdapter()
    try:
        # fail_after=2 means: seed remember succeeds (call 1), first repeat
        # succeeds (call 2), second repeat raises.
        wrapper = _MidLoopFailingWrapper(inner, fail_after=2)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_retry_resume_repeats",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")  # call 1, succeeds.

        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_stale_repeats(times=5)  # 1 succeeds, 2nd raises.
        # 1 seed + 1 repeat = 2 episodes in the namespace.
        assert len(inner.list_episodes("ayush")) == 2

        wrapper.heal()
        contract.with_stale_repeats(times=5)  # retry — must resume.
        eps = inner.list_episodes("ayush")
        # 1 original remember + 5 repeats = 6 rows total. Crucially NOT 7
        # (which would mean the 1 orphan repeat was re-written under a new
        # invocation instead of being reused).
        assert len(eps) == 6
        repeat_ids = [ep.id for ep in eps if ep.id.startswith("mut-stale_repeats-")]
        assert len(repeat_ids) == 5

        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        assert mutation_events[0].payload["status"] == "partial_failed"
        assert mutation_events[1].payload["status"] == "completed"
        assert (
            mutation_events[0].payload["invocation"]
            == mutation_events[1].payload["invocation"]
        )
    finally:
        inner.close()


def test_deliberate_re_call_after_successful_completion_gets_fresh_invocation() -> None:
    """The retry-resume mechanism must NOT collapse two successful calls.

    Round-2 property: calling the same mutation twice deliberately under
    one user must produce distinct episode IDs. The in-flight fingerprint
    is cleared on success, so the second call allocates a fresh
    invocation and the IDs do not collide.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_deliberate_recall",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run)
        contract.given_user("ayush")
        contract.with_distractors(2, seed=0)  # succeeds, fingerprint cleared
        contract.with_distractors(2, seed=0)  # fresh invocation, fresh IDs
        eps = adapter.list_episodes("ayush")
        # 2 + 2 = 4 distinct distractors.
        assert len(eps) == 4
        assert len({ep.id for ep in eps}) == 4
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        # Invocations must DIFFER between the two successful calls.
        assert (
            mutation_events[0].payload["invocation"]
            != mutation_events[1].payload["invocation"]
        )
    finally:
        adapter.close()


def test_partial_failed_does_not_steal_invocations_from_other_fingerprints() -> None:
    """An in-flight failed mutation under user A must not affect user B's
    invocation numbering. The fingerprint isolation in
    ``_in_flight_mutations`` keeps each (type, user, key) tuple
    independent.
    """
    inner = ReferenceMemoryAdapter()
    try:
        wrapper = _MidLoopFailingWrapper(inner, fail_after=2)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_fingerprint_isolation",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)

        # Alice partial-fails.
        contract.given_user("alice")
        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_distractors(5, seed=0)

        # Bob succeeds with the same seed. His invocation must NOT collide
        # with Alice's in-flight one, and the resulting IDs must be
        # distinct from Alice's orphan rows.
        wrapper.heal()
        contract.given_user("bob")
        contract.with_distractors(3, seed=0)

        alice_ids = {ep.id for ep in inner.list_episodes("alice")}
        bob_ids = {ep.id for ep in inner.list_episodes("bob")}
        assert alice_ids.isdisjoint(bob_ids)
        assert len(alice_ids) == 2  # 2 succeeded before the failure
        assert len(bob_ids) == 3
    finally:
        inner.close()


# ---- round-6: resurrection guard for with_stale_repeats ------------------
def test_with_stale_repeats_refuses_to_resurrect_after_forget_by_matching() -> None:
    """A forgotten source remember must NOT be revived by ``with_stale_repeats``.

    Round-6 Codex finding: ``_last_remember_for`` walks REMEMBER events
    only and never accounts for later FORGET events. Without this guard
    a contract that does
    ``remember("I am allergic to peanuts.") → forget(matching="peanuts")
    → with_stale_repeats(times=1)`` would write the deleted text back
    under a mutation id, silently undermining forget / privacy contracts
    and making the trace look intentional rather than a resurrection.
    """
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        contract.remember("I am allergic to peanuts.")
        deleted = contract.forget(matching="peanuts")
        assert deleted == 1
        assert adapter.list_episodes("ayush") == []
        with pytest.raises(RuntimeError, match=r"forgotten in this contract"):
            contract.with_stale_repeats(times=1)
        # No mutation writes landed.
        assert adapter.list_episodes("ayush") == []
    finally:
        adapter.close()


def test_with_stale_repeats_refuses_to_resurrect_after_forget_by_episode_id() -> None:
    """The guard must catch episode-id-based forgets as well as matching-based.

    ``_last_remember_for`` resolves the source by walking the trace; only
    a provider-side check distinguishes "remembered but still live" from
    "remembered but later forgotten by id".
    """
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        seeded = contract.remember("Project codename: Aurora.")
        contract.forget(episode_id=seeded.id)
        assert adapter.list_episodes("ayush") == []
        with pytest.raises(RuntimeError, match=r"forgotten in this contract"):
            contract.with_stale_repeats(times=1)
    finally:
        adapter.close()


def test_with_stale_repeats_works_when_fresh_remember_follows_forget() -> None:
    """After forget, a fresh ``remember`` becomes the new source — and the
    resurrection guard accepts it because the new episode is live.
    """
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        contract.remember("I am allergic to peanuts.")
        contract.forget(matching="peanuts")
        # Fresh remember after the forget — this is now the latest live
        # source for stale_repeats.
        contract.remember("I live in Mumbai.")
        contract.with_stale_repeats(times=3)
        eps = adapter.list_episodes("ayush")
        # 1 fresh remember + 3 stale repeats = 4 rows, all with the
        # Mumbai text (the peanuts remember was forgotten and not
        # resurrected).
        assert len(eps) == 4
        assert all(ep.text == "I live in Mumbai." for ep in eps)
    finally:
        adapter.close()


# ---- round-6: tighter retry fingerprint (n and times must be part of key)
def test_partial_failed_distractors_then_different_n_is_not_a_retry() -> None:
    """``with_distractors(10, seed=0)`` partial-fail must NOT be matched as a
    retry by a later ``with_distractors(5, seed=0)``.

    Round-6 Codex finding: the in-flight fingerprint must include ``n``
    so distinct calls with the same seed but different counts each get
    their own invocation. Without this, the smaller-count call would
    clear the larger one's in-flight slot and the trace would record
    "completed" with the wrong requested count.
    """
    inner = ReferenceMemoryAdapter()
    try:
        wrapper = _MidLoopFailingWrapper(inner, fail_after=3)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_distinct_n",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")

        # First call partial-fails (3 writes land, 4th raises).
        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_distractors(10, seed=0)
        assert len(inner.list_episodes("ayush")) == 3
        first_orphan_ids = {ep.id for ep in inner.list_episodes("ayush")}

        # Second call with the SAME seed but DIFFERENT n. Must allocate
        # a fresh invocation and write at distinct IDs.
        wrapper.heal()
        contract.with_distractors(5, seed=0)
        eps = inner.list_episodes("ayush")
        # 3 orphan + 5 fresh = 8 distinct rows.
        assert len(eps) == 8
        new_ids = {ep.id for ep in eps} - first_orphan_ids
        assert len(new_ids) == 5
        assert new_ids.isdisjoint(first_orphan_ids)

        # Two MUTATION events: first partial_failed (requested=10), second
        # completed (requested=5). Invocations must DIFFER.
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        assert mutation_events[0].payload["status"] == "partial_failed"
        assert mutation_events[0].payload["requested"] == 10
        assert mutation_events[1].payload["status"] == "completed"
        assert mutation_events[1].payload["requested"] == 5
        assert (
            mutation_events[0].payload["invocation"]
            != mutation_events[1].payload["invocation"]
        )
    finally:
        inner.close()


def test_partial_failed_stale_repeats_then_different_times_is_not_a_retry() -> None:
    """Same property for ``with_stale_repeats``: a later call with the same
    source remember but different ``times`` is NOT a retry.
    """
    inner = ReferenceMemoryAdapter()
    try:
        # fail_after=2 → seed remember (call 1) succeeds; first stale repeat
        # (call 2) succeeds; second stale repeat (call 3) raises.
        wrapper = _MidLoopFailingWrapper(inner, fail_after=2)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_distinct_times",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")

        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_stale_repeats(times=5)
        assert len(inner.list_episodes("ayush")) == 2  # 1 seed + 1 repeat

        # Now retry with a DIFFERENT ``times``. Must allocate a fresh
        # invocation, not match the in-flight times=5 fingerprint.
        wrapper.heal()
        contract.with_stale_repeats(times=2)
        eps = inner.list_episodes("ayush")
        # 1 seed + 1 orphan from times=5 + 2 fresh from times=2 = 4.
        assert len(eps) == 4

        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        assert mutation_events[0].payload["times"] == 5
        assert mutation_events[1].payload["times"] == 2
        assert (
            mutation_events[0].payload["invocation"]
            != mutation_events[1].payload["invocation"]
        )
    finally:
        inner.close()


def test_partial_failed_stale_repeats_then_new_source_is_not_a_retry() -> None:
    """Two ``with_stale_repeats`` calls with the same text but distinct source
    remembers (e.g. the user re-stated the same fact) must not collapse
    into a single retry. The fingerprint includes the source remember's
    trace sequence, which is distinct per remember event.
    """
    inner = ReferenceMemoryAdapter()
    try:
        wrapper = _MidLoopFailingWrapper(inner, fail_after=2)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_new_source",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")

        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_stale_repeats(times=3)
        # 1 seed + 1 orphan.
        assert len(inner.list_episodes("ayush")) == 2

        # Re-state the same fact (a new REMEMBER event with the same text).
        wrapper.heal()
        contract.remember("I live in Bangalore.")

        # New stale_repeats with SAME times=3 against the new source.
        # Different source sequence → different fingerprint → fresh
        # invocation → distinct mutation IDs from the orphan.
        contract.with_stale_repeats(times=3)
        eps = inner.list_episodes("ayush")
        # 2 seeds + 1 orphan + 3 fresh = 6 rows.
        assert len(eps) == 6

        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        assert (
            mutation_events[0].payload["invocation"]
            != mutation_events[1].payload["invocation"]
        )
    finally:
        inner.close()


# ---- round-7: resurrection guard must not block valid non-authoritative providers
class _NonAuthoritativeListProvider:
    """Honours custom episode IDs but reports an empty ``list_episodes``.

    Models the canonical hosted-MCP failure mode: ``remember`` honours the
    requested id, but the server's wildcard recall (used by the adapter's
    ``list_episodes`` shim) doesn't enumerate everything — so
    ``list_episodes`` returns a best-effort subset (often empty).

    Before round-7, ``with_stale_repeats`` treated that empty list as
    proof of deletion and falsely raised on every call. The capability
    gate (``supports_authoritative_list``) now keeps the provider check
    off; the trace-based check still catches in-contract forgets.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._episodes: list[Episode] = []

    def capabilities(self) -> CapabilityFlags:
        return CapabilityFlags(
            supports_forget=True,
            supports_provenance=True,
            supports_custom_episode_ids=True,
            # Best-effort listing — flag stays False.
            supports_authoritative_list=False,
        )

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        self.calls += 1
        ep = Episode(
            id=episode_id or f"stub-{self.calls}",
            user_id=user_id,
            text=text,
            created_at=datetime.now(tz=UTC),
            metadata=metadata,
        )
        self._episodes.append(ep)
        return ep

    def recall(self, user_id: str, query: str, *, k: int = 5) -> list[Any]:
        return []

    def forget(
        self,
        user_id: str,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        before = len(self._episodes)
        if episode_id is not None:
            self._episodes = [
                ep for ep in self._episodes if ep.id != episode_id
            ]
        elif matching is not None:
            needle = matching.lower()
            self._episodes = [
                ep for ep in self._episodes if needle not in ep.text.lower()
            ]
        return before - len(self._episodes)

    def list_episodes(self, user_id: str) -> list[Episode]:
        # Pretend wildcard recall always returns nothing — best-effort
        # listing the protocol doesn't promise to be authoritative.
        return []

    def delete_user(self, user_id: str) -> None:
        self._episodes = [ep for ep in self._episodes if ep.user_id != user_id]


def test_with_stale_repeats_runs_on_non_authoritative_list_provider() -> None:
    """A provider with non-authoritative ``list_episodes`` must NOT trip the
    resurrection guard for a normal ``remember → with_stale_repeats``.

    Round-7 Codex finding: treating the listing as authoritative falsely
    raises on configured MCP servers whose wildcard recall doesn't
    enumerate. The trace-based check has no FORGET event, and the
    provider-side check is gated on the capability flag, so the
    mutation proceeds.
    """
    provider = _NonAuthoritativeListProvider()
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_non_auth_list",
        provider="stub-non-auth",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    contract.remember("I live in Bangalore.")
    contract.with_stale_repeats(times=3)
    # 1 seed + 3 stale repeats — the provider-side liveness check was
    # skipped because the capability flag is False.
    user_episodes = [ep for ep in provider._episodes if ep.user_id == "ayush"]
    assert len(user_episodes) == 4
    mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
    assert len(mutation_events) == 1
    assert mutation_events[0].payload["status"] == "completed"


def test_trace_based_resurrection_guard_fires_on_non_authoritative_provider() -> None:
    """Trace-based detection works regardless of provider listing semantics.

    Even when the provider's ``list_episodes`` is not authoritative, an
    in-contract ``forget`` followed by ``with_stale_repeats`` must still
    raise — because the FORGET event is recorded in the trace and the
    walk catches it.
    """
    provider = _NonAuthoritativeListProvider()
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_trace_guard",
        provider="stub-non-auth",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(provider, run)
    contract.given_user("ayush")
    contract.remember("I am allergic to peanuts.")
    contract.forget(matching="peanuts")
    with pytest.raises(RuntimeError, match=r"forgotten in this contract"):
        contract.with_stale_repeats(times=1)


def test_trace_resurrection_guard_does_not_false_positive_on_recreate() -> None:
    """A forget-then-recreate sequence must NOT trip the resurrection guard.

    The walk for FORGET events starts AFTER the most recent REMEMBER, so
    a forget that happened before the re-creation is out of scope. The
    new remember is the source; it has not been forgotten.
    """
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        contract.remember("I am allergic to peanuts.")
        contract.forget(matching="peanuts")
        # Re-state the same text. This new remember is the source for
        # the upcoming stale_repeats; the earlier forget should not
        # block it.
        contract.remember("I am allergic to peanuts.")
        contract.with_stale_repeats(times=2)
        eps = adapter.list_episodes("ayush")
        # 1 fresh remember + 2 stale repeats = 3 rows. (The original
        # remember + forget cleared cleanly.)
        assert len(eps) == 3
    finally:
        adapter.close()


def test_trace_resurrection_guard_catches_substring_match() -> None:
    """``forget(matching=X)`` where X is contained in the source text must
    trip the guard even though the forget query was a fragment.

    Reference and LangGraph forget delete every episode whose text
    contains the matching substring, so any matching FORGET removes the
    source. Conservative for MCP semantics (we'd rather refuse than
    resurrect).
    """
    contract, adapter, _ = _new_contract()
    try:
        contract.given_user("ayush")
        contract.remember("I live in Bangalore right now.")
        # Forget matches a substring of the source text.
        contract.forget(matching="Bangalore")
        with pytest.raises(RuntimeError, match=r"forgotten in this contract"):
            contract.with_stale_repeats(times=1)
        # And no mutation writes leaked through.
        assert adapter.list_episodes("ayush") == []
    finally:
        adapter.close()


# ---- round-8: deterministic IDs must be stable under unrelated edits ------
def test_stale_repeat_ids_stable_when_unrelated_prior_events_added() -> None:
    """Adding unrelated events before the source remember must NOT change the
    deterministic mutation IDs for ``with_stale_repeats``.

    Round-8 Codex finding: a previous design used the source event's
    trace ``sequence`` in the deterministic ID key. That sequence shifts
    if the user adds any earlier event (a recall, a remember under a
    different user, a setup remember of a different text), so a rerun
    of the "same logical contract" against the same persistent store
    requested fresh mutation IDs and double-wrote instead of deduping.

    The fix derives the key from the *ordinal* of the source among
    prior remembers of the same (user, text) — invariant under
    unrelated events, distinct between back-to-back same-text
    remembers, and stable across fresh runs of the same contract code.
    """
    contract_id = "tests/x.py::test_unrelated_edits"

    # Baseline run: the original "logical contract" — remember then
    # stale_repeats. Capture the mutation episode IDs.
    baseline_ids: list[str] = []
    baseline_adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id=contract_id,
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(baseline_adapter, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=3)
        baseline_ids = sorted(
            ep.id
            for ep in baseline_adapter.list_episodes("ayush")
            if ep.id.startswith("mut-stale_repeats-")
        )
    finally:
        baseline_adapter.close()
    assert len(baseline_ids) == 3

    # Edited run: same logical contract, but with an UNRELATED setup
    # event prepended (a recall under a different user, a remember for
    # someone else, even an extra recall by Ayush before the source).
    # The deterministic IDs requested by ``with_stale_repeats`` MUST
    # match the baseline so a retry against the same persistent store
    # dedupes the orphan rows instead of double-writing.
    edited_adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id=contract_id,
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(edited_adapter, run)
        # Unrelated setup — a recall for someone else, a different-user
        # remember, and an extra recall by Ayush. None of these change
        # the count of prior "I live in Bangalore." remembers for ayush
        # (still zero).
        contract.given_user("alice")
        contract.remember("Project codename: Aurora.")
        contract.recall("anything", k=1)
        contract.given_user("ayush")
        contract.recall("warmup query", k=1)
        # Now the SAME source remember and SAME mutation call as the
        # baseline.
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=3)
        edited_ids = sorted(
            ep.id
            for ep in edited_adapter.list_episodes("ayush")
            if ep.id.startswith("mut-stale_repeats-")
        )
    finally:
        edited_adapter.close()

    assert edited_ids == baseline_ids, (
        "Adding unrelated prior events changed the deterministic "
        "stale_repeat IDs — retries against a persistent store would "
        "double-write instead of deduping."
    )


def test_partial_failed_stale_repeats_then_different_source_records_abandoned_orphans() -> None:
    """``remember(A) → with_stale_repeats partial-fails → remember(B) →
    with_stale_repeats`` is a realistic recovery path. The second call
    is a fresh mutation (different source), so it gets a new
    invocation. But the A-mutation orphan rows are now stranded in
    storage AND the A-fingerprint stays in the in-flight map.

    Round-11 Codex finding: silently moving on to B leaves the operator
    with no signal that orphans exist. The fix surfaces the abandoned
    in-flight fingerprints in the new mutation's ``MUTATION`` payload
    under ``abandoned_in_flight``, so the trace explains the orphan
    state the retry didn't address.
    """
    inner = ReferenceMemoryAdapter()
    try:
        wrapper = _MidLoopFailingWrapper(inner, fail_after=2)
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_abandoned_orphans",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(wrapper, run)
        contract.given_user("ayush")

        # fail_after=2 → seed remember (call 1) succeeds, first stale
        # repeat (call 2) succeeds, second stale repeat (call 3) raises.
        contract.remember("I live in Bangalore.")
        with pytest.raises(RuntimeError, match=r"simulated mid-loop"):
            contract.with_stale_repeats(times=5)

        # Now do a different remember and call stale_repeats again. This
        # is NOT a retry of the failed Bangalore mutation — different
        # source means different fingerprint.
        wrapper.heal()
        contract.remember("I live in Mumbai.")
        contract.with_stale_repeats(times=3)

        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 2
        first = mutation_events[0].payload
        second = mutation_events[1].payload
        assert first["status"] == "partial_failed"
        assert second["status"] == "completed"

        # The new (successful) mutation event must explicitly surface
        # the abandoned in-flight Bangalore fingerprint. Without this
        # the operator would have no clue from the second event alone
        # that there are orphan rows from the failed first mutation.
        assert "abandoned_in_flight" in second
        abandoned = second["abandoned_in_flight"]
        assert len(abandoned) == 1
        assert abandoned[0]["mutation_type"] == "stale_repeats"
        assert abandoned[0]["invocation"] == first["invocation"]
        # The key encodes (text_hash | times | source_ordinal) so we
        # don't compare the exact value, just that it's distinct from
        # the new mutation's key.
        assert abandoned[0]["key"] != second.get("invocation")

        # And the storage still holds the orphans from the failed
        # mutation — 1 Bangalore + 1 Bangalore-repeat + 1 Mumbai + 3
        # Mumbai-repeats = 6 rows total.
        eps = inner.list_episodes("ayush")
        assert len(eps) == 6
    finally:
        inner.close()


def test_normal_mutation_payload_has_empty_abandoned_in_flight() -> None:
    """When there's no partial-failure history, the new field must be
    empty (consistent payload shape).
    """
    adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_no_abandoned",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=2)
        mutation_events = [e for e in run.events if e.kind == EventKind.MUTATION]
        assert len(mutation_events) == 1
        assert mutation_events[0].payload["abandoned_in_flight"] == []
    finally:
        adapter.close()


def test_stale_repeat_back_to_back_same_text_calls_get_distinct_ids() -> None:
    """Two ``with_stale_repeats`` calls on back-to-back remembers of the same
    text must still produce distinct, non-colliding mutation IDs.

    The round-6 property "distinct source remembers ⇒ distinct
    fingerprints" must survive the round-8 refactor away from
    ``last_remember.sequence``. The new ordinal-based discriminator
    correctly distinguishes "first remember of X" from "second remember
    of X" without needing the volatile sequence.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id="tests/x.py::test_back_to_back",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run)
        contract.given_user("ayush")
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=2)
        # Fresh remember of the same text, then another stale_repeats.
        contract.remember("I live in Bangalore.")
        contract.with_stale_repeats(times=2)
        eps = adapter.list_episodes("ayush")
        # 2 seeds + 2 + 2 = 6 rows total. All distinct ids.
        assert len(eps) == 6
        assert len({ep.id for ep in eps}) == 6
        repeat_ids = [ep.id for ep in eps if ep.id.startswith("mut-stale_repeats-")]
        assert len(repeat_ids) == 4
        assert len(set(repeat_ids)) == 4
    finally:
        adapter.close()
