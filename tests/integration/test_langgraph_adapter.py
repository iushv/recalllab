"""Integration test: LangGraph Store adapter against an in-process InMemoryStore.

Covers the v0.2.0 mutation-idempotency contract that LangGraph declares:
``supports_custom_episode_ids=True`` means a retry of ``remember(...,
episode_id=X)`` must be a real no-op (not an overwrite that refreshes
``created_at`` or silently replaces a different value). The adapter does
a read-before-write on explicit ids; this file pins that behaviour.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.store.memory import InMemoryStore

from recalllab.adapters.langgraph_store import LangGraphStoreAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.traces.schema import ContractRun


@pytest.fixture
def adapter() -> LangGraphStoreAdapter:
    return LangGraphStoreAdapter(InMemoryStore())


def test_langgraph_declares_custom_episode_id_support(
    adapter: LangGraphStoreAdapter,
) -> None:
    assert adapter.capabilities().supports_custom_episode_ids is True


def test_remember_with_existing_id_same_content_preserves_created_at(
    adapter: LangGraphStoreAdapter,
) -> None:
    """Idempotent replay: same id+text+metadata returns the original episode.

    The naive ``BaseStore.put`` always overwrites and would refresh
    ``created_at``. The adapter must read first and return the existing
    value unchanged.
    """
    first = adapter.remember(
        "ayush", "I live in Mumbai.", episode_id="cust-1",
        metadata={"source": "chat", "confidence": 0.9},
    )
    second = adapter.remember(
        "ayush", "I live in Mumbai.", episode_id="cust-1",
        metadata={"source": "chat", "confidence": 0.9},
    )
    assert first.id == second.id == "cust-1"
    # created_at must NOT be refreshed on idempotent replay.
    assert first.created_at == second.created_at


def test_remember_with_existing_id_metadata_key_order_is_idempotent(
    adapter: LangGraphStoreAdapter,
) -> None:
    """Metadata dict equality must be order-insensitive."""
    first = adapter.remember(
        "ayush", "Same text.", episode_id="cust-1",
        metadata={"a": 1, "b": 2},
    )
    second = adapter.remember(
        "ayush", "Same text.", episode_id="cust-1",
        metadata={"b": 2, "a": 1},
    )
    assert first.created_at == second.created_at


def test_remember_with_existing_id_different_text_raises(
    adapter: LangGraphStoreAdapter,
) -> None:
    """Same id + different text must NOT silently overwrite.

    Without the read-before-write check, a regenerated mutation that hashes
    to the same deterministic key (e.g. after a distractor-pool edit) would
    silently replace the original episode and break temporal contracts.
    """
    adapter.remember("ayush", "I live in Mumbai.", episode_id="cust-1")
    with pytest.raises(ValueError, match=r"different \(text, metadata\)"):
        adapter.remember("ayush", "I live in Bangalore.", episode_id="cust-1")


def test_remember_with_existing_id_different_metadata_raises(
    adapter: LangGraphStoreAdapter,
) -> None:
    """Same id + different metadata is a real collision — must raise."""
    adapter.remember(
        "ayush", "Same text.", episode_id="cust-1",
        metadata={"confidence": 0.9},
    )
    with pytest.raises(ValueError, match=r"different \(text, metadata\)"):
        adapter.remember(
            "ayush", "Same text.", episode_id="cust-1",
            metadata={"confidence": 0.95},
        )


def test_remember_with_existing_id_metadata_none_vs_empty_distinct(
    adapter: LangGraphStoreAdapter,
) -> None:
    """``metadata=None`` and ``metadata={}`` are distinct states (mirrors reference)."""
    adapter.remember("ayush", "text", episode_id="cust-1", metadata=None)
    with pytest.raises(ValueError, match=r"different \(text, metadata\)"):
        adapter.remember("ayush", "text", episode_id="cust-1", metadata={})


def test_mutation_retry_against_same_langgraph_store_is_idempotent(
    adapter: LangGraphStoreAdapter,
) -> None:
    """End-to-end: re-running the same mutation must not duplicate or overwrite."""
    contract_id = "tests/x.py::test_lg_retry"
    for _ in range(2):
        run = ContractRun(
            id=uuid4().hex,
            contract_id=contract_id,
            provider="langgraph",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run)
        contract.given_user("ayush")
        contract.with_distractors(5, seed=0)
    eps = adapter.list_episodes("ayush")
    # 5 distractors from the first run, second run idempotently returned
    # the same rows — no duplication, no overwrite.
    assert len(eps) == 5
    # All five must still carry the deterministic ids.
    assert all(ep.id.startswith("mut-distractors-") for ep in eps)


def test_list_episodes_raises_when_scan_limit_saturates() -> None:
    """``list_episodes`` must refuse to return a silently truncated subset.

    Round-9 Codex finding: the LangGraph adapter declares
    ``supports_authoritative_list=True``, which makes
    ``with_stale_repeats`` use ``list_episodes`` as a liveness oracle.
    But the underlying scan was bounded by ``scan_limit`` and would
    silently drop items past the bound — so a live source episode in a
    large namespace could be mis-diagnosed as deleted. Now the listing
    raises when saturation is possible, matching the
    ``forget(matching=...)`` and ``delete_user`` semantics on the same
    adapter.
    """
    adapter = LangGraphStoreAdapter(InMemoryStore(), scan_limit=3)
    adapter.remember("ayush", "fact 1")
    adapter.remember("ayush", "fact 2")
    adapter.remember("ayush", "fact 3")
    with pytest.raises(RuntimeError, match=r"list_episodes hit scan_limit"):
        adapter.list_episodes("ayush")


def test_with_stale_repeats_raises_when_listing_saturates() -> None:
    """End-to-end: ``with_stale_repeats`` propagates the scan-limit error.

    A user running mutations against a namespace bigger than the
    configured ``scan_limit`` gets a clear error pointing at the bound,
    not a silent false-positive resurrection diagnosis.
    """
    adapter = LangGraphStoreAdapter(InMemoryStore(), scan_limit=2)
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x.py::test_lg_scan_saturation",
        provider="langgraph",
        started_at=datetime.now(tz=UTC),
    )
    contract = MemoryContract(adapter, run)
    contract.given_user("ayush")
    contract.remember("filler 1")
    contract.remember("filler 2")
    contract.remember("I live in Mumbai.")  # the would-be source
    with pytest.raises(RuntimeError, match=r"list_episodes hit scan_limit"):
        contract.with_stale_repeats(times=1)


def test_concurrent_custom_id_writes_do_not_silently_overwrite() -> None:
    """Round-12 Codex finding: ``BaseStore.put`` overwrites silently, so a
    naive get-then-put is racy. Two concurrent ``remember(..., episode_id=X)``
    calls with different text could both see no existing item and both
    proceed to ``put`` — last-writer-wins. The adapter's process-local
    write lock serialises the get+put critical section so the second
    writer sees the first's row and either raises (different content)
    or returns the existing episode (same content).

    Multi-process / multi-host stores still need application-level
    coordination, documented on the adapter's ``_write_lock`` attribute.
    """
    adapter = LangGraphStoreAdapter(InMemoryStore())
    results: list[Exception | str] = []
    barrier = threading.Barrier(2)

    def writer(text: str) -> None:
        barrier.wait()  # release both threads at the same instant
        try:
            ep = adapter.remember(
                "ayush", text, episode_id="cust-conflict-1"
            )
            results.append(ep.text)
        except ValueError as exc:
            results.append(exc)

    t1 = threading.Thread(target=writer, args=("I live in Mumbai.",))
    t2 = threading.Thread(target=writer, args=("I live in Bangalore.",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one writer must have inserted the row; the other must
    # have seen the existing-but-different-content row and raised.
    assert len(results) == 2
    successes = [r for r in results if isinstance(r, str)]
    failures = [r for r in results if isinstance(r, ValueError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "different (text, metadata)" in str(failures[0])

    # And the store holds exactly one row at the contested id — the
    # winner's row — NOT a silently overwritten last-writer-wins state.
    eps = adapter.list_episodes("ayush")
    assert len(eps) == 1
    assert eps[0].id == "cust-conflict-1"
    # The surviving text must match the success result.
    assert eps[0].text == successes[0]


def test_concurrent_same_content_writes_are_idempotent() -> None:
    """Two concurrent writes with the SAME id+text+metadata must both
    succeed (both return the same Episode) without producing duplicate
    rows. The lock ensures the second writer sees the first's row and
    returns it via the idempotency path rather than re-inserting.
    """
    adapter = LangGraphStoreAdapter(InMemoryStore())
    results: list[str] = []
    barrier = threading.Barrier(3)

    def writer() -> None:
        barrier.wait()
        ep = adapter.remember(
            "ayush",
            "Same content.",
            episode_id="cust-same-1",
            metadata={"source": "chat"},
        )
        results.append(ep.id)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3
    assert all(r == "cust-same-1" for r in results)
    eps = adapter.list_episodes("ayush")
    assert len(eps) == 1
