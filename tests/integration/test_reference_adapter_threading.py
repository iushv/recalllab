"""Concurrency tests for ``ReferenceMemoryAdapter``.

Pins the round-1 code-review finding: the adapter held a single
``sqlite3.Connection`` without ``check_same_thread=False`` or a lock, so
any user running pytest-xdist or hopping threads via an asyncio adapter
would hit ``sqlite3.ProgrammingError`` on the first method call from a
non-creator thread. Worse, two threads racing the SELECT-then-INSERT in
``remember(episode_id=X)`` could both pass the existence check and both
INSERT — only one would survive the PRIMARY KEY constraint, but the
idempotent-replay-vs-collision-raise semantics would be non-deterministic.

The fix mirrors the round-12 LangGraph pattern: a process-local
``threading.Lock`` serialises all hot-path connection access, plus
``check_same_thread=False`` on connect so the locked sections can run
from any thread.
"""

from __future__ import annotations

import threading

from recalllab.adapters.reference import ReferenceMemoryAdapter


def test_remember_and_recall_from_a_different_thread_works() -> None:
    """Without ``check_same_thread=False`` the second thread's call would
    raise ``ProgrammingError``. Pinned so a future "tighten thread check"
    refactor can't silently regress.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember("ayush", "I live in Mumbai.")
        results: list[object] = []

        def worker() -> None:
            try:
                ep = adapter.remember("ayush", "Cross-thread write.")
                hits = adapter.recall("ayush", "Mumbai", k=5)
                results.append((ep.id, [r.text for r in hits]))
            except Exception as exc:
                results.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert len(results) == 1
        result = results[0]
        assert not isinstance(result, Exception), (
            f"cross-thread call raised: {result!r}"
        )
    finally:
        adapter.close()


def test_concurrent_custom_id_writes_serialize_under_lock() -> None:
    """Two threads racing ``remember(episode_id=X)`` with DIFFERENT content
    must produce deterministic semantics: one writer's row lands, the
    other raises ``ValueError`` because the SELECT-then-INSERT check
    catches the collision.

    Mirrors ``test_concurrent_custom_id_writes_do_not_silently_overwrite``
    in ``tests/integration/test_langgraph_adapter.py`` — the reference
    adapter must give the same guarantee.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        results: list[object] = []
        barrier = threading.Barrier(2)

        def writer(text: str) -> None:
            barrier.wait()
            try:
                ep = adapter.remember("ayush", text, episode_id="cust-race-1")
                results.append(("ok", ep.text))
            except ValueError as exc:
                results.append(("collide", str(exc)))

        t1 = threading.Thread(target=writer, args=("I live in Mumbai.",))
        t2 = threading.Thread(target=writer, args=("I live in Bangalore.",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one writer succeeded; the other detected the collision.
        ok = [r for r in results if isinstance(r, tuple) and r[0] == "ok"]
        collisions = [r for r in results if isinstance(r, tuple) and r[0] == "collide"]
        assert len(ok) == 1, f"expected exactly one successful writer, got {results!r}"
        assert len(collisions) == 1, (
            f"expected exactly one collision raise, got {results!r}"
        )
        # The collision message must point at the contested id.
        assert "cust-race-1" in collisions[0][1]

        # And the store holds exactly the winner's row at that id.
        eps = adapter.list_episodes("ayush")
        assert len(eps) == 1
        assert eps[0].id == "cust-race-1"
        assert isinstance(ok[0], tuple)
        assert eps[0].text == ok[0][1]
    finally:
        adapter.close()


def test_concurrent_same_content_writes_are_idempotent() -> None:
    """Three threads writing the SAME id + content concurrently must all
    succeed (idempotent replay) without producing duplicate rows.

    This is the v0.2.0 "retry-resume after partial failure" property —
    if a mutation pipeline issues the same write twice (the retry path
    after partial_failed), both calls must return the same Episode.
    """
    adapter = ReferenceMemoryAdapter()
    try:
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
    finally:
        adapter.close()


def test_concurrent_remember_and_forget_do_not_corrupt_state() -> None:
    """A ``forget(matching=X)`` thread running alongside ``remember()``
    threads must produce consistent state — either the matching writes
    landed and then got swept, or they raced and survived. The lock
    means the scan-then-delete in ``forget`` is atomic relative to
    other writers; the result is non-deterministic but always
    self-consistent.

    Concretely: at the end, every surviving row whose text contains
    ``"forget"`` would have been written AFTER the forget acquired the
    lock. We don't assert specific counts, only that ``list_episodes``
    + ``recall`` agree (no half-deleted rows visible to one but not the
    other).
    """
    adapter = ReferenceMemoryAdapter()
    try:
        barrier = threading.Barrier(11)  # 10 writers + 1 sweeper

        def writer(i: int) -> None:
            barrier.wait()
            adapter.remember("ayush", f"please forget item {i}.")

        def sweeper() -> None:
            barrier.wait()
            adapter.forget("ayush", matching="forget")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        threads.append(threading.Thread(target=sweeper))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Self-consistency: list_episodes and recall must agree on which
        # rows exist. A half-deleted row would show up in one but not
        # the other.
        listed = {ep.id for ep in adapter.list_episodes("ayush")}
        recalled = adapter.recall("ayush", "forget item", k=20)
        recalled_ids = {r.episode_id for r in recalled if r.episode_id is not None}
        # Every recalled id must be in the listed set.
        assert recalled_ids.issubset(listed), (
            f"recall returned ids not in list_episodes: "
            f"{recalled_ids - listed!r}"
        )
    finally:
        adapter.close()


def test_cross_thread_access_does_not_raise_programming_error() -> None:
    """Defense-in-depth: a fresh thread's call must not raise
    ``sqlite3.ProgrammingError``. A future refactor that drops
    ``check_same_thread=False`` would re-introduce the surface this
    whole change is supposed to close.
    """
    adapter = ReferenceMemoryAdapter()
    try:
        errors: list[BaseException] = []

        def probe() -> None:
            try:
                adapter.list_episodes("nobody")
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        assert not errors, f"cross-thread access raised: {errors[0]!r}"
    finally:
        adapter.close()
