"""Regressions for the Codex adversarial review findings.

Each test pins exactly one fix in place:

1. The pytest plugin must NOT create ``.recalllab/`` just because the
   package is installed — only when a contract test actually uses
   ``memory_contract``.
2. ``ReferenceMemoryAdapter.forget(matching=...)`` must treat the input
   as a literal substring, not a SQL LIKE pattern (``'%'`` is just a
   percent sign, not "match everything").
3. ``LangGraphStoreAdapter.forget(matching=...)`` and ``delete_user``
   must refuse to silently succeed when the scan window is saturated.
4. ``MCPMemoryAdapter._extract`` must traverse dotted paths so users
   can point at ``data.results`` style nested responses.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from recalllab.adapters.mcp_configurable.adapter import MCPMemoryAdapter
from recalllab.adapters.reference import ReferenceMemoryAdapter


def _run_inner_pytest(testfile: Path) -> subprocess.CompletedProcess[str]:
    """Run pytest in a subprocess against ``testfile``; cwd is the file's parent.

    Subprocess gives us a clean process boundary for asserting "the
    plugin's autoload had no on-disk side effects" without depending on
    pytester's pytest11 reload semantics.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(testfile)],
        cwd=testfile.parent,
        capture_output=True,
        text=True,
        check=False,
    )


# ----------------------------------------------------------------- finding #1
def test_plugin_does_not_create_trace_store_without_contract_use(
    tmp_path: Path,
) -> None:
    """Installing RecallLab must not pollute unrelated repos with .recalllab/."""
    test_file = tmp_path / "test_unrelated.py"
    test_file.write_text(
        "def test_passes_with_no_recalllab_use():\n"
        "    assert 1 + 1 == 2\n"
    )
    result = _run_inner_pytest(test_file)
    assert result.returncode == 0, (
        f"inner pytest failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert not (tmp_path / ".recalllab").exists(), (
        "RecallLab pytest plugin created .recalllab/ in a project that "
        "doesn't use memory_contract — pytest11 autoload must be inert "
        "until the fixture is actually requested."
    )


def test_plugin_creates_trace_store_when_contract_runs(tmp_path: Path) -> None:
    """And it MUST create the trace store the moment a contract runs."""
    test_file = tmp_path / "test_uses_contract.py"
    test_file.write_text(
        "def test_basic(memory_contract):\n"
        '    memory_contract.given_user("ayush")\n'
        '    memory_contract.remember("hello world")\n'
        '    memory_contract.should_recall("hello", contains="hello")\n'
    )
    result = _run_inner_pytest(test_file)
    assert result.returncode == 0, (
        f"inner pytest failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert (tmp_path / ".recalllab" / "traces.sqlite").exists(), (
        "memory_contract was used but no trace was persisted"
    )


# ----------------------------------------------------------------- finding #2
@pytest.mark.parametrize(
    "wildcard,seeded",
    [
        ("%", ["I love python", "I love rust"]),
        ("_", ["one", "two", "three"]),
        ("\\", ["nothing", "still nothing"]),
    ],
)
def test_reference_forget_treats_matching_as_literal_substring(
    wildcard: str, seeded: list[str]
) -> None:
    """``forget(matching='%')`` must NOT delete every memory for the user."""
    adapter = ReferenceMemoryAdapter()
    try:
        for text in seeded:
            adapter.remember("ayush", text)
        deleted = adapter.forget("ayush", matching=wildcard)
        assert deleted == 0, (
            f"matching={wildcard!r} should not match any seeded text "
            f"({seeded!r}); deleted {deleted}"
        )
        remaining = [ep.text for ep in adapter.list_episodes("ayush")]
        assert sorted(remaining) == sorted(seeded), (
            f"forget(matching={wildcard!r}) deleted memories that didn't "
            f"contain {wildcard!r} literally; survived: {remaining!r}"
        )
    finally:
        adapter.close()


def test_reference_forget_still_deletes_real_substring_match() -> None:
    """The literal substring rewrite must not break the legitimate case."""
    adapter = ReferenceMemoryAdapter()
    try:
        adapter.remember("ayush", "I am allergic to peanuts.")
        adapter.remember("ayush", "I love mangoes.")
        deleted = adapter.forget("ayush", matching="peanut")
        assert deleted == 1
        remaining = [ep.text for ep in adapter.list_episodes("ayush")]
        assert remaining == ["I love mangoes."]
    finally:
        adapter.close()


# ----------------------------------------------------------------- finding #3
def test_langgraph_forget_match_raises_when_scan_limit_saturated() -> None:
    """Match-forget must refuse to claim success on a partial scan."""
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from recalllab.adapters.langgraph_store import LangGraphStoreAdapter

    store = InMemoryStore()
    adapter = LangGraphStoreAdapter(store, scan_limit=3)
    # Seed > scan_limit items; even one of them matches the pattern, so a
    # naive scan would return "deleted=1" and silently leave the rest.
    for i in range(5):
        adapter.remember("ayush", f"memory number {i} mentions peanuts")

    with pytest.raises(RuntimeError, match=r"scan_limit"):
        adapter.forget("ayush", matching="peanuts")


def test_langgraph_delete_user_raises_when_scan_limit_saturated() -> None:
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from recalllab.adapters.langgraph_store import LangGraphStoreAdapter

    store = InMemoryStore()
    adapter = LangGraphStoreAdapter(store, scan_limit=2)
    for i in range(4):
        adapter.remember("ayush", f"item {i}")
    with pytest.raises(RuntimeError, match=r"scan_limit"):
        adapter.delete_user("ayush")


def test_langgraph_forget_episode_id_unaffected_by_scan_limit() -> None:
    """Episode-id forget is exact and must keep working even if the namespace
    holds far more items than ``scan_limit``."""
    pytest.importorskip("langgraph.store.memory")
    from langgraph.store.memory import InMemoryStore

    from recalllab.adapters.langgraph_store import LangGraphStoreAdapter

    store = InMemoryStore()
    adapter = LangGraphStoreAdapter(store, scan_limit=2)
    target = adapter.remember("ayush", "the one to delete")
    for i in range(5):
        adapter.remember("ayush", f"distractor {i}")
    deleted = adapter.forget("ayush", episode_id=target.id)
    assert deleted == 1


# ----------------------------------------------------------------- finding #4
def test_mcp_extract_supports_dotted_paths() -> None:
    """``_extract`` must walk dotted paths so dotted recall fields work."""
    payload = {"data": {"results": [1, 2, 3], "meta": {"count": 3}}}
    assert MCPMemoryAdapter._extract(payload, "data.results") == [1, 2, 3]
    assert MCPMemoryAdapter._extract(payload, "data.meta.count") == 3


def test_mcp_extract_returns_none_on_missing_segment() -> None:
    payload: dict[str, dict[str, list[int]]] = {"data": {"results": []}}
    assert MCPMemoryAdapter._extract(payload, "data.missing.further") is None
    assert MCPMemoryAdapter._extract(payload, "missing") is None


def test_mcp_extract_returns_none_when_path_hits_non_dict() -> None:
    payload: dict[str, list[dict[str, str]]] = {"results": [{"text": "hi"}]}
    # Trying to walk into a list with a dotted segment must not crash.
    assert MCPMemoryAdapter._extract(payload, "results.text") is None


def test_mcp_extract_flat_path_still_works() -> None:
    """Backwards compatibility with the simple-key case."""
    payload = {"results": ["a", "b"]}
    assert MCPMemoryAdapter._extract(payload, "results") == ["a", "b"]


# ------------------------------------------ round-11: legacy provider compat
def test_memory_contract_remember_works_against_legacy_provider() -> None:
    """``MemoryContract.remember`` must NOT forward ``episode_id`` to
    the provider when the caller didn't supply one.

    Round-11 Codex finding: the round-5 round-trip fix made
    ``MemoryContract.remember(text)`` always call
    ``provider.remember(user_id, text, episode_id=...)``. Legacy
    third-party adapters written against the v0.1 protocol surface
    have ``remember(user_id, text)`` and don't accept the keyword —
    every ordinary ``memory_contract.remember("...")`` raised
    ``TypeError: unexpected keyword argument 'episode_id'``. The
    ``MemoryProvider`` Protocol is ``runtime_checkable`` but Python's
    structural typing only checks method *names*, not signatures, so
    a legacy adapter passes ``isinstance(x, MemoryProvider)`` and
    then fails at call time.
    """
    from datetime import UTC, datetime
    from typing import Any
    from uuid import uuid4

    from recalllab.adapters.base import CapabilityFlags, Episode
    from recalllab.core.contract.dsl import MemoryContract
    from recalllab.core.traces.schema import ContractRun

    class _LegacyProvider:
        """v0.1-style adapter — accepts ``(user_id, text)`` ONLY."""

        def __init__(self) -> None:
            self._episodes: list[Episode] = []

        def capabilities(self) -> CapabilityFlags:
            return CapabilityFlags(supports_forget=True)

        # Crucially: NO ``episode_id`` / ``metadata`` keyword arguments.
        # Any caller that forwards them unconditionally TypeErrors.
        def remember(self, user_id: str, text: str) -> Episode:
            ep = Episode(
                id=uuid4().hex,
                user_id=user_id,
                text=text,
                created_at=datetime.now(tz=UTC),
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
            return list(self._episodes)

        def delete_user(self, user_id: str) -> None:
            return None

    provider = _LegacyProvider()
    run = ContractRun(
        id=uuid4().hex,
        contract_id="tests/x::test_legacy_compat",
        provider="legacy",
        started_at=datetime.now(tz=UTC),
    )
    # _LegacyProvider intentionally lacks the v0.2.0 episode_id/metadata
    # kwargs — that's the whole point of this regression test. Suppress
    # the Protocol mismatch so mypy doesn't fail CI on the test that
    # specifically proves the DSL tolerates this shape.
    contract = MemoryContract(provider, run)  # type: ignore[arg-type]
    contract.given_user("ayush")
    # Ordinary remember — caller passed no episode_id. This must NOT
    # raise. Pre-fix it raised TypeError on the very first call.
    ep = contract.remember("I live in Mumbai.")
    assert ep.text == "I live in Mumbai."
    assert ep.user_id == "ayush"


def test_memory_contract_remember_forwards_episode_id_when_supplied() -> None:
    """The kwarg IS forwarded when the caller supplied it — that's the
    round-5 round-trip path. We just don't force the keyword on legacy
    providers when no id was requested. A modern provider (the
    reference adapter) must still receive the id and use it.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from recalllab.adapters.reference import ReferenceMemoryAdapter
    from recalllab.core.contract.dsl import MemoryContract
    from recalllab.core.traces.schema import ContractRun

    adapter = ReferenceMemoryAdapter()
    try:
        run = ContractRun(
            id=uuid4().hex,
            contract_id="unit::dsl_compat",
            provider="reference",
            started_at=datetime.now(tz=UTC),
        )
        contract = MemoryContract(adapter, run)
        contract.given_user("ayush")
        ep = contract.remember("I live in Mumbai.", episode_id="cust-1")
        assert ep.id == "cust-1"
        # The reference adapter stored it at the requested id.
        eps = adapter.list_episodes("ayush")
        assert len(eps) == 1
        assert eps[0].id == "cust-1"
    finally:
        adapter.close()
