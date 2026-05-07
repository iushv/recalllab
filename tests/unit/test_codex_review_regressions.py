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
