"""pytest plugin entry point for RecallLab.

Exposes the ``memory_contract`` fixture and the ``recalllab_optional``
marker. Loads ``recalllab.toml`` from the rootdir on session start, builds
the configured provider, and persists each contract run to the SQLite trace
store after the test completes (whether it passed or failed).
"""

from __future__ import annotations

import tomllib
import uuid
from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recalllab.adapters.base import MemoryProvider
from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.traces.schema import CapabilitySkip, ContractRun, RunStatus
from recalllab.core.traces.sqlite_store import TraceStore

_DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "provider": {"type": "reference"},
    "trace": {"path": ".recalllab/traces.sqlite"},
    "judge": {"provider": "none"},
}

_CONFIG_KEY: pytest.StashKey[dict[str, dict[str, Any]]] = pytest.StashKey()
_TRACE_STORE_KEY: pytest.StashKey[TraceStore] = pytest.StashKey()
_CALL_REPORT_KEY: pytest.StashKey[pytest.TestReport] = pytest.StashKey()


def _load_config(rootdir: Path) -> dict[str, dict[str, Any]]:
    config_path = rootdir / "recalllab.toml"
    merged: dict[str, dict[str, Any]] = {
        section: dict(defaults) for section, defaults in _DEFAULT_CONFIG.items()
    }
    if not config_path.exists():
        return merged
    with config_path.open("rb") as fh:
        loaded = tomllib.load(fh)
    for section, section_defaults in _DEFAULT_CONFIG.items():
        if section in loaded and isinstance(loaded[section], dict):
            merged[section] = {**section_defaults, **loaded[section]}
    return merged


def _build_provider(config: dict[str, dict[str, Any]]) -> MemoryProvider:
    provider_type = config["provider"].get("type", "reference")
    if provider_type == "reference":
        return ReferenceMemoryAdapter()
    if provider_type == "langgraph_store":
        try:
            from langgraph.store.memory import InMemoryStore

            from recalllab.adapters.langgraph_store import LangGraphStoreAdapter
        except ImportError as exc:
            raise RuntimeError(
                "provider.type='langgraph_store' requires the [langgraph] "
                "extra. Install with: pip install 'recalllab[langgraph]'"
            ) from exc
        return LangGraphStoreAdapter(InMemoryStore())
    if provider_type == "mcp":
        try:
            from recalllab.adapters.mcp_configurable import (
                MCPMemoryAdapter,
                MCPMemoryConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "provider.type='mcp' requires the [mcp] extra. "
                "Install with: pip install 'recalllab[mcp]'"
            ) from exc
        mcp_section = config["provider"].get("mcp", {})
        if not isinstance(mcp_section, dict):
            raise ValueError(
                "[provider.mcp] must be a table in recalllab.toml when "
                "provider.type='mcp'"
            )
        if "server_url" not in mcp_section:
            raise ValueError(
                "[provider.mcp] is missing required 'server_url' "
                "(plus remember_tool / recall_tool); see docs/concepts.md"
            )
        return MCPMemoryAdapter(MCPMemoryConfig(**mcp_section))
    raise ValueError(
        f"unknown provider.type {provider_type!r} in recalllab.toml "
        f"(supported in v0.1: 'reference', 'langgraph_store', 'mcp')"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker and stash config — but do NOT touch the disk yet.

    The pytest11 entry point causes this module to load for *every* pytest
    session in any project that has RecallLab installed. Creating the trace
    store here would write ``.recalllab/traces.sqlite`` into unrelated repos
    (or fail entirely under read-only / sandboxed CI). Trace-store creation
    is deferred to the ``memory_contract`` fixture, which only runs when a
    contract test actually uses it.
    """
    config.addinivalue_line(
        "markers",
        "recalllab_optional(capability): contract requires a provider capability "
        "and is skipped automatically when the configured provider lacks it.",
    )
    rl_config = _load_config(Path(config.rootpath))
    config.stash[_CONFIG_KEY] = rl_config


def _trace_store(config: pytest.Config) -> TraceStore:
    """Get-or-create the session-wide trace store.

    Lazy: the ``.recalllab/`` directory and SQLite file are only created
    the first time a contract actually needs to persist a run.
    """
    store = config.stash.get(_TRACE_STORE_KEY, None)
    if store is None:
        rl_config = config.stash[_CONFIG_KEY]
        store = TraceStore(rl_config["trace"]["path"])
        config.stash[_TRACE_STORE_KEY] = store
    return store


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, None, None]:
    """Capture the call-phase report so the fixture finalizer can read it.

    Without this, the fixture only knows about failures that flowed through
    the DSL's own assertion helpers — raw ``assert`` statements, provider
    exceptions, and unrelated errors would all be persisted as PASSED.
    """
    outcome = yield
    rep: pytest.TestReport = outcome.get_result()  # type: ignore[attr-defined]
    if rep.when == "call":
        item.stash[_CALL_REPORT_KEY] = rep


@pytest.fixture
def recalllab_provider(pytestconfig: pytest.Config) -> Iterator[MemoryProvider]:
    """Per-test memory provider built from the configured adapter.

    Function-scoped so each contract starts with an empty memory namespace —
    cheap for the reference adapter; revisit for v0.2 once expensive
    network-backed adapters land.
    """
    config = pytestconfig.stash[_CONFIG_KEY]
    provider = _build_provider(config)
    try:
        yield provider
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@pytest.fixture
def memory_contract(
    request: pytest.FixtureRequest,
    recalllab_provider: MemoryProvider,
) -> Iterator[MemoryContract]:
    """The DSL fixture every contract test receives.

    Auto-skips the test when ``@pytest.mark.recalllab_optional("supports_X")``
    is present and the configured provider doesn't expose capability ``X``.
    """
    config = request.config.stash[_CONFIG_KEY]
    provider_name = config["provider"].get("type", "reference")

    # Build the run unconditionally so capability-gated skips are also
    # persisted — the Failure Gallery / capability matrix needs every
    # contract attempt to render an N/A row, not silently disappear.
    run = ContractRun(
        id=uuid.uuid4().hex,
        contract_id=request.node.nodeid,
        provider=provider_name,
        started_at=datetime.now(tz=UTC),
        status=RunStatus.PASSED,
    )
    contract = MemoryContract(recalllab_provider, run)

    skip_reason: str | None = None
    marker = request.node.get_closest_marker("recalllab_optional")
    if marker is not None and marker.args:
        capability = marker.args[0]
        caps = recalllab_provider.capabilities()
        if not getattr(caps, capability, False):
            skip_reason = (
                f"provider {provider_name!r} does not support capability "
                f"{capability!r}"
            )
            run.capability_skips.append(
                CapabilitySkip(capability=capability, reason=skip_reason)
            )

    try:
        if skip_reason is not None:
            pytest.skip(skip_reason)
        yield contract
    finally:
        # Map the actual pytest call-phase outcome to the contract run
        # status. The DSL's own assertion helpers flip status to FAILED on
        # their own, but raw ``assert`` statements, provider exceptions, and
        # capability-gated skips only show up via this finally block.
        rep = request.node.stash.get(_CALL_REPORT_KEY, None)
        if rep is None:
            # No call-phase report — either we skipped during fixture setup
            # (capability gate) or the fixture errored before yield. Use
            # skip_reason to distinguish.
            run.status = RunStatus.SKIPPED if skip_reason else RunStatus.ERROR
        elif rep.outcome == "failed":
            run.status = RunStatus.FAILED
        elif rep.outcome == "skipped":
            run.status = RunStatus.SKIPPED
        elif run.status not in (RunStatus.FAILED, RunStatus.ERROR):
            run.status = RunStatus.PASSED

        run.finished_at = datetime.now(tz=UTC)
        _trace_store(request.config).write_run(run)
