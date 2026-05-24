"""pytest plugin entry point for RecallLab.

Exposes the ``memory_contract`` fixture and the ``recalllab_optional``
marker. Loads ``recalllab.toml`` from the rootdir on session start, builds
the configured provider AND judge, and persists each contract run to the
SQLite trace store after the test completes (whether it passed or failed).

The ``recalllab_optional`` marker takes a capability name and resolves it
through a capability-source resolver: provider-capability names (``supports_*``)
route to ``MemoryProvider.capabilities()``; judge-capability names
(``judge_configured``) route to ``JudgeProvider.capabilities()``. Unknown
capability names raise ``pytest.UsageError`` at collection time so typos
surface near the collected-items count rather than buried in fixture errors.
"""

from __future__ import annotations

import tomllib
import uuid
from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recalllab.adapters.base import CapabilityFlags, MemoryProvider
from recalllab.adapters.reference import ReferenceMemoryAdapter
from recalllab.core.contract.dsl import MemoryContract
from recalllab.core.judge import (
    JudgeCapabilities,
    JudgeProvider,
    NoOpJudge,
)
from recalllab.core.traces.schema import CapabilitySkip, ContractRun, RunStatus
from recalllab.core.traces.sqlite_store import TraceStore

_DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "provider": {"type": "reference"},
    "trace": {"path": ".recalllab/traces.sqlite"},
    "judge": {"provider": "none"},
}

_CONFIG_KEY: pytest.StashKey[dict[str, dict[str, Any]]] = pytest.StashKey()
_TRACE_STORE_KEY: pytest.StashKey[TraceStore] = pytest.StashKey()
_JUDGE_KEY: pytest.StashKey[JudgeProvider] = pytest.StashKey()
_CALL_REPORT_KEY: pytest.StashKey[pytest.TestReport] = pytest.StashKey()

# Capability-source resolver constants. The provider set is derived from
# ``CapabilityFlags`` so adding a new capability flag automatically registers
# its marker name. The judge set uses an explicit alias map because the
# user-facing marker name (``judge_configured``) intentionally differs from
# the model field name (``available``) — the marker reads as a sentence
# (`recalllab_optional("judge_configured")`) while the field is a generic
# JudgeCapabilities member. Future judge capabilities add an alias entry
# alongside their model field.
_PROVIDER_CAPABILITY_NAMES: frozenset[str] = frozenset(
    CapabilityFlags.model_fields.keys()
)
_JUDGE_CAPABILITY_ALIASES: dict[str, str] = {
    "judge_configured": "available",
}
_JUDGE_CAPABILITY_NAMES: frozenset[str] = frozenset(
    _JUDGE_CAPABILITY_ALIASES.keys()
)
_KNOWN_CAPABILITY_NAMES: frozenset[str] = (
    _PROVIDER_CAPABILITY_NAMES | _JUDGE_CAPABILITY_NAMES
)
# Hint kept in the JudgeCapabilities import path so a static-checker sees
# the model is actually used (the alias map references its fields by string).
_ = JudgeCapabilities


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


def _build_judge(config: dict[str, dict[str, Any]]) -> JudgeProvider:
    """Construct the judge backend from the ``[judge]`` config section.

    v0.2.2 step 2 ships ``NoOpJudge`` only. ``provider = "anthropic"`` is
    accepted in the schema but lands its implementation in step 4
    (``AnthropicJudge``); attempting to use it before step 4 raises with a
    clear "requires the [judge] extra" message because the import path
    won't resolve yet.
    """
    judge_provider = config["judge"].get("provider", "none")
    if judge_provider == "none":
        return NoOpJudge()
    if judge_provider == "anthropic":
        try:
            # Lazy import: the AnthropicJudge adapter ships in v0.2.2 step 4.
            # In step 2 the module doesn't exist yet, so this raises
            # ImportError and we surface the "[judge] extra not installed"
            # message — slightly misleading on step 2 (the extra IS
            # installed, the code just doesn't exist) but the right message
            # for the v0.2.2 release.
            from recalllab.core.judge.anthropic import (  # type: ignore[import-untyped]
                AnthropicJudge,
            )
        except ImportError as exc:
            raise RuntimeError(
                "judge.provider='anthropic' requires the [judge] extra. "
                "Install with: pip install 'recalllab[judge]'. "
                "(If the extra is installed and you still see this error, "
                "you may be on a v0.2.2-in-progress branch where the "
                "AnthropicJudge adapter has not landed yet.)"
            ) from exc
        return AnthropicJudge(config["judge"])  # type: ignore[no-any-return]
    raise ValueError(
        f"unknown judge.provider {judge_provider!r} in recalllab.toml "
        f"(supported in v0.2.2: 'none', 'anthropic')"
    )


def _resolve_capability(
    name: str,
    *,
    provider_caps: CapabilityFlags,
    judge_caps: JudgeCapabilities,
) -> bool:
    """Look up a capability name through the provider/judge resolver.

    Returns ``True`` when the capability is satisfied. Unknown capability
    names should never reach this function — ``pytest_collection_modifyitems``
    validates them at collection time. If one does (e.g. via a programmatic
    marker added after collection), we raise rather than silently treating
    it as unsatisfied so the bug surfaces loudly.
    """
    if name in _PROVIDER_CAPABILITY_NAMES:
        return bool(getattr(provider_caps, name))
    if name in _JUDGE_CAPABILITY_NAMES:
        field = _JUDGE_CAPABILITY_ALIASES[name]
        return bool(getattr(judge_caps, field))
    raise RuntimeError(
        f"unknown recalllab_optional capability {name!r} reached setup "
        f"phase; this should have been caught at collection time. "
        f"Known names: {sorted(_KNOWN_CAPABILITY_NAMES)}"
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker and stash config — but do NOT touch the disk yet.

    The pytest11 entry point causes this module to load for *every* pytest
    session in any project that has RecallLab installed. Creating the trace
    store here would write ``.recalllab/traces.sqlite`` into unrelated repos
    (or fail entirely under read-only / sandboxed CI). Trace-store creation
    is deferred to the ``memory_contract`` fixture, which only runs when a
    contract test actually uses it.

    The judge backend, by contrast, IS built here — ``NoOpJudge`` has no
    side effects and ``AnthropicJudge`` (step 4) only constructs a client
    object, not a network connection. Having the judge available at
    collection time lets the capability resolver run before any test
    fixture executes.
    """
    config.addinivalue_line(
        "markers",
        "recalllab_optional(capability): contract requires a provider or judge "
        "capability not present in the configured environment (e.g. "
        "supports_forget, judge_configured); the test is skipped automatically "
        "when the capability is missing.",
    )
    rl_config = _load_config(Path(config.rootpath))
    config.stash[_CONFIG_KEY] = rl_config
    config.stash[_JUDGE_KEY] = _build_judge(rl_config)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Validate every ``recalllab_optional`` capability name at collection time.

    Typos like ``recalllab_optional("supprots_forget")`` raise
    ``pytest.UsageError`` here, before any fixture runs, so the error
    appears at the top of the pytest output near the collected-items
    count rather than buried inside per-test fixture errors. See
    ``docs/judge-assertions.md`` OPEN-1 (locked decision).
    """
    unknown: list[tuple[str, str]] = []
    for item in items:
        for marker in item.iter_markers("recalllab_optional"):
            if not marker.args:
                continue
            capability_name = marker.args[0]
            if not isinstance(capability_name, str):
                unknown.append(
                    (item.nodeid, f"<non-string {type(capability_name).__name__}>")
                )
                continue
            if capability_name not in _KNOWN_CAPABILITY_NAMES:
                unknown.append((item.nodeid, capability_name))
    if not unknown:
        return
    msg_lines = ["unknown recalllab_optional capability name(s):"]
    for nodeid, name in unknown:
        msg_lines.append(f"  {nodeid}: {name!r}")
    msg_lines.append("")
    msg_lines.append(
        "known provider capabilities: "
        + ", ".join(sorted(_PROVIDER_CAPABILITY_NAMES))
    )
    msg_lines.append(
        "known judge capabilities:    "
        + ", ".join(sorted(_JUDGE_CAPABILITY_NAMES))
    )
    raise pytest.UsageError("\n".join(msg_lines))


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

    Capability gating walks **every** ``@pytest.mark.recalllab_optional``
    marker on the test (not just the innermost) and resolves each through
    the capability-source resolver: ``supports_*`` names route to the
    provider's ``capabilities()``; judge-capability names route to the
    judge's ``capabilities()``. The first missing capability wins and
    skips the test cleanly; multi-marker contracts honor *every* declared
    gate.

    Judge-mode kwargs in ``should_recall`` (``latest_fact_is``,
    ``must_not_answer_as``, ``judge_assertion``) used against an
    unconfigured judge raise ``JudgeUnavailableError`` UNLESS the
    contract is marked
    ``@pytest.mark.recalllab_optional("judge_configured")``, in which
    case the call short-circuits to ``pytest.skip``. This is the fail-
    loud default from ``docs/judge-assertions.md`` Decision #3b.
    """
    config = request.config.stash[_CONFIG_KEY]
    provider_name = config["provider"].get("type", "reference")
    judge = request.config.stash[_JUDGE_KEY]
    judge_provider_name = config["judge"].get("provider", "none")

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

    # Track whether the contract opted into "skip cleanly when judge is
    # not configured" via the marker. The DSL gate in ``should_recall``
    # reads this to decide between ``pytest.skip`` (marker present) and
    # ``raise JudgeUnavailableError`` (marker absent — fail-loud default).
    judge_optional = any(
        marker.args
        and isinstance(marker.args[0], str)
        and marker.args[0] == "judge_configured"
        for marker in request.node.iter_markers("recalllab_optional")
    )
    contract = MemoryContract(
        recalllab_provider,
        run,
        judge=judge,
        judge_optional=judge_optional,
    )

    # Walk every recalllab_optional marker, resolve via the capability
    # source, and skip on the first missing capability.
    provider_caps = recalllab_provider.capabilities()
    judge_caps = judge.capabilities()
    skip_reason: str | None = None
    for marker in request.node.iter_markers("recalllab_optional"):
        if not marker.args:
            continue
        capability = marker.args[0]
        if not isinstance(capability, str):
            # Collection-time validation should have caught this; defensive.
            continue
        if _resolve_capability(
            capability,
            provider_caps=provider_caps,
            judge_caps=judge_caps,
        ):
            continue
        if capability in _JUDGE_CAPABILITY_NAMES:
            skip_reason = (
                f"judge {judge_provider_name!r} does not provide "
                f"capability {capability!r}"
            )
        else:
            skip_reason = (
                f"provider {provider_name!r} does not support capability "
                f"{capability!r}"
            )
        run.capability_skips.append(
            CapabilitySkip(capability=capability, reason=skip_reason)
        )
        break  # First missing capability wins.

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
