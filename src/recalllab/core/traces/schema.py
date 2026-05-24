"""Pydantic schema for contract traces.

A contract run produces a `ContractRun` containing an ordered sequence of
`TraceEvent`s plus the assertions evaluated against them. The whole record
is JSON-serialisable and round-trips cleanly through SQLite.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventKind(StrEnum):
    GIVEN_USER = "given_user"
    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
    ASSERT = "assert"
    MUTATION = "mutation"


class RunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TraceEvent(BaseModel):
    """A single recorded step in a contract run.

    ``cost_estimate`` is unset for most events. For ``EventKind.ASSERT``
    events emitted by judge-mode assertions, v0.2.2 populates it with the
    judge-call accounting payload::

        {
            "provider": str,        # e.g. "anthropic"
            "model": str,           # e.g. "claude-haiku-4-5-20251022"
            "input_tokens": int,    # accumulated across attempts (incl. retries)
            "output_tokens": int,
            "estimated_usd": float, # accumulated across attempts
            "attempts": int,        # 1 for clean call, 2 if malformed-JSON retry fired
        }

    Rule-based ASSERTs (``contains`` / ``excludes``) leave ``cost_estimate``
    as ``None``. Decision #3a (``docs/judge-assertions.md``) forbids
    combining two judge-mode kwargs in one ``should_recall`` call, so at
    most one ASSERT per call carries a populated ``cost_estimate`` — no
    ambiguity about which row owns the cost. v0.2.2 ships the payload
    shape; the AnthropicJudge adapter (step 4) populates it.
    """

    model_config = ConfigDict(frozen=True)

    sequence: int
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    latency_ms: float | None = None
    cost_estimate: dict[str, Any] | None = None


class AssertionResult(BaseModel):
    """Outcome of one `should_recall` / `forget` / etc. assertion.

    ``passed`` is three-valued from v0.2.2:

    - ``True`` — assertion evaluated and passed.
    - ``False`` — assertion evaluated and failed (the contract failed).
    - ``None`` — assertion was *not evaluated*. Used for short-circuited
      judge-mode placeholders per Decision #9 (``docs/judge-assertions.md``):
      when a ``should_recall`` call combines rule-based + judge-mode
      kwargs and the rule-based assertion fails first, the judge never
      runs, but a placeholder ASSERT row is still emitted so
      ``recalllab record`` can faithfully regenerate the original call
      with both kwargs intact. The placeholder carries
      ``passed=None, reason="short_circuited: ..."`` and is not counted
      as a failure for run-status purposes.

    All status logic must distinguish ``passed is False`` (true failure)
    from ``passed is None`` (placeholder / not evaluated). Treating
    ``not passed`` as "failure" would silently flag placeholders as
    failures.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool | None
    mode: str
    expected: Any
    actual: str
    reason: str | None = None
    sequence: int


class CapabilitySkip(BaseModel):
    """Records which capability gated a contract skip."""

    model_config = ConfigDict(frozen=True)

    capability: str
    reason: str


class ContractRun(BaseModel):
    """Top-level run record persisted to the trace store.

    ``judge_cost_usd`` is the per-run aggregate of every judge call's
    ``cost_estimate.estimated_usd`` (see ``TraceEvent`` above). v0.2.2
    records on the trace; the Failure Gallery dashboard column lands in
    v0.2.2.1. Default ``0.0`` keeps rule-based-only runs unchanged.
    """

    id: str
    contract_id: str
    provider: str
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.PASSED
    events: list[TraceEvent] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
    capability_skips: list[CapabilitySkip] = Field(default_factory=list)
    mutation_source: str | None = None
    judge_cost_usd: float = Field(default=0.0, ge=0.0)
