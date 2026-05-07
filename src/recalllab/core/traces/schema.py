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
    """A single recorded step in a contract run."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    kind: EventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    latency_ms: float | None = None
    cost_estimate: dict[str, Any] | None = None


class AssertionResult(BaseModel):
    """Outcome of one `should_recall` / `forget` / etc. assertion."""

    model_config = ConfigDict(frozen=True)

    passed: bool
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
    """Top-level run record persisted to the trace store."""

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
