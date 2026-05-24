"""Judge-provider protocol, capability flags, exceptions, and request/verdict shapes.

The ``JudgeProvider`` protocol is provider-neutral so swapping in OpenAI /
Bedrock / a local NLI model is a config change rather than a v0.2.2-breaking
PR. v0.2.2 ships ``NoOpJudge`` here and ``AnthropicJudge`` in step 4.

Identity tuple for a judge call (per ``docs/judge-assertions.md``
§Identity audit): ``(query, recall_results, expected, rubric, model, mode,
prompt_template_version)``. ``mode`` and ``prompt_template_version`` are
included from v0.2.2 specifically so the v0.2.3 verdict cache keys off a
complete identity surface and does not collide
``latest_fact_is="X"`` with ``must_not_answer_as=["X"]``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class JudgeMode(StrEnum):
    """Which assertion mode produced this judge call.

    Part of the prompt-identity tuple — same recall / expected with a
    different mode is a semantically different question and must not
    share a cache slot. See ``docs/judge-assertions.md`` §Identity audit.
    """

    LATEST_FACT_IS = "latest_fact_is"
    MUST_NOT_ANSWER_AS = "must_not_answer_as"
    JUDGE_ASSERTION = "judge_assertion"


class JudgeCapabilities(BaseModel):
    """What a judge provider can do.

    Read by the pytest plugin to gate the
    ``recalllab_optional("judge_configured")`` marker. ``NoOpJudge`` returns
    ``available=False``; real backends like ``AnthropicJudge`` return
    ``available=True`` once their API key is present.
    """

    model_config = ConfigDict(frozen=True)

    available: bool = False


class JudgeError(Exception):
    """Base class for judge-runtime errors."""


class JudgeUnavailableError(JudgeError):
    """The judge backend cannot answer.

    Raised at two distinct sites:

    1. ``MemoryContract.should_recall`` when a judge-mode kwarg is used
       against an unconfigured judge (``[judge].provider = "none"``) AND
       the contract is not marked
       ``@pytest.mark.recalllab_optional("judge_configured")``. This is
       the fail-loud default from ``docs/judge-assertions.md`` Decision
       #3b: silent skip turned forgotten CI config into a green build,
       which is the exact failure mode RecallLab exists to prevent.
    2. ``AnthropicJudge.evaluate`` on API errors, rate limits, and
       network failures (landing in step 4).
    """


class JudgeBudgetExceededError(JudgeError):
    """Per-run or per-session judge spending cap reached.

    Raised by ``AnthropicJudge`` (step 4) before issuing a new judge call
    once a cap has been hit. The in-flight invocation that hit the cap
    always completes (bounded overshoot of one invocation's spend
    including any retry); subsequent invocations raise. See
    ``docs/judge-assertions.md`` §Cost & budget.
    """


class JudgeCostEstimate(BaseModel):
    """Cost payload populated on the judge-mode ASSERT event.

    Lives on ``TraceEvent.cost_estimate`` per
    ``docs/judge-assertions.md`` Decision #4 (no separate
    ``EventKind.JUDGE`` in v0.2.2; that split returns in v0.2.3 alongside
    verdict caching).

    ``attempts`` includes retries: a malformed-JSON retry that re-hits
    the provider bumps ``attempts`` to 2 and sums both calls' tokens
    and USD.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_usd: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=0, ge=0)


class JudgeRequest(BaseModel):
    """The full identity tuple a judge call resolves against.

    Built by ``MemoryContract.should_recall`` at evaluation time and
    handed to ``JudgeProvider.evaluate``. ``prompt_template_version`` is
    a small integer hard-coded in the prompt-builder module and bumped
    whenever the template changes in a verdict-affecting way; v0.2.2
    ships at version 1.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    recall_result: str
    expected: str | list[str]
    rubric: str | None = None  # The Rubric.criterion text (not the Rubric instance).
    model: str
    mode: JudgeMode
    prompt_template_version: int = 1


class JudgeVerdict(BaseModel):
    """The judge's response to a ``JudgeRequest``.

    ``cost`` is the realized cost for *this verdict* (including any
    retries), populated on the judge-mode ASSERT's ``cost_estimate``.
    ``raw_responses`` stores up to one entry per attempt (truncated to
    4 KB each) so adversarial-content debugging is possible without
    re-querying the model. See ``docs/judge-assertions.md``
    §Failed-judge ASSERT lifecycle.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    reason: str
    cost: JudgeCostEstimate
    raw_responses: list[str] = Field(default_factory=list)


@runtime_checkable
class JudgeProvider(Protocol):
    """Minimal contract every judge backend implements.

    Two methods only: ``capabilities()`` for the fail-loud gate, and
    ``evaluate(request)`` for the actual judging. Real backends
    implement ``evaluate`` against a remote API; ``NoOpJudge`` always
    raises ``JudgeUnavailableError`` from ``evaluate`` because the gate
    should have caught the call before it got here.
    """

    def capabilities(self) -> JudgeCapabilities: ...

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict: ...
