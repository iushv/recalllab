"""Judge runtime — protocol, capabilities, exceptions, rubric, NoOpJudge.

Real judge backends (``AnthropicJudge``) land in v0.2.2 step 4 alongside
verdict caching and the per-session cost cap. See
``docs/judge-assertions.md`` for the v0.2.2 design and implementation
order.
"""

from __future__ import annotations

from recalllab.core.judge.base import (
    JudgeBudgetExceededError,
    JudgeCapabilities,
    JudgeCostEstimate,
    JudgeError,
    JudgeMode,
    JudgeProvider,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
)
from recalllab.core.judge.noop import NoOpJudge
from recalllab.core.judge.rubric import Rubric

__all__ = [
    "JudgeBudgetExceededError",
    "JudgeCapabilities",
    "JudgeCostEstimate",
    "JudgeError",
    "JudgeMode",
    "JudgeProvider",
    "JudgeRequest",
    "JudgeUnavailableError",
    "JudgeVerdict",
    "NoOpJudge",
    "Rubric",
]
