"""Default judge: no API key, no calls, always raises on evaluate.

Wired up when ``[judge].provider = "none"`` (the default in
``recalllab.toml``). The pytest plugin's fail-loud gate in
``MemoryContract.should_recall`` checks ``capabilities().available``
BEFORE calling ``evaluate()`` and raises ``JudgeUnavailableError`` at
the DSL boundary with a friendly long-form message instead. ``evaluate``
raising here is the last-resort safety net — anything that reaches it
means the gate logic upstream was bypassed.
"""

from __future__ import annotations

from recalllab.core.judge.base import (
    JudgeCapabilities,
    JudgeProvider,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
)

__all__ = ["NoOpJudge"]


class NoOpJudge(JudgeProvider):
    """Judge that declines every request.

    ``capabilities().available`` is ``False``; the pytest plugin uses
    that flag to gate judge-mode assertions per Decision #3b
    (``docs/judge-assertions.md``).
    """

    def capabilities(self) -> JudgeCapabilities:
        return JudgeCapabilities(available=False)

    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        # NoOpJudge is the default backend when ``[judge].provider =
        # "none"``. Contracts written against
        # ``MemoryContract.should_recall`` never reach this method
        # because the DSL gate (see ``docs/judge-assertions.md``
        # Decision #3b) raises a friendlier ``JudgeUnavailableError``
        # upstream with guidance on configuring ``[judge]`` or marking
        # the contract optional. If you're calling ``evaluate()``
        # directly (e.g. third-party tooling that builds a
        # ``JudgeProvider`` outside the DSL), that's unsupported by
        # design — instantiate ``AnthropicJudge`` or another configured
        # backend instead.
        raise JudgeUnavailableError(
            "NoOpJudge cannot evaluate. Configure [judge] in "
            "recalllab.toml (set provider = \"anthropic\" and install "
            "the [judge] extra) to enable judge-mode assertions. "
            "NoOpJudge.evaluate() is intentionally a no-op; the "
            "MemoryContract DSL gate handles the user-facing error "
            "message and should not reach this method in normal use."
        )
