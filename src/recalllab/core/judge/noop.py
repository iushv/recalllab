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
        # Should be unreachable in normal flow: the DSL gate catches
        # unconfigured-judge calls before they get here. If we DO get
        # here, raise the same loud error the gate would have raised so
        # the failure mode is consistent regardless of who caught it.
        raise JudgeUnavailableError(
            "NoOpJudge cannot evaluate; configure [judge] in "
            "recalllab.toml (set provider = \"anthropic\" and install the "
            "[judge] extra). This call reached NoOpJudge.evaluate, which "
            "means the DSL fail-loud gate was bypassed — please file an "
            "issue against RecallLab if you see this in production."
        )
