"""AnthropicJudge — Claude-backed judge for v0.2.2 judge-mode assertions.

Lazy-imported by ``recalllab.core.pytest_plugin.plugin._build_judge`` when
``[judge].provider = "anthropic"`` in ``recalllab.toml``. Requires:

- The ``[judge]`` optional extra (``pip install 'recalllab[judge]'``).
- ``ANTHROPIC_API_KEY`` in the environment.

Enforces the per-pytest-session budget cap (``[judge].max_session_cost_usd``)
post-call (``docs/judge-assertions.md`` §Cost & budget): an invocation
already in flight always completes — including its malformed-JSON retry
— and the *next* invocation raises ``JudgeBudgetExceededError`` once the
running total has reached the cap. Worst-case session overshoot is one
full invocation-plus-retry.

The per-run cap (``[judge].max_cost_usd``) is enforced by the DSL caller
(step 5+) because the judge does not have visibility into the active
``ContractRun``. The judge exposes ``max_cost_usd`` as a read-only
attribute so the caller has one place to read it from.

Determinism: ``temperature=0`` minimizes variance, but Anthropic does
not contractually guarantee greedy decoding and may update the pinned
snapshot under the same model name. RecallLab does not promise
pass/fail stability for judge assertions across provider snapshots —
see ``docs/judge-assertions.md`` §Determinism & drift.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from recalllab.core.judge.base import (
    JudgeBudgetExceededError,
    JudgeCapabilities,
    JudgeCostEstimate,
    JudgePartialFailureError,
    JudgeProvider,
    JudgeRequest,
    JudgeUnavailableError,
    JudgeVerdict,
)
from recalllab.core.judge.prompts import build_judge_prompt

__all__ = ["AnthropicJudge"]


# Default model snapshot per ``docs/judge-assertions.md`` Decision #7.
# Pin the snapshot suffix; bumping it is a verdict-affecting change so
# JUDGE_PROMPT_TEMPLATE_VERSION should also rev if the caller wants the
# old verdicts invalidated.
_DEFAULT_MODEL = "claude-haiku-4-5-20251022"

# Approximate ~Haiku-tier pricing. Overridable via [judge] config keys so
# users on a different model can dial these in without forking the
# adapter. The defaults are conservative — slightly higher than real
# Haiku — so the budget caps trip a little early rather than late.
_DEFAULT_INPUT_PER_MTOK_USD = 1.00
_DEFAULT_OUTPUT_PER_MTOK_USD = 5.00

# Strict-JSON output cap. The judge response should always be a short
# JSON object; 512 tokens is comfortable for the reason field and keeps
# accidental long-form replies from blowing through the budget.
_MAX_OUTPUT_TOKENS = 512

# Raw response bodies stored in JudgeVerdict.raw_responses are capped to
# this many bytes each so the trace store doesn't grow unbounded on a
# pathological reply. Per docs/judge-assertions.md §Failed-judge ASSERT
# lifecycle.
_MAX_RAW_RESPONSE_BYTES = 4096

# Reminder appended to the user message when the first attempt's reply
# was not parseable as the required JSON shape.
_RETRY_REMINDER = (
    "\n\nYour previous response was not valid JSON in the required "
    'shape. Reply with strictly one JSON object: {"verdict": "PASS" | '
    '"FAIL", "reason": str}. No markdown code fences, no prose before '
    "or after, no leading explanation."
)


class _ParsedVerdict(BaseModel):
    """Pydantic-strict shape the judge's response must match.

    A response missing ``reason`` is allowed (defaults to ""), but the
    ``verdict`` field is required and must be one of the two literal
    values — anything else is treated as a parse failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: str
    reason: str = ""


_VERDICT_VALUES = frozenset({"PASS", "FAIL"})


class AnthropicJudge(JudgeProvider):
    """Anthropic-backed implementation of ``JudgeProvider``.

    Construction is side-effect-free (no API call, no client init).
    The Anthropic SDK client is built lazily on first ``evaluate()``.
    """

    def __init__(self, judge_config: dict[str, Any]) -> None:
        self._api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model: str = str(judge_config.get("model", _DEFAULT_MODEL))
        self._max_cost_usd: float = float(
            judge_config.get("max_cost_usd", 0.10)
        )
        self._max_session_cost_usd: float = float(
            judge_config.get("max_session_cost_usd", 1.00)
        )
        self._input_per_mtok_usd: float = float(
            judge_config.get(
                "input_cost_per_mtok_usd", _DEFAULT_INPUT_PER_MTOK_USD
            )
        )
        self._output_per_mtok_usd: float = float(
            judge_config.get(
                "output_cost_per_mtok_usd", _DEFAULT_OUTPUT_PER_MTOK_USD
            )
        )
        # Running per-pytest-session judge spend. Bounded by
        # ``max_session_cost_usd`` post-call (see module docstring).
        self._session_cost_usd: float = 0.0
        # Lazy: ``_get_client`` constructs on first use.
        self._client: anthropic.Anthropic | None = None

    # ----------------------------------------------------------- capabilities
    def capabilities(self) -> JudgeCapabilities:
        # Don't try to construct a client here — that can fail for
        # reasons unrelated to availability (e.g. transient DNS). The
        # presence of an API key is the strongest cheap signal we have.
        return JudgeCapabilities(available=bool(self._api_key))

    # ------------------------------------------------------ read-only state
    @property
    def model_name(self) -> str:
        """Pinned model identifier used in every ``messages.create`` call.

        Copied into ``JudgeRequest.model`` by the DSL so the request's
        identity tuple matches the prompt the judge actually uses.
        """
        return self._model

    @property
    def session_cost_usd(self) -> float:
        """Running judge spend for the current pytest session."""
        return self._session_cost_usd

    @property
    def max_cost_usd(self) -> float:
        """Configured per-contract-run cap (enforced by the DSL caller)."""
        return self._max_cost_usd

    @property
    def max_session_cost_usd(self) -> float:
        """Configured per-pytest-session cap (enforced inside ``evaluate``)."""
        return self._max_session_cost_usd

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, request: JudgeRequest) -> JudgeVerdict:
        # Pre-call session-budget gate (post-call overshoot policy from
        # docs/judge-assertions.md §Cost & budget). An invocation already
        # in flight always completes; the next invocation raises here.
        if self._session_cost_usd >= self._max_session_cost_usd:
            raise JudgeBudgetExceededError(
                f"per-session judge cost cap reached: "
                f"${self._session_cost_usd:.4f} >= "
                f"${self._max_session_cost_usd:.4f}. Raise "
                "[judge].max_session_cost_usd in recalllab.toml or "
                "reduce the number of judge-mode contracts in this run."
            )

        client = self._get_client()
        system_prompt, user_message = build_judge_prompt(request)

        raw_responses: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        attempts = 0

        # First attempt.
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            raise JudgeUnavailableError(
                f"Anthropic API error on initial judge call: {exc}"
            ) from exc

        attempts += 1
        raw_text = self._extract_text(response)
        raw_responses.append(raw_text[:_MAX_RAW_RESPONSE_BYTES])
        total_input_tokens += self._extract_usage(response, "input_tokens")
        total_output_tokens += self._extract_usage(response, "output_tokens")

        verdict_dict = self._try_parse(raw_text)

        if verdict_dict is None:
            # Retry once with reminder. The retry's cost is always
            # billed regardless of its outcome.
            retry_message = user_message + _RETRY_REMINDER
            attempts += 1  # Bump BEFORE the call so an API error here
            # records "tried twice" accurately, even though the retry
            # added no completed tokens — matches the doc's "attempts
            # records the call count" promise.
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": retry_message}],
                )
            except anthropic.APIError as exc:
                # Bill the first call's cost — the user's account paid
                # for it. Raise JudgePartialFailureError (not
                # JudgeUnavailableError) so the DSL caller can record a
                # failed-judge ASSERT row carrying the realized cost
                # rather than dropping it into private state.
                cost_estimate = self._cost(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    attempts=attempts,
                )
                self._session_cost_usd += cost_estimate.estimated_usd
                raise JudgePartialFailureError(
                    f"Anthropic API error on judge retry "
                    f"(first attempt was malformed JSON): {exc}",
                    cost=cost_estimate,
                    raw_responses=raw_responses,
                    underlying_error=exc,
                ) from exc

            raw_text = self._extract_text(response)
            raw_responses.append(raw_text[:_MAX_RAW_RESPONSE_BYTES])
            total_input_tokens += self._extract_usage(response, "input_tokens")
            total_output_tokens += self._extract_usage(response, "output_tokens")

            verdict_dict = self._try_parse(raw_text)

        # All API attempts settled — bill once, regardless of whether
        # parsing eventually succeeded. Both the budget cap and the
        # JudgeVerdict.cost reflect every provider call made.
        cost_estimate = self._cost(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            attempts=attempts,
        )
        self._session_cost_usd += cost_estimate.estimated_usd

        if verdict_dict is None:
            # Both attempts failed. docs/judge-assertions.md §Failed-judge
            # ASSERT lifecycle: emit a verdict with passed=False and a
            # judge_unparseable reason; the raw response bodies live on
            # raw_responses so the failure is debuggable without
            # re-querying the model.
            return JudgeVerdict(
                passed=False,
                reason=(
                    "judge_unparseable: both attempts returned content "
                    'that did not parse as {"verdict": "PASS"|"FAIL", '
                    '"reason": str}'
                ),
                cost=cost_estimate,
                raw_responses=raw_responses,
            )

        return JudgeVerdict(
            passed=verdict_dict["verdict"] == "PASS",
            reason=str(verdict_dict.get("reason", "")),
            cost=cost_estimate,
            raw_responses=raw_responses,
        )

    # ----------------------------------------------------------- internals
    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._api_key:
                # Defense in depth: capabilities() returns
                # available=False so the DSL gate should have
                # short-circuited before ``evaluate`` ran. If we still
                # get here, surface a clear error.
                raise JudgeUnavailableError(
                    "AnthropicJudge requires ANTHROPIC_API_KEY in the "
                    "environment; capabilities().available was true but "
                    "the key is now missing"
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        attempts: int,
    ) -> JudgeCostEstimate:
        usd = (
            input_tokens * self._input_per_mtok_usd / 1_000_000.0
            + output_tokens * self._output_per_mtok_usd / 1_000_000.0
        )
        return JudgeCostEstimate(
            provider="anthropic",
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_usd=usd,
            attempts=attempts,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate every text-type content block from a messages reply.

        Anthropic's messages API returns ``content`` as a list of typed
        blocks (text / tool_use / etc.). Judge calls only ever produce
        text blocks; we sum them just in case the SDK fragments the
        reply across multiple blocks.
        """
        parts: list[str] = []
        for block in getattr(response, "content", None) or ():
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage(response: Any, key: str) -> int:
        """Read ``usage.input_tokens`` / ``usage.output_tokens`` defensively.

        Returns 0 if the SDK omitted the field (e.g. a non-standard
        mock) rather than crashing on a missing attribute.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        value = getattr(usage, key, None)
        return int(value) if value is not None else 0

    @staticmethod
    def _try_parse(text: str) -> dict[str, Any] | None:
        """Strict JSON parse + Pydantic validation; ``None`` on any failure.

        Strips a single layer of markdown code fences if the model
        added one despite the instruction; doesn't try harder than
        that (a model that ignores the system prompt twice deserves
        the retry-then-fail path).
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            # Drop the opening fence (``` or ```json).
            stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -3].rstrip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        try:
            verdict = _ParsedVerdict.model_validate(parsed)
        except ValidationError:
            return None
        if verdict.verdict not in _VERDICT_VALUES:
            return None
        return verdict.model_dump()
