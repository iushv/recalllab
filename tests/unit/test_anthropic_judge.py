"""Tests for AnthropicJudge with a mocked Anthropic client.

Covers:

- ``capabilities()`` driven by ``ANTHROPIC_API_KEY`` presence.
- Happy path: valid JSON on the first attempt → ``JudgeVerdict`` with
  ``attempts=1`` and accumulated cost.
- Malformed-JSON retry: first attempt returns garbage, retry returns
  valid JSON → ``attempts=2`` and cost summed across both calls.
- Both attempts malformed → ``passed=False`` with
  ``reason="judge_unparseable..."`` and ``attempts=2``.
- ``anthropic.APIError`` on first call → ``JudgeUnavailableError``;
  no session cost incurred.
- ``anthropic.APIError`` on retry → ``JudgeUnavailableError``; first
  call's cost IS billed to the session.
- Per-session cap → ``JudgeBudgetExceededError`` *before* any new
  provider call when running total >= cap.

The mocked client is injected via ``judge._client`` to bypass real
network init. No real API calls are made in any test.
"""

from __future__ import annotations

import os
import unittest.mock
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest

from recalllab.core.judge.anthropic import AnthropicJudge
from recalllab.core.judge.base import (
    JudgeBudgetExceededError,
    JudgeMode,
    JudgePartialFailureError,
    JudgeRequest,
    JudgeUnavailableError,
)

# --------------------------------------------------------- builders


def _mock_response(text: str, *, input_tokens: int = 100, output_tokens: int = 20) -> Any:
    """Build a fake ``anthropic.types.Message`` good enough for ``_extract_text``
    and ``_extract_usage``."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


def _make_judge(
    *,
    api_key: str | None = "test-anthropic-key",
    config: dict[str, Any] | None = None,
) -> AnthropicJudge:
    """Build an AnthropicJudge with ``ANTHROPIC_API_KEY`` set via env-patch."""
    full_config = {
        "provider": "anthropic",
        "model": "test-model",
        # Tight pricing so we can read cost arithmetic in tests easily.
        "input_cost_per_mtok_usd": 1.00,
        "output_cost_per_mtok_usd": 5.00,
        "max_cost_usd": 0.10,
        "max_session_cost_usd": 1.00,
    }
    if config:
        full_config.update(config)
    env: dict[str, str] = {}
    if api_key is None:
        # Remove the env var entirely for capabilities() == False tests.
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            judge = AnthropicJudge(full_config)
            # Restore for the rest of the test session — patch.dict
            # cleanup restores on context exit so we need to reapply.
        return judge
    env["ANTHROPIC_API_KEY"] = api_key
    with unittest.mock.patch.dict(os.environ, env, clear=False):
        return AnthropicJudge(full_config)


def _attach_mock_client(judge: AnthropicJudge, responses: list[Any]) -> MagicMock:
    """Wire a mock client into the judge that returns ``responses`` in order.

    Returns the mock so tests can inspect call args.
    """
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses
    judge._client = mock_client
    return mock_client


def _req(*, mode: JudgeMode = JudgeMode.LATEST_FACT_IS) -> JudgeRequest:
    return JudgeRequest(
        query="Where do I live?",
        recall_result="ayush lives in Mumbai",
        expected="Mumbai",
        rubric=None,
        model="test-model",
        mode=mode,
    )


# ------------------------------------------------------- capabilities


def test_capabilities_true_with_api_key() -> None:
    judge = _make_judge(api_key="key")
    assert judge.capabilities().available is True


def test_capabilities_false_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = AnthropicJudge({"provider": "anthropic"})
    assert judge.capabilities().available is False


# ------------------------------------------------------- happy path


def test_happy_path_single_call_pass() -> None:
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [_mock_response('{"verdict": "PASS", "reason": "looks good"}')],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is True
    assert verdict.reason == "looks good"
    assert verdict.cost.attempts == 1
    assert verdict.cost.input_tokens == 100
    assert verdict.cost.output_tokens == 20
    # ($1/Mtok x 100) + ($5/Mtok x 20) = $0.0001 + $0.0001 = $0.0002
    assert verdict.cost.estimated_usd == pytest.approx(0.0002)
    assert judge.session_cost_usd == pytest.approx(0.0002)
    assert len(verdict.raw_responses) == 1


def test_happy_path_single_call_fail() -> None:
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [_mock_response('{"verdict": "FAIL", "reason": "wrong city"}')],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is False
    assert verdict.reason == "wrong city"


# ----------------------------------------------------- malformed-JSON retry


def test_malformed_json_retry_succeeds() -> None:
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [
            _mock_response("garbage not json", input_tokens=80, output_tokens=15),
            _mock_response('{"verdict": "PASS", "reason": "ok"}', input_tokens=120, output_tokens=18),
        ],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is True
    assert verdict.cost.attempts == 2
    assert verdict.cost.input_tokens == 200
    assert verdict.cost.output_tokens == 33
    # Both calls billed to the session.
    assert verdict.cost.estimated_usd == pytest.approx(
        (200 * 1.00 + 33 * 5.00) / 1_000_000
    )
    assert judge.session_cost_usd == pytest.approx(verdict.cost.estimated_usd)
    assert len(verdict.raw_responses) == 2


def test_both_attempts_malformed_returns_unparseable_verdict() -> None:
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [
            _mock_response("not json", input_tokens=50, output_tokens=10),
            _mock_response("still not json", input_tokens=60, output_tokens=12),
        ],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is False
    assert "judge_unparseable" in verdict.reason
    assert verdict.cost.attempts == 2
    assert len(verdict.raw_responses) == 2
    # Session cost includes both calls.
    expected_usd = (110 * 1.00 + 22 * 5.00) / 1_000_000
    assert judge.session_cost_usd == pytest.approx(expected_usd)


def test_invalid_verdict_value_triggers_retry() -> None:
    """A JSON response whose verdict is not PASS/FAIL is parse-failure.
    The model returning '{"verdict": "MAYBE", "reason": ...}' should
    NOT silently be treated as PASS."""
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [
            _mock_response('{"verdict": "MAYBE", "reason": "uncertain"}'),
            _mock_response('{"verdict": "FAIL", "reason": "clarified"}'),
        ],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is False
    assert verdict.reason == "clarified"
    assert verdict.cost.attempts == 2


def test_markdown_fenced_response_is_unwrapped() -> None:
    """The model sometimes ignores 'no markdown fences' and wraps the
    JSON anyway. The parser unwraps a single ``` fence."""
    judge = _make_judge()
    _attach_mock_client(
        judge,
        [_mock_response('```json\n{"verdict": "PASS", "reason": "ok"}\n```')],
    )

    verdict = judge.evaluate(_req())

    assert verdict.passed is True
    assert verdict.cost.attempts == 1


# ----------------------------------------------------- API errors


def test_first_call_api_error_raises_unavailable() -> None:
    judge = _make_judge()
    api_error = anthropic.APIError(
        message="boom",
        request=MagicMock(),
        body=None,
    )
    _attach_mock_client(judge, [api_error])

    with pytest.raises(JudgeUnavailableError, match="initial judge call"):
        judge.evaluate(_req())
    # No call completed, so no cost charged.
    assert judge.session_cost_usd == 0.0


def test_retry_api_error_raises_partial_failure_with_realized_cost() -> None:
    """Codex round-1 step-4 finding: retry API errors must NOT trap the
    partial cost in private state and raise the same exception type as
    "nothing was billed." They raise JudgePartialFailureError, which carries
    the realized cost + raw_responses + attempts so the DSL caller
    (step 5+) can emit a failed-judge ASSERT row reflecting what the
    user actually paid for."""
    judge = _make_judge()
    api_error = anthropic.APIError(
        message="boom",
        request=MagicMock(),
        body=None,
    )
    _attach_mock_client(
        judge,
        [
            _mock_response("garbage", input_tokens=100, output_tokens=20),
            api_error,
        ],
    )

    with pytest.raises(JudgePartialFailureError, match="judge retry") as exc_info:
        judge.evaluate(_req())

    # First call DID hit the API and produced tokens; the user's
    # account paid for it, so the session counter must reflect it AND
    # the exception must carry the same cost so the caller can record it.
    expected_usd = (100 * 1.00 + 20 * 5.00) / 1_000_000
    assert judge.session_cost_usd == pytest.approx(expected_usd)
    assert exc_info.value.cost.estimated_usd == pytest.approx(expected_usd)
    # attempts=2 because the retry call was attempted (even though it
    # errored out without billing tokens). Matches the doc's
    # "attempts records the call count" promise.
    assert exc_info.value.cost.attempts == 2
    # raw_responses carries the first (malformed) reply for debugging.
    assert len(exc_info.value.raw_responses) == 1
    assert "garbage" in exc_info.value.raw_responses[0]
    # underlying_error is the APIError so callers can re-raise or chain.
    assert isinstance(exc_info.value.underlying_error, anthropic.APIError)


# ----------------------------------------------------- budget enforcement


def test_session_cap_blocks_next_call() -> None:
    """Post-call enforcement: once running total >= cap, the NEXT
    evaluate() raises before any new API call."""
    judge = _make_judge(config={"max_session_cost_usd": 0.0001})
    # First call: input=100 / output=20 → $0.0002 — exceeds the $0.0001 cap.
    mock_client = _attach_mock_client(
        judge,
        [_mock_response('{"verdict": "PASS", "reason": "ok"}')],
    )

    # First call completes (post-call policy: in-flight always finishes).
    verdict = judge.evaluate(_req())
    assert verdict.passed is True
    assert judge.session_cost_usd == pytest.approx(0.0002)

    # Second call must refuse before issuing any new request.
    assert mock_client.messages.create.call_count == 1
    with pytest.raises(JudgeBudgetExceededError, match="per-session"):
        judge.evaluate(_req())
    # Still 1 call — refused before issuing.
    assert mock_client.messages.create.call_count == 1


def test_session_cap_includes_retry_overshoot() -> None:
    """An invocation with a malformed-JSON retry counts BOTH calls
    against the session cap when reporting overshoot. Per Decision: the
    unit of enforcement is a judge invocation (incl. retries), not a
    call."""
    # Cap chosen so the FIRST call alone would have crossed it, proving
    # the in-flight invocation (incl. retry) always completes — the
    # bounded-overshoot guarantee.
    judge = _make_judge(config={"max_session_cost_usd": 0.0001})
    mock_client = _attach_mock_client(
        judge,
        [
            _mock_response("garbage", input_tokens=100, output_tokens=20),
            _mock_response('{"verdict": "PASS", "reason": "ok"}',
                            input_tokens=150, output_tokens=25),
        ],
    )

    verdict = judge.evaluate(_req())
    # Both calls billed.
    expected = ((100 + 150) * 1.00 + (20 + 25) * 5.00) / 1_000_000
    assert verdict.cost.attempts == 2
    assert judge.session_cost_usd == pytest.approx(expected)
    # The invocation completed even though its cost ($0.000475) far
    # exceeds the $0.0001 cap. The next invocation refuses.
    assert judge.session_cost_usd > judge.max_session_cost_usd
    with pytest.raises(JudgeBudgetExceededError):
        judge.evaluate(_req())
    # No additional API call was issued.
    assert mock_client.messages.create.call_count == 2


def test_evaluate_without_api_key_raises_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: if capabilities() said False, the DSL gate normally
    short-circuits; this test pins ``evaluate`` raising cleanly when
    bypassed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = AnthropicJudge({"provider": "anthropic"})

    with pytest.raises(JudgeUnavailableError, match="ANTHROPIC_API_KEY"):
        judge.evaluate(_req())


# ----------------------------------------------------- prompt routing


def test_evaluate_passes_system_prompt_and_envelope_to_client() -> None:
    """End-to-end: the call to ``client.messages.create`` carries the
    system prompt from ``build_judge_prompt`` and the envelope-wrapped
    user message."""
    judge = _make_judge()
    mock_client = _attach_mock_client(
        judge,
        [_mock_response('{"verdict": "PASS", "reason": "ok"}')],
    )

    judge.evaluate(_req())

    create_kwargs = mock_client.messages.create.call_args.kwargs
    assert create_kwargs["model"] == "test-model"
    assert create_kwargs["temperature"] == 0
    # System prompt names the assertion mode (sanity check that the
    # right rubric was selected).
    assert "latest_fact_is" in create_kwargs["system"]
    user_message = create_kwargs["messages"][0]["content"]
    assert user_message.startswith("<recall_result_")
    assert "Mumbai" in user_message
