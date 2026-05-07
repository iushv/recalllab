"""Contract DSL — the user-facing API tests are written against.

The fixture exposed by the pytest plugin is an instance of ``MemoryContract``.
Each call records a ``TraceEvent`` on the underlying ``ContractRun`` so the
full conversation is debuggable from the Failure Gallery.

v0.1 ships rule-based assertion modes (``contains`` / ``excludes``) only;
the judge-driven modes (``latest_fact_is``, ``must_not_answer_as``,
``judge_assertion``) land in a follow-up task and are gated on a configured
``[judge]`` section in ``recalllab.toml``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from recalllab.core.traces.schema import (
    AssertionResult,
    ContractRun,
    EventKind,
    RunStatus,
    TraceEvent,
)

if TYPE_CHECKING:
    from recalllab.adapters.base import Episode, MemoryProvider, Recalled


class MemoryContract:
    """The DSL object passed to every contract test as the ``memory_contract`` fixture.

    Methods are chainable where it reads naturally (``given_user`` returns
    ``self``); assertion methods raise ``AssertionError`` on failure so pytest
    surfaces the failure in the standard way.
    """

    def __init__(self, provider: MemoryProvider, run: ContractRun) -> None:
        self._provider = provider
        self._run = run
        self._user_id: str | None = None
        self._sequence = 0

    # ---------------------------------------------------------------- introspection
    @property
    def run(self) -> ContractRun:
        """The mutable ``ContractRun`` being recorded; the pytest reporter persists it."""
        return self._run

    @property
    def provider(self) -> MemoryProvider:
        return self._provider

    # ------------------------------------------------------------------- DSL: setup
    def given_user(self, user_id: str) -> MemoryContract:
        """Set the active user for subsequent operations."""
        self._user_id = user_id
        self._record_event(EventKind.GIVEN_USER, {"user_id": user_id})
        return self

    # ----------------------------------------------------------------- DSL: actions
    def remember(self, text: str) -> Episode:
        """Write a memory for the active user."""
        user_id = self._require_user()
        start = time.perf_counter()
        episode = self._provider.remember(user_id, text)
        latency_ms = (time.perf_counter() - start) * 1000
        self._record_event(
            EventKind.REMEMBER,
            {
                "user_id": user_id,
                "text": text,
                "episode_id": episode.id,
            },
            latency_ms=latency_ms,
        )
        return episode

    def recall(self, query: str, *, k: int = 5) -> list[Recalled]:
        """Search for memories matching the query for the active user."""
        user_id = self._require_user()
        start = time.perf_counter()
        results = self._provider.recall(user_id, query, k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        self._record_event(
            EventKind.RECALL,
            {
                "user_id": user_id,
                "query": query,
                "k": k,
                "results": [r.model_dump() for r in results],
            },
            latency_ms=latency_ms,
        )
        return results

    def forget(
        self,
        *,
        matching: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        """Delete memories for the active user. Returns the deletion count."""
        user_id = self._require_user()
        if matching is None and episode_id is None:
            raise ValueError("forget() requires either matching= or episode_id=")
        start = time.perf_counter()
        deleted = self._provider.forget(user_id, matching=matching, episode_id=episode_id)
        latency_ms = (time.perf_counter() - start) * 1000
        self._record_event(
            EventKind.FORGET,
            {
                "user_id": user_id,
                "matching": matching,
                "episode_id": episode_id,
                "deleted": deleted,
            },
            latency_ms=latency_ms,
        )
        return deleted

    # -------------------------------------------------------------- DSL: assertions
    def should_recall(
        self,
        query: str,
        *,
        k: int = 5,
        contains: str | list[str] | None = None,
        excludes: str | list[str] | None = None,
    ) -> list[Recalled]:
        """Run a recall and assert against the joined response text.

        ``contains`` — at least one of the listed values must appear (case-insensitive)
        in the recalled text. Vacuously ``False`` when nothing is recalled.

        ``excludes`` — none of the listed values may appear (case-insensitive).
        Vacuously ``True`` when nothing is recalled.

        Both can be set in the same call; they're evaluated independently.
        """
        if contains is None and excludes is None:
            raise ValueError(
                "should_recall() needs at least one of contains= or excludes="
            )
        results = self.recall(query, k=k)
        joined = "\n".join(r.text for r in results)
        if contains is not None:
            self._assert_contains(joined, contains)
        if excludes is not None:
            self._assert_excludes(joined, excludes)
        return results

    # ---------------------------------------------------------------- internal API
    def _require_user(self) -> str:
        if self._user_id is None:
            raise RuntimeError(
                "no active user — call memory_contract.given_user(user_id) first"
            )
        return self._user_id

    def _next_sequence(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq

    def _record_event(
        self,
        kind: EventKind,
        payload: dict[str, object],
        *,
        latency_ms: float | None = None,
    ) -> None:
        self._run.events.append(
            TraceEvent(
                sequence=self._next_sequence(),
                kind=kind,
                payload=dict(payload),
                timestamp=datetime.now(tz=UTC),
                latency_ms=latency_ms,
            )
        )

    def _record_assertion(
        self,
        *,
        passed: bool,
        mode: str,
        expected: object,
        actual: str,
        reason: str | None = None,
    ) -> None:
        seq = self._next_sequence()
        self._run.assertions.append(
            AssertionResult(
                passed=passed,
                mode=mode,
                expected=expected,
                actual=actual,
                reason=reason,
                sequence=seq,
            )
        )
        self._run.events.append(
            TraceEvent(
                sequence=seq,
                kind=EventKind.ASSERT,
                payload={
                    "mode": mode,
                    "expected": expected,
                    "passed": passed,
                    "reason": reason,
                },
                timestamp=datetime.now(tz=UTC),
            )
        )
        if not passed:
            self._run.status = RunStatus.FAILED

    def _assert_contains(self, actual: str, expected: str | list[str]) -> None:
        expected_list = [expected] if isinstance(expected, str) else list(expected)
        actual_lower = actual.lower()
        matched = [e for e in expected_list if e.lower() in actual_lower]
        if matched:
            self._record_assertion(
                passed=True, mode="contains", expected=expected, actual=actual
            )
            return
        reason = (
            f"none of {expected_list} found in recall response"
            if expected_list
            else "no expected values supplied"
        )
        self._record_assertion(
            passed=False,
            mode="contains",
            expected=expected,
            actual=actual,
            reason=reason,
        )
        raise AssertionError(
            f"contains assertion failed: expected any of {expected_list!r} "
            f"in recall response, got: {actual!r}"
        )

    def _assert_excludes(self, actual: str, expected: str | list[str]) -> None:
        expected_list = [expected] if isinstance(expected, str) else list(expected)
        actual_lower = actual.lower()
        leaked = [e for e in expected_list if e.lower() in actual_lower]
        if not leaked:
            self._record_assertion(
                passed=True, mode="excludes", expected=expected, actual=actual
            )
            return
        reason = f"found {leaked} in recall response (expected none)"
        self._record_assertion(
            passed=False,
            mode="excludes",
            expected=expected,
            actual=actual,
            reason=reason,
        )
        raise AssertionError(
            f"excludes assertion failed: {leaked!r} present in recall response, "
            f"expected absent. Full response: {actual!r}"
        )
