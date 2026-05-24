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

import hashlib
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from recalllab.adapters.base import UnconfirmedRemoteWriteError
from recalllab.core.judge import (
    JudgeProvider,
    JudgeUnavailableError,
    NoOpJudge,
    Rubric,
)
from recalllab.core.mutations import sample_distractors, validate_seed
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

    def __init__(
        self,
        provider: MemoryProvider,
        run: ContractRun,
        *,
        judge: JudgeProvider | None = None,
        judge_optional: bool = False,
    ) -> None:
        self._provider = provider
        self._run = run
        # Default to NoOpJudge so the fail-loud gate in should_recall works
        # uniformly regardless of who constructed the contract (production
        # fixture supplies the configured judge; unit tests can pass None).
        self._judge: JudgeProvider = judge if judge is not None else NoOpJudge()
        # True when the contract is decorated with
        # ``@pytest.mark.recalllab_optional("judge_configured")``; controls
        # whether the gate skips or raises. Set by the pytest fixture; unit
        # tests can flip it directly.
        self._judge_optional = judge_optional
        self._user_id: str | None = None
        self._sequence = 0
        # Counter for mutation invocations, used as part of the deterministic
        # episode-id hash so calling the same mutation twice in one contract
        # (e.g. for two different users) does not collide.
        self._mutation_invocation = 0
        # Fingerprint → invocation map for mutations that have NOT yet
        # completed successfully. A partial-failed mutation leaves its
        # fingerprint here so that retrying the same mutation under the
        # same user with the same key (seed for distractors, source-text
        # hash for stale repeats) reuses the original invocation number —
        # which makes the deterministic episode IDs match the orphan rows
        # left in the store. Idempotent adapters (reference, langgraph)
        # return existing episodes for the writes that already landed, so
        # the retry resumes from the failure point rather than
        # double-writing. Successful completion clears the fingerprint,
        # so a *deliberate* re-call with the same args gets a fresh
        # invocation (distinct IDs) — preserving the round-2 property
        # that two deliberate calls of the same mutation under one user
        # don't collide.
        self._in_flight_mutations: dict[tuple[str, str, str], int] = {}

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
    def remember(self, text: str, *, episode_id: str | None = None) -> Episode:
        """Write a memory for the active user.

        ``episode_id`` is optional and rarely needed by hand-written
        contracts — the provider assigns an id by default. It exists for
        ``recalllab record``-generated regressions: the trace records the
        exact id the original run produced, and the regenerated test
        passes it back so any later ``forget(episode_id=...)`` or
        ID-paired check in the same contract actually addresses the
        same row. Passing a custom id against an adapter without
        ``supports_custom_episode_ids`` will not error here but may
        silently degrade to provider-assigned ids — check the adapter
        before relying on it.

        ``episode_id`` is forwarded to ``provider.remember`` ONLY when
        the caller supplied one. Always forwarding it as a keyword
        would break legacy third-party adapters whose ``remember``
        signature is ``(user_id, text)`` — every ordinary
        ``memory_contract.remember("...")`` would raise
        ``TypeError: unexpected keyword argument 'episode_id'`` even
        though the new feature isn't being used. The
        ``MemoryProvider`` Protocol is runtime-checkable and Python's
        structural typing checks method *names*, not signatures, so a
        mismatch passes ``isinstance(x, MemoryProvider)`` but fails at
        call time. Skipping the kwarg in the None case keeps the v0.1
        provider surface working unchanged — round-11 Codex finding.
        """
        user_id = self._require_user()
        start = time.perf_counter()
        if episode_id is None:
            episode = self._provider.remember(user_id, text)
        else:
            episode = self._provider.remember(
                user_id, text, episode_id=episode_id
            )
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

    # ------------------------------------------------------------- DSL: mutations
    def with_distractors(self, n: int, *, seed: int = 0) -> MemoryContract:
        """Inject ``n`` deterministic distractor episodes into the active user's namespace.

        Stresses retrieval by polluting the namespace with unrelated content
        before the next recall. The distractor pool is fixed; the same
        ``(n, seed)`` pair always produces the same texts in the same order
        and the same requested episode IDs (deterministic from the
        ``contract_id`` + mutation type + seed + index). Distractors are
        scoped to the active user — every write goes through
        ``provider.remember(user_id, ...)`` and cannot leak across tenants.

        ``seed`` defaults to ``0`` so traces are replayable without callers
        having to remember to supply a seed. Pass an explicit non-zero seed
        to vary the distractor sample.

        On a mid-mutation provider exception the partial set of inserted
        episode IDs is recorded with ``status="partial_failed"`` and the
        original exception re-raised. That keeps the trace honest about
        orphan writes against hosted providers (Mem0, Zep, custom MCP).
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        user_id = self._require_user()
        seed = validate_seed(seed)
        # Fingerprint key includes ``n`` so two calls with the same seed
        # but different counts are NOT misclassified as retries of each
        # other (round-5 Codex finding). Without ``n`` in the key, a
        # later ``with_distractors(5, seed=0)`` would clear the in-flight
        # slot left by a partial-failed ``with_distractors(10, seed=0)``,
        # recording a completed mutation with the wrong requested count.
        key = f"{seed}|{n}"
        fingerprint = ("distractors", user_id, key)
        invocation = self._allocate_or_resume_invocation(fingerprint)
        texts = sample_distractors(n, seed=seed)
        inserted_episode_ids: list[str] = []
        requested_episode_ids: list[str] = []
        unconfirmed_writes: list[str] = []
        abandoned_in_flight = self._abandoned_in_flight_for(
            mutation_type="distractors",
            user_id=user_id,
            current_key=key,
        )
        payload: dict[str, Any] = {
            "type": "distractors",
            "user_id": user_id,
            "seed": seed,
            "requested": n,
            "invocation": invocation,
            "requested_episode_ids": requested_episode_ids,
            "inserted_episode_ids": inserted_episode_ids,
            "unconfirmed_writes": unconfirmed_writes,
            "abandoned_in_flight": abandoned_in_flight,
            "status": "completed",
        }
        self._require_custom_episode_ids_for_mutation(payload, write_count=n)
        try:
            for index, text in enumerate(texts):
                requested_id = self._mutation_episode_id(
                    mutation_type="distractors",
                    user_id=user_id,
                    invocation=invocation,
                    key=key,
                    index=index,
                )
                requested_episode_ids.append(requested_id)
                self._remember_mutation_episode(
                    user_id=user_id,
                    text=text,
                    requested_id=requested_id,
                    inserted_episode_ids=inserted_episode_ids,
                    unconfirmed_writes=unconfirmed_writes,
                )
        except Exception as exc:
            # Do NOT clear the fingerprint — leaving it in-flight is what
            # makes a retry reuse this same invocation number, so the
            # deterministic IDs match the orphan rows from this attempt.
            payload["status"] = "partial_failed"
            payload["error"] = repr(exc)
            self._record_event(EventKind.MUTATION, payload)
            raise
        self._mark_mutation_completed(fingerprint)
        self._record_event(EventKind.MUTATION, payload)
        return self

    def with_stale_repeats(self, *, times: int) -> MemoryContract:
        """Repeat the most recent ``remember`` for the active user ``times`` more times.

        Stresses temporal-update contracts by amplifying the stale fact in
        the namespace before the corrective remember lands. The "most
        recent remember" is resolved by walking back through trace events
        for the active user, so this only operates on the *current* user
        even if other users have been seeded in the same contract.

        Repeat episode IDs are deterministic from ``contract_id`` + mutation
        type + a stable hash of the source text + index, so retrying a
        contract against a hosted provider does not silently double-write
        with fresh UUIDs.

        Same partial-failure semantics as ``with_distractors``: the trace
        records ``inserted_episode_ids`` + ``status`` + ``error`` and
        re-raises on mid-mutation exceptions.
        """
        if times < 0:
            raise ValueError("times must be non-negative")
        user_id = self._require_user()
        last_remember = self._last_remember_for(user_id)
        if last_remember is None:
            raise RuntimeError(
                "with_stale_repeats() needs a prior remember() call for the "
                "active user; none found"
            )
        text = last_remember.payload.get("text")
        if not isinstance(text, str):
            raise RuntimeError(
                "with_stale_repeats(): the most recent remember event for "
                f"{user_id!r} has no text payload"
            )
        source_episode_id = last_remember.payload.get("episode_id")
        # Resurrection guard: refuse to repeat a source remember that has
        # already been deleted in this contract. Without this guard a
        # ``remember(X) → forget(matching=X) → with_stale_repeats(times=1)``
        # sequence would re-write the forgotten text under a mutation id and
        # silently break forget / privacy contracts.
        #
        # The check is two-layered so it does not rely on provider behaviour
        # the protocol does not promise:
        #
        # 1. **Trace-based (always on, works on every provider).** Walk
        #    FORGET events that happened *after* the source REMEMBER. If any
        #    FORGET targeted the source by id, or used a ``matching=`` whose
        #    substring is contained in the source text, the source is
        #    considered deleted and we raise. This catches every in-contract
        #    deletion path regardless of the adapter — reference, langgraph,
        #    or MCP.
        # 2. **Provider-side (gated on capability).** Only adapters that
        #    declare ``supports_authoritative_list=True`` get the additional
        #    ``provider.list_episodes`` check. The MCP adapter's
        #    ``list_episodes`` is a best-effort wildcard ``recall`` and many
        #    memory servers reject empty queries or cap the result set; using
        #    that listing as a liveness oracle would falsely fail every
        #    stale-repeat call. Authoritative-list providers run the extra
        #    check for defense in depth against provider-side eviction.
        if self._source_remember_deleted_in_trace(
            user_id=user_id,
            source_event=last_remember,
            source_text=text,
            source_episode_id=source_episode_id,
        ):
            raise RuntimeError(
                f"with_stale_repeats(): the source remember "
                f"{source_episode_id!r} for user {user_id!r} has been "
                f"forgotten in this contract; refusing to resurrect "
                f"deleted content. If you want to amplify a different "
                f"fact, call remember(...) again to make it the most "
                f"recent remember for this user."
            )
        if (
            self._provider.capabilities().supports_authoritative_list
            and isinstance(source_episode_id, str)
        ):
            live_episode_ids = {
                ep.id for ep in self._provider.list_episodes(user_id)
            }
            if source_episode_id not in live_episode_ids:
                raise RuntimeError(
                    f"with_stale_repeats(): the source remember "
                    f"{source_episode_id!r} for user {user_id!r} is no "
                    f"longer present in the provider's authoritative "
                    f"listing; refusing to resurrect deleted content."
                )
        text_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        # Fingerprint key includes ``times`` and the source remember's
        # *ordinal* among prior remembers of this (user_id, text) — i.e.
        # how many earlier REMEMBER events also matched. Round-5 Codex
        # finding: ``(type, user, text_hash)`` alone would let a later
        # ``with_stale_repeats(times=2)`` clear a prior partial-failed
        # ``with_stale_repeats(times=5)``'s in-flight slot, recording a
        # completed mutation with the wrong requested count. Round-8
        # Codex finding: using the source's ``sequence`` for that
        # discrimination is brittle — adding any unrelated earlier event
        # (a setup remember for another user, an extra recall) shifts
        # the sequence and breaks the deterministic IDs the next run
        # requests, so retries against a persistent store double-write
        # instead of deduping. The ordinal of "which remember of this
        # text is this" is invariant under unrelated edits, distinct
        # between back-to-back same-text remembers, and stable across
        # fresh runs of the same contract code.
        prior_same_text_remembers = sum(
            1
            for e in self._run.events
            if e.sequence < last_remember.sequence
            and e.kind == EventKind.REMEMBER
            and e.payload.get("user_id") == user_id
            and e.payload.get("text") == text
        )
        key = f"{text_key}|{times}|{prior_same_text_remembers}"
        fingerprint = ("stale_repeats", user_id, key)
        invocation = self._allocate_or_resume_invocation(fingerprint)
        inserted_episode_ids: list[str] = []
        requested_episode_ids: list[str] = []
        unconfirmed_writes: list[str] = []
        abandoned_in_flight = self._abandoned_in_flight_for(
            mutation_type="stale_repeats",
            user_id=user_id,
            current_key=key,
        )
        payload: dict[str, Any] = {
            "type": "stale_repeats",
            "user_id": user_id,
            "times": times,
            "invocation": invocation,
            "source_episode_id": last_remember.payload.get("episode_id"),
            "requested_episode_ids": requested_episode_ids,
            "inserted_episode_ids": inserted_episode_ids,
            "unconfirmed_writes": unconfirmed_writes,
            "abandoned_in_flight": abandoned_in_flight,
            "status": "completed",
        }
        self._require_custom_episode_ids_for_mutation(payload, write_count=times)
        try:
            for index in range(times):
                requested_id = self._mutation_episode_id(
                    mutation_type="stale_repeats",
                    user_id=user_id,
                    invocation=invocation,
                    key=key,
                    index=index,
                )
                requested_episode_ids.append(requested_id)
                self._remember_mutation_episode(
                    user_id=user_id,
                    text=text,
                    requested_id=requested_id,
                    inserted_episode_ids=inserted_episode_ids,
                    unconfirmed_writes=unconfirmed_writes,
                )
        except Exception as exc:
            # Same recovery semantics as with_distractors — leave the
            # fingerprint in-flight so a retry reuses this invocation.
            payload["status"] = "partial_failed"
            payload["error"] = repr(exc)
            self._record_event(EventKind.MUTATION, payload)
            raise
        self._mark_mutation_completed(fingerprint)
        self._record_event(EventKind.MUTATION, payload)
        return self

    # --------------------------------------------------------- mutation helpers
    def _allocate_or_resume_invocation(
        self, fingerprint: tuple[str, str, str]
    ) -> int:
        """Return the invocation number for a mutation fingerprint.

        Fingerprint is ``(mutation_type, user_id, key)`` where ``key`` is
        the seed for distractors or the source-text hash for stale
        repeats. If the fingerprint is currently in flight (the prior
        attempt partial-failed and has not yet completed), the original
        invocation is reused — so the deterministic episode IDs on retry
        match the orphan rows from the failed attempt, and idempotent
        adapters dedupe rather than double-writing.

        Otherwise the per-contract counter advances and the new
        invocation is registered as in-flight; it will be cleared by
        ``_mark_mutation_completed`` when the mutation succeeds end-to-end.
        """
        existing = self._in_flight_mutations.get(fingerprint)
        if existing is not None:
            return existing
        self._mutation_invocation += 1
        self._in_flight_mutations[fingerprint] = self._mutation_invocation
        return self._mutation_invocation

    def _abandoned_in_flight_for(
        self,
        *,
        mutation_type: str,
        user_id: str,
        current_key: str,
    ) -> list[dict[str, Any]]:
        """Return in-flight fingerprints the current mutation will NOT retry.

        When a partial-failed mutation under user ``X`` for key ``K1`` is
        followed by a different-key mutation for the same user (because
        e.g. a fresh ``remember`` shifted the source remember's
        ordinal), the new call won't dedupe the orphan rows from the
        failed attempt. Surfacing the abandoned fingerprints in the new
        mutation's ``MUTATION`` payload keeps the trace honest about
        orphan state the retry left behind — round-11 Codex finding.

        Returns ``[]`` when there's no abandonment (the normal happy
        path). Each entry is a dict with ``mutation_type``, ``key``, and
        ``invocation`` so the Failure Gallery can render the orphan
        episode IDs from the original ``MUTATION`` partial_failed event.
        """
        return [
            {"mutation_type": fp[0], "key": fp[2], "invocation": inv}
            for fp, inv in self._in_flight_mutations.items()
            if fp[0] == mutation_type
            and fp[1] == user_id
            and fp[2] != current_key
        ]

    def _mark_mutation_completed(
        self, fingerprint: tuple[str, str, str]
    ) -> None:
        """Clear an in-flight mutation fingerprint after successful completion.

        After this call, a subsequent invocation with the same
        fingerprint will get a fresh invocation number — preserving the
        "two deliberate calls of the same mutation under one user do not
        collide" property.
        """
        self._in_flight_mutations.pop(fingerprint, None)

    def _source_remember_deleted_in_trace(
        self,
        *,
        user_id: str,
        source_event: TraceEvent,
        source_text: str,
        source_episode_id: Any,
    ) -> bool:
        """Trace-based liveness check for ``with_stale_repeats``'s source.

        Walks FORGET events that occurred *after* the source REMEMBER. A
        FORGET is considered to have deleted the source if either:

        - ``forget(episode_id=...)`` cited the source's episode id, or
        - ``forget(matching=...)`` used a substring contained in the
          source text. The reference and LangGraph adapters' ``forget``
          deletes *every* episode matching the substring, so any
          matching forget would have removed the source. (Conservative
          for unknown MCP semantics: better to refuse than resurrect.)

        Returns ``True`` only when we can prove a deletion occurred
        after this specific REMEMBER. The "delete then re-create same
        text" case is handled correctly because ``_last_remember_for``
        returns the most recent REMEMBER, and the walk starts after
        *that* event — earlier forgets are out of scope.
        """
        source_text_lower = source_text.lower()
        for event in self._run.events:
            if event.sequence <= source_event.sequence:
                continue
            if event.kind != EventKind.FORGET:
                continue
            if event.payload.get("user_id") != user_id:
                continue
            forget_id = event.payload.get("episode_id")
            if (
                isinstance(source_episode_id, str)
                and isinstance(forget_id, str)
                and forget_id == source_episode_id
            ):
                return True
            forget_matching = event.payload.get("matching")
            if (
                isinstance(forget_matching, str)
                and forget_matching.lower() in source_text_lower
            ):
                return True
        return False

    def _require_custom_episode_ids_for_mutation(
        self,
        payload: dict[str, Any],
        *,
        write_count: int,
    ) -> None:
        """Fail loudly when mutation idempotency cannot be guaranteed."""
        if write_count == 0:
            return
        if self._provider.capabilities().supports_custom_episode_ids:
            return
        reason = (
            f"provider {type(self._provider).__name__} does not declare "
            "supports_custom_episode_ids; mutation retries would not be "
            "idempotent"
        )
        payload["status"] = "unsupported"
        payload["error"] = reason
        self._record_event(EventKind.MUTATION, payload)
        raise RuntimeError(reason)

    def _remember_mutation_episode(
        self,
        *,
        user_id: str,
        text: str,
        requested_id: str,
        inserted_episode_ids: list[str],
        unconfirmed_writes: list[str],
    ) -> Episode:
        """Remember a mutation episode and verify the provider honoured the ID.

        Three failure modes the trace must distinguish:

        - **Confirmed insert.** Provider returns an ``Episode`` whose ``id``
          matches ``requested_id``. Append to ``inserted_episode_ids``.
        - **Mismatched id (lying provider).** Provider returns an ``Episode``
          whose ``id`` differs. The actual id is recorded in
          ``inserted_episode_ids`` (so the trace knows what landed), then
          we raise — the mutation pipeline's outer except logs
          ``partial_failed`` and the user sees a clear error.
        - **Unconfirmed remote write.** Adapter raises
          ``UnconfirmedRemoteWriteError`` because the upstream provider was
          already called but the response is unusable. The remote store
          may have a row at ``requested_id``; we add it to
          ``unconfirmed_writes`` and re-raise. Without this layer the
          trace would silently report an empty inserted list while the
          server held an orphan.
        """
        try:
            episode = self._provider.remember(
                user_id, text, episode_id=requested_id
            )
        except UnconfirmedRemoteWriteError as exc:
            unconfirmed_writes.append(exc.requested_episode_id)
            raise
        inserted_episode_ids.append(episode.id)
        if episode.id != requested_id:
            raise RuntimeError(
                f"provider {type(self._provider).__name__} returned episode_id "
                f"{episode.id!r} after RecallLab requested {requested_id!r}; "
                "mutation retry idempotency requires authoritative custom "
                "episode IDs"
            )
        return episode

    def _mutation_episode_id(
        self,
        *,
        mutation_type: str,
        user_id: str,
        invocation: int,
        key: str,
        index: int,
    ) -> str:
        """Stable episode ID for mutation writes.

        Hashes ``contract_id`` + ``user_id`` + ``invocation`` +
        ``mutation_type`` + ``key`` + ``index``. The same contract retried
        against the same hosted provider asks for the same IDs (idempotent),
        but two different users in the same contract — or two invocations
        of the same mutation under one user — get distinct IDs.
        Adapters and stores that honour ``episode_id`` (the reference
        SQLite adapter and ``LangGraphStoreAdapter``) treat the second
        attempt as idempotent rather than as a fresh bulk-insert. The
        configurable MCP adapter forwards the requested ID; if the upstream
        tool overrides it, the actual returned id is what lands in
        ``inserted_episode_ids``.
        """
        digest = hashlib.sha256(
            (
                f"{self._run.contract_id}|{user_id}|{invocation}|"
                f"{mutation_type}|{key}|{index}"
            ).encode()
        ).hexdigest()[:16]
        return f"mut-{mutation_type}-{digest}-{index:04d}"

    # -------------------------------------------------------------- DSL: assertions
    def should_recall(
        self,
        query: str,
        *,
        k: int = 5,
        contains: str | list[str] | None = None,
        excludes: str | list[str] | None = None,
        latest_fact_is: str | None = None,
        must_not_answer_as: list[str] | None = None,
        judge_assertion: Rubric | None = None,
    ) -> list[Recalled]:
        """Run a recall and assert against the joined response text.

        **Rule-based modes** (work with the default ``[judge].provider = "none"``):

        - ``contains`` — at least one of the listed values must appear
          (case-insensitive) in the recalled text. Vacuously ``False`` when
          nothing is recalled.
        - ``excludes`` — none of the listed values may appear (case-insensitive).
          Vacuously ``True`` when nothing is recalled.

        **Judge-driven modes** (require ``[judge]`` configured; v0.2.2 step 4):

        - ``latest_fact_is`` — the latest fact must be present and dominant;
          older facts may appear only as historical framing.
        - ``must_not_answer_as`` — the response must not assert any of these
          as the *current* state.
        - ``judge_assertion`` — free-form ``Rubric(criterion=..., ...)`` rubric
          escape hatch.

        **Combination rules (Decision #3a + Decision #9):**

        - At most one judge-mode kwarg per call. Combining two raises
          ``ValueError`` at call time.
        - Rule-based + judge-mode can coexist. Rule-based evaluates first
          with fail-fast; judge runs only if all rule-based assertions
          passed. A failing ``contains=`` + ``latest_fact_is=`` call
          therefore never spends judge cost.

        **Fail-loud default (Decision #3b):** if a judge-mode kwarg is used
        but ``[judge]`` is unconfigured (``provider = "none"``), this raises
        ``JudgeUnavailableError`` UNLESS the contract is decorated with
        ``@pytest.mark.recalllab_optional("judge_configured")``, in which
        case the call short-circuits to ``pytest.skip``.

        v0.2.2 step 2 ships the signature + gate but NOT the judge
        evaluation logic — calling a judge mode against a configured judge
        currently raises ``NotImplementedError``. Step 4 lands the
        ``AnthropicJudge`` adapter.
        """
        # Decision #3a: at most one judge-mode kwarg per call.
        active_judge_modes = [
            name
            for name, value in (
                ("latest_fact_is", latest_fact_is),
                ("must_not_answer_as", must_not_answer_as),
                ("judge_assertion", judge_assertion),
            )
            if value is not None
        ]
        if len(active_judge_modes) > 1:
            raise ValueError(
                "only one judge-mode kwarg per should_recall call; saw: "
                + ", ".join(active_judge_modes)
            )

        if (
            contains is None
            and excludes is None
            and not active_judge_modes
        ):
            raise ValueError(
                "should_recall() needs at least one of contains=, "
                "excludes=, latest_fact_is=, must_not_answer_as=, or "
                "judge_assertion="
            )

        results = self.recall(query, k=k)
        joined = "\n".join(r.text for r in results)

        # Decision #9: rule-based assertions evaluate FIRST with fail-fast.
        # A failing rule-based assertion raises AssertionError here and
        # the judge code below never runs — which means the fail-loud gate
        # also never fires. That's intentional: a contract whose
        # rule-based assertion already failed should report the rule-based
        # failure, not the judge-availability error. The behavior is
        # identical regardless of whether the judge is configured, so
        # CI can reproduce the failure even when [judge] = "none".
        if contains is not None:
            self._assert_contains(joined, contains)
        if excludes is not None:
            self._assert_excludes(joined, excludes)

        # Decision #3b: fail-loud gate runs AFTER rule-based assertions
        # pass. Codex finding (round-3 adversarial on step 2): placing the
        # gate before recall meant `contains="missing", latest_fact_is="X"`
        # against NoOpJudge raised JudgeUnavailableError instead of the
        # AssertionError the user expected — that contradicted Decision #9
        # because cheap assertions should always run first. With the gate
        # here, the order is rule-based → gate → judge eval, which matches
        # the documented short-circuit semantics.
        if active_judge_modes and not self._judge.capabilities().available:
            judge_kwarg = active_judge_modes[0]
            if self._judge_optional:
                pytest.skip(
                    f"judge mode {judge_kwarg!r} used but [judge] is not "
                    f"configured; skipping per "
                    f"@pytest.mark.recalllab_optional('judge_configured')"
                )
            raise JudgeUnavailableError(
                f"judge mode {judge_kwarg!r} used but [judge] is not "
                f"configured. Either configure it in recalllab.toml "
                f"(set [judge].provider = 'anthropic' and install the "
                f"[judge] extra), or mark this contract with "
                f"@pytest.mark.recalllab_optional('judge_configured') if "
                f"the test is genuinely optional in this environment."
            )

        if active_judge_modes:
            # Step 2 plumbing only: judge evaluation logic lands in step 4.
            # The gate above already proves the judge is configured, so
            # raising NotImplementedError here is honest — we have a
            # capable judge but no code yet to drive it.
            raise NotImplementedError(
                f"judge mode {active_judge_modes[0]!r} evaluation lands "
                f"in v0.2.2 step 4 (AnthropicJudge adapter). v0.2.2 step "
                f"2 ships the protocol, the [judge] config wiring, and "
                f"the fail-loud gate only."
            )

        return results

    # ---------------------------------------------------------------- internal API
    def _last_remember_for(self, user_id: str) -> TraceEvent | None:
        """Walk back through trace events to find the most recent REMEMBER for ``user_id``."""
        for event in reversed(self._run.events):
            if (
                event.kind == EventKind.REMEMBER
                and event.payload.get("user_id") == user_id
            ):
                return event
        return None

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
        passed: bool | None,
        mode: str,
        expected: object,
        actual: str,
        reason: str | None = None,
        cost_estimate: dict[str, Any] | None = None,
    ) -> None:
        """Record one assertion's outcome on both the assertions list and the trace.

        ``passed=True`` / ``False`` represent evaluated outcomes. ``passed=None``
        is a placeholder for an assertion that was *not evaluated* — used by
        Decision #9 short-circuit (``docs/judge-assertions.md``): when a
        combined rule + judge call fails on the rule-based side, the judge
        side records a placeholder so ``recalllab record`` can faithfully
        regenerate the original call with both kwargs intact. The run
        status only flips to FAILED when ``passed is False``; placeholders
        do not count as failures.

        ``cost_estimate`` carries the judge-call accounting payload for
        judge-mode ASSERTs (see ``TraceEvent`` docstring); rule-based
        ASSERTs leave it ``None``.
        """
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
                cost_estimate=cost_estimate,
            )
        )
        # Only true failures flip the run status. ``passed=None`` is a
        # short-circuit placeholder (Decision #9) and must not be counted
        # as a failure — ``if not passed:`` would silently catch it.
        if passed is False:
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
