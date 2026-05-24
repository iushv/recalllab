"""User-facing ``Rubric`` model for the ``judge_assertion=`` mode.

``Rubric`` carries the free-form criterion text plus the user's
preferred pass/fail labels. The judge prompt sees only ``criterion``;
labels are local to the runtime and used to render verdict reasons in
the trace and pytest failure messages.

Rubric identity (locked in ``docs/judge-assertions.md`` §Rubric class):

- **Prompt identity** = ``criterion`` only. Same criterion → same
  prompt envelope → same judge verdict (when caching lands in v0.2.3,
  this is what it keys off — together with the rest of the identity
  tuple in ``recalllab.core.judge.base.JudgeRequest``).
- **Trace identity** = full ``model_dump()``. The ASSERT row's
  ``expected`` field stores criterion + both labels so the trace is
  human-readable with the user's own vocabulary, and the
  trace-to-test emitter regenerates the literal with explicit
  kwargs.

Label-only changes do NOT regenerate the verdict.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Rubric"]


class Rubric(BaseModel):
    """User-supplied free-form rubric for ``judge_assertion=``.

    Pass an instance to ``should_recall(query, judge_assertion=...)``::

        memory_contract.should_recall(
            "Where do I live?",
            judge_assertion=Rubric(
                criterion="The response must cite the source episode.",
                pass_label="CITED",
                fail_label="UNCITED",
            ),
        )

    Only ``criterion`` enters the judge prompt envelope; ``pass_label``
    and ``fail_label`` are reflected back to the user in
    ``AssertionResult.reason`` so failure messages match the rubric's
    own vocabulary.
    """

    model_config = ConfigDict(frozen=True)

    criterion: str = Field(min_length=1)
    pass_label: str = "PASS"
    fail_label: str = "FAIL"
