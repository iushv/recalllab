"""Contract mutations.

v0.2.0 ships two deterministic, seeded mutations that operate against the
active user's namespace through the existing ``MemoryProvider`` API:

- :func:`sample_distractors` — picks ``n`` plausible-but-unrelated facts from
  a fixed pool (or pads with replacement if ``n`` exceeds the pool size),
  using ``random.Random(seed)`` so the sample is reproducible.

The DSL methods on ``MemoryContract`` (``with_distractors`` and
``with_stale_repeats``) live in ``recalllab.core.contract.dsl`` and call into
this module for the deterministic-sampling piece.
"""

from recalllab.core.mutations.distractors import (
    DISTRACTOR_POOL,
    sample_distractors,
    validate_seed,
)

__all__ = ["DISTRACTOR_POOL", "sample_distractors", "validate_seed"]
