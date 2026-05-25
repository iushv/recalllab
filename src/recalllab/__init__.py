"""RecallLab — pytest for agent memory.

Top-level re-exports for the user-facing surface. The pytest plugin and
DSL live under ``recalllab.core.*``; adapters under ``recalllab.adapters.*``.
"""

from __future__ import annotations

from recalllab.core.judge.rubric import Rubric

__all__ = ["Rubric"]
