"""Distractor pool + seeded sampler for ``with_distractors``.

The pool is intentionally bland, plausible-looking everyday facts. We avoid
domain-specific or topical content so distractors don't accidentally satisfy
an unrelated contract's assertion (e.g. a contract recalling a city name
shouldn't get false-positive matches from the distractor pool).
"""

from __future__ import annotations

import random
from typing import Any

DISTRACTOR_POOL: tuple[str, ...] = (
    "I had black coffee this morning.",
    "The weather is overcast today.",
    "I watched a documentary about volcanoes last night.",
    "My favourite background colour for terminals is dark grey.",
    "I read a chapter of a novel before bed.",
    "I went for a thirty minute walk after lunch.",
    "I bookmarked an article about typography.",
    "I cleaned the kitchen and refilled the kettle.",
    "I tried a new playlist on the train.",
    "I forgot my umbrella at the office on Tuesday.",
    "I rearranged the icons on my second monitor.",
    "The neighbour's cat sat on my windowsill again.",
    "I wrote a postcard to a friend overseas.",
    "I rotated my running shoes for the season.",
    "I downloaded an album recommended by a podcast guest.",
    "I switched my desk lamp to a warmer bulb.",
    "I ordered a replacement charging cable.",
    "I tested a new espresso bean from the local roaster.",
    "I planted basil in a pot on the balcony.",
    "I fixed the squeaky hinge on the cupboard door.",
    "I reinstalled the keyboard layout on my laptop.",
    "I scheduled a haircut for next Saturday afternoon.",
    "I paid the electricity bill before the deadline.",
    "I refreshed the sticky notes on my cork board.",
    "I scanned an old receipt for the warranty file.",
    "I tried a new recipe with roasted vegetables.",
    "I changed the batteries in the wall clock.",
    "I cleared the unread items in my reading queue.",
    "I labelled three boxes for the storage cupboard.",
    "I updated the contact card for my dentist.",
)


def validate_seed(seed: Any) -> int:
    """Validate and return the canonical integer seed for deterministic sampling."""
    if type(seed) is not int:
        raise TypeError("seed must be an int; pass an explicit integer seed")
    return seed


def sample_distractors(n: int, *, seed: int = 0) -> list[str]:
    """Return ``n`` distractor strings drawn deterministically from the pool.

    Sampling is without replacement when ``n <= len(DISTRACTOR_POOL)``;
    larger ``n`` pads with random choices from the same pool. The same
    ``(n, seed)`` pair always yields the same list.

    ``seed`` defaults to ``0`` rather than ``None`` so the function is
    reproducible by default. ``random.Random(None)`` would seed from system
    entropy, which silently breaks the determinism contract advertised in
    the docstring and recorded in the trace. Callers who want entropy
    should pass an explicit non-zero seed.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    seed = validate_seed(seed)
    rng = random.Random(seed)
    pool = list(DISTRACTOR_POOL)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    extras = [rng.choice(DISTRACTOR_POOL) for _ in range(n - len(pool))]
    return pool + extras
