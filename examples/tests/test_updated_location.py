"""Temporal updates — the new value must be retrievable after the user updates a fact.

The reference adapter is intentionally minimal (lexical retrieval only); it
will return both the stale and the updated memories, so ``contains`` is the
right assertion mode here. For the stricter "the old value must not be the
*current* answer", use ``must_not_answer_as`` once a judge is configured in
recalllab.toml::

    memory_contract.should_recall(
        "Where do I live now?",
        must_not_answer_as=["Bangalore"],
    )
"""


def test_updated_location_overrides_stale_memory(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I live in Bangalore.")
    memory_contract.remember("Correction: I moved to Mumbai.")
    memory_contract.should_recall("Where do I live now?", contains="Mumbai")
