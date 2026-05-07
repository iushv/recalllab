"""Contradiction resolution — when a fact is updated, the new value must be retrievable.

Like ``test_updated_location``, this test asserts the new value is *present*
in the recalled context using ``contains``. To assert the older value is
no longer the *current* answer (rather than absent from the recall list),
use ``must_not_answer_as`` once a judge is configured in recalllab.toml.
"""


def test_promotion_overrides_old_title(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("My job title is Junior Engineer.")
    memory_contract.remember(
        "Update on my job title: I was promoted to Senior Engineer."
    )
    memory_contract.should_recall(
        "What is my job title now?", contains="Senior Engineer"
    )
