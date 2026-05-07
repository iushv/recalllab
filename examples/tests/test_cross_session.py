"""Cross-session recall — a fact stated early must remain recallable later.

Simulates a long conversation by interleaving a single load-bearing fact
with many unrelated turns. The memory provider must surface the fact when
asked, despite the surrounding chatter.
"""


def test_birthday_persists_across_many_turns(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("My birthday is December 28.")
    for topic in ("weather", "sports", "movies", "books", "food", "travel"):
        memory_contract.remember(f"We talked about {topic}.")
    memory_contract.should_recall("When is my birthday?", contains="December 28")
