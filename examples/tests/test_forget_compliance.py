"""Forget compliance — after ``forget(...)``, the data must not appear in recalls.

Tests the immediate-removal property: a ``forget`` call returns synchronously
and subsequent recalls must not surface the deleted memory. The reference
adapter physically removes the row, so this passes trivially; providers
with eventual-consistency or tombstoning behaviour will reveal the gap.
"""


def test_forget_removes_allergy_immediately(memory_contract):
    memory_contract.given_user("ayush")
    memory_contract.remember("I am allergic to peanuts.")

    # Precondition: prove the fact is retrievable BEFORE we forget it,
    # otherwise the post-forget excludes assertion is vacuously true.
    memory_contract.should_recall("What am I allergic to?", contains="peanut")

    deleted = memory_contract.forget(matching="peanuts")
    assert deleted == 1, f"expected exactly 1 memory removed, got {deleted}"

    memory_contract.should_recall("What am I allergic to?", excludes="peanut")
