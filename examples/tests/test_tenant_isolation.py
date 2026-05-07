"""Tenant isolation — User B must not see User A's memories.

The reference adapter scopes every operation by ``user_id``; recalls for
``bob`` never see episodes seeded for ``alice``. Providers that share a
namespace across users (i.e. don't support tenant isolation) will fail
this contract — that's the point.
"""


def test_user_b_cannot_see_user_a_memories(memory_contract):
    memory_contract.given_user("alice").remember("Project codename: Aurora.")
    # Precondition: alice can recall her own data, otherwise the cross-tenant
    # excludes assertion below would be vacuously true.
    memory_contract.should_recall(
        "What is the project codename?", contains="Aurora"
    )

    memory_contract.given_user("bob")
    memory_contract.should_recall(
        "What is the project codename?", excludes="Aurora"
    )
