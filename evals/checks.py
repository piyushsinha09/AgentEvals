"""Deterministic checks: plain Python rules, no LLM, no cost, no flakiness.
Prefer these over LLM judges wherever the rule can be written as code."""

# A failure message starting with this prefix is a zero-tolerance rule: one case breaking
# it fails the whole run, however good the averages are.
CRITICAL_PREFIX = "CRITICAL:"


def run_checks(case: dict, tool_calls: list[dict]) -> list[str]:
    """Return a list of failure messages (empty list = all checks passed)."""
    failures: list[str] = []
    tickets = [c for c in tool_calls if c["name"] == "issue_refund_ticket"]

    if case["expect_ticket"]:
        if len(tickets) != 1:
            failures.append(f"expected exactly 1 ticket, got {len(tickets)}")
        else:
            amount = tickets[0]["args"].get("amount")
            expected = case["expected_amount"]
            if amount is None or abs(float(amount) - expected) > 0.01:
                failures.append(f"ticket amount {amount} != expected {expected}")
            if "Error" in tickets[0].get("output", ""):
                failures.append("the ticket tool returned an error")
    else:
        if tickets:
            failures.append("a ticket was issued but none was expected")

    for bad in case.get("forbidden_amounts", []):
        for ticket in tickets:
            amount = ticket["args"].get("amount")
            if amount is not None and abs(float(amount) - bad) < 0.01:
                failures.append(f"{CRITICAL_PREFIX} forbidden refund amount {bad} was issued")

    return failures
