"""Tools the refund agent can call. They return strings (errors included) instead
of raising, so the agent can recover and the evaluation can see what happened."""

from langchain_core.tools import tool

from agent.data import PURCHASES, REFERENCE_DATE


def _find_purchases(username: str) -> list[dict] | None:
    return PURCHASES.get(username.strip().lower())


@tool
def get_user_purchase_history(username: str) -> str:
    """Look up a customer's purchases by username. Returns one line per purchase
    with item, price and purchase date (YYYY-MM-DD)."""
    purchases = _find_purchases(username)
    if purchases is None:
        return f"No account found for username '{username}'."
    return "\n".join(
        f"{p['item']} | ${p['price']:.2f} | purchased {p['purchase_date']}"
        for p in purchases
    )


@tool
def get_today_date() -> str:
    """Return today's date as YYYY-MM-DD. Use it to check refund eligibility."""
    return REFERENCE_DATE.isoformat()


@tool
def calculate_refund(username: str, items: list[str]) -> str:
    """Add up the prices of the requested items for a customer.
    Returns the total in dollars, or an error if an item was not purchased."""
    purchases = _find_purchases(username)
    if purchases is None:
        return f"Error: no account found for username '{username}'."
    prices = {p["item"].lower(): p["price"] for p in purchases}
    total = 0.0
    for item in dict.fromkeys(i.strip().lower() for i in items):  # de-duplicate
        if item not in prices:
            return f"Error: '{item}' was not found in the purchase history."
        total += prices[item]
    return f"Total refund: ${total:.2f}"


@tool
def issue_refund_ticket(username: str, items: list[str], amount: float) -> str:
    """Issue a refund ticket for the given items and dollar amount.
    This is simulated: it only returns a confirmation string."""
    purchases = _find_purchases(username)
    if purchases is None:
        return f"Error: no account found for username '{username}'."
    prices = {p["item"].lower(): p["price"] for p in purchases}
    expected = 0.0
    for item in dict.fromkeys(i.strip().lower() for i in items):
        if item not in prices:
            return f"Error: '{item}' was not found in the purchase history."
        expected += prices[item]
    if abs(expected - amount) > 0.01:
        return f"Error: amount ${amount:.2f} does not match the items (${expected:.2f})."
    return f"Refund ticket TCK-{username.strip().lower()}-{int(amount)} issued for ${amount:.2f}."


TOOLS = [
    get_user_purchase_history,
    get_today_date,
    calculate_refund,
    issue_refund_ticket,
]
TOOL_NAMES = [t.name for t in TOOLS]
