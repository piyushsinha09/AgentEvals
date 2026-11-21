"""Sample purchase data.

Dates are computed relative to a FIXED reference date, so test expectations
never drift as real time passes (a flaw in many tutorial datasets).
"""

from datetime import date, timedelta

REFERENCE_DATE = date(2026, 11, 1)
REFUND_WINDOW_DAYS = 30


def _days_ago(n: int) -> str:
    return (REFERENCE_DATE - timedelta(days=n)).isoformat()


PURCHASES: dict[str, list[dict]] = {
    "user101": [
        {"item": "headphones", "price": 35.0, "purchase_date": _days_ago(10)},
        {"item": "mouse", "price": 20.0, "purchase_date": _days_ago(12)},
    ],
    "user102": [
        {"item": "book", "price": 18.0, "purchase_date": _days_ago(45)},
    ],
    "user103": [
        {"item": "paintbrush", "price": 14.0, "purchase_date": _days_ago(5)},
        {"item": "canvas", "price": 20.0, "purchase_date": _days_ago(5)},
    ],
    "user104": [
        {"item": "usb drive", "price": 16.0, "purchase_date": _days_ago(3)},
    ],
    "user105": [
        # Exactly 30 days old: still eligible ("no more than 30 days").
        {"item": "keyboard", "price": 60.0, "purchase_date": _days_ago(30)},
    ],
    "user106": [
        {"item": "monitor", "price": 150.0, "purchase_date": _days_ago(31)},
        {"item": "lamp", "price": 25.0, "purchase_date": _days_ago(2)},
    ],
    "user107": [
        {"item": "sneakers", "price": 80.0, "purchase_date": _days_ago(1)},
    ],
}
