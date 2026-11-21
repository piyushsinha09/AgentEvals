"""Offline tests: no API key, no cost. Run them on every push."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.agent import extract_result
from agent.data import PURCHASES, REFERENCE_DATE
from agent.tools import (
    TOOL_NAMES,
    calculate_refund,
    get_today_date,
    get_user_purchase_history,
    issue_refund_ticket,
)
from evals.checks import run_checks
from evals.runner import NO_ARG_TOOLS, argument_test_case, load_cases, to_test_case


# ---- tools -----------------------------------------------------------------
def test_purchase_history_known_and_unknown_user():
    assert "headphones" in get_user_purchase_history.invoke({"username": "user101"})
    assert "No account found" in get_user_purchase_history.invoke({"username": "nobody"})


def test_today_is_fixed_reference_date():
    assert get_today_date.invoke({}) == REFERENCE_DATE.isoformat()


def test_calculate_refund_sums_and_dedupes():
    out = calculate_refund.invoke({"username": "user103", "items": ["paintbrush", "canvas", "canvas"]})
    assert "$34.00" in out


def test_calculate_refund_unknown_item():
    out = calculate_refund.invoke({"username": "user101", "items": ["keyboard"]})
    assert out.startswith("Error")


def test_ticket_rejects_wrong_amount():
    out = issue_refund_ticket.invoke({"username": "user107", "items": ["sneakers"], "amount": 500})
    assert out.startswith("Error")


# ---- dataset consistency ---------------------------------------------------
def test_case_ids_are_unique():
    ids = [c["id"] for c in load_cases()]
    assert len(ids) == len(set(ids))


def test_expected_tools_exist():
    for c in load_cases():
        assert set(c["expected_tools"]) <= set(TOOL_NAMES), c["id"]


def test_expected_ticket_fields_are_consistent():
    for c in load_cases():
        assert (c["expected_amount"] is not None) == c["expect_ticket"], c["id"]


def test_expected_amounts_match_data():
    """Expected refund amounts in the dataset must be real sums of item prices."""
    all_sums = set()
    for items in PURCHASES.values():
        prices = [p["price"] for p in items]
        for mask in range(1, 2 ** len(prices)):
            all_sums.add(sum(p for i, p in enumerate(prices) if mask >> i & 1))
    for c in load_cases():
        if c["expected_amount"] is not None:
            assert c["expected_amount"] in all_sums, c["id"]


# ---- deterministic checks --------------------------------------------------
def _ticket(amount):
    return {"name": "issue_refund_ticket", "args": {"amount": amount}, "output": "ok"}


def test_checks_pass_and_fail():
    case = {"expect_ticket": True, "expected_amount": 35.0}
    assert run_checks(case, [_ticket(35.0)]) == []
    assert run_checks(case, [_ticket(20.0)])
    assert run_checks(case, [])
    no_ticket = {"expect_ticket": False, "expected_amount": None}
    assert run_checks(no_ticket, []) == []
    assert run_checks(no_ticket, [_ticket(35.0)])


def test_checks_forbidden_amount():
    case = {"expect_ticket": False, "expected_amount": None, "forbidden_amounts": [500.0]}
    assert any("forbidden" in f for f in run_checks(case, [_ticket(500.0)]))


# ---- message parsing -------------------------------------------------------
def test_extract_result_gets_answer_and_tool_calls():
    messages = [
        HumanMessage(content="refund please"),
        AIMessage(content="", tool_calls=[{"name": "get_today_date", "args": {}, "id": "c1"}]),
        ToolMessage(content="2026-11-01", tool_call_id="c1"),
        AIMessage(content=[{"type": "text", "text": "Done."}]),
    ]
    result = extract_result(messages)
    assert result.answer == "Done."
    assert result.tool_calls == [{"name": "get_today_date", "args": {}, "output": "2026-11-01"}]


def test_argument_test_case_drops_no_arg_tools():
    assert NO_ARG_TOOLS == {"get_today_date"}
    case = load_cases()[0]
    messages = [
        HumanMessage(content="x"),
        AIMessage(content="", tool_calls=[
            {"name": "get_today_date", "args": {}, "id": "a"},
            {"name": "get_user_purchase_history", "args": {"username": "user101"}, "id": "b"},
        ]),
        ToolMessage(content="d", tool_call_id="a"),
        ToolMessage(content="h", tool_call_id="b"),
        AIMessage(content="ok"),
    ]
    tc = to_test_case(case, extract_result(messages))
    filtered = argument_test_case(tc)
    assert [c.name for c in filtered.tools_called] == ["get_user_purchase_history"]
    assert len(tc.tools_called) == 2  # original untouched
