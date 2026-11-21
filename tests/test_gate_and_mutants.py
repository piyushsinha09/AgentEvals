"""Offline tests (free) for the gate rules and the mutant machinery."""

import config
from agent.prompts import SYSTEM_PROMPT
from evals.checks import CRITICAL_PREFIX, run_checks
from evals.mutants import MUTANTS, OVERRIDE_HEADER
from scripts.run_eval import ADVISORY, aggregate, failed_signals, gate, newly_failing
from scripts.run_mutants import assess, build_report

GOOD = {
    "tool_correctness": 1.0,
    "argument_correctness": 1.0,
    "answer_correctness": 1.0,
    "task_completion": 1.0,
    "step_efficiency": 1.0,
    "plan_quality": 1.0,
    "plan_adherence": 1.0,
}


def make_rows(failing: dict[str, list[str]] | None = None, n: int = 10, scores=None):
    failing = failing or {}
    rows = []
    for i in range(n):
        cid = f"case{i}"
        rows.append({"id": cid, "scores": dict(scores or GOOD), "check_failures": failing.get(cid, [])})
    return rows


def summary_of(rows):
    return aggregate(rows, list(GOOD))


# ---- advisory metrics ------------------------------------------------------
def test_plan_metrics_are_advisory():
    assert {"plan_quality", "plan_adherence"} <= ADVISORY


def test_advisory_metric_cannot_fail_the_gate():
    bad_plan = dict(GOOD, plan_quality=0.0, plan_adherence=0.0)
    assert gate(summary_of(make_rows(scores=bad_plan))) == []


def test_gating_metric_below_threshold_fails():
    low = dict(GOOD, tool_correctness=0.5)
    problems = gate(summary_of(make_rows(scores=low)))
    assert any("tool_correctness" in p for p in problems)


# ---- critical rule ---------------------------------------------------------
def test_forbidden_amount_is_critical():
    case = {"expect_ticket": False, "expected_amount": None, "forbidden_amounts": [500.0]}
    tickets = [{"name": "issue_refund_ticket", "args": {"amount": 500.0}, "output": "ok"}]
    failures = run_checks(case, tickets)
    assert any(f.startswith(CRITICAL_PREFIX) for f in failures)


def test_one_critical_failure_fails_even_with_good_averages():
    # 1 failing case out of 10 = 90% pass rate, above the 85% minimum
    rows = make_rows({"case3": [f"{CRITICAL_PREFIX} forbidden refund amount 500.0 was issued"]})
    summary = summary_of(rows)
    assert summary["check_pass_rate"] >= config.MIN_CHECK_PASS_RATE
    assert any("CRITICAL" in p for p in gate(summary))
    assert "checks" in failed_signals(summary)


def test_one_noncritical_failure_alone_does_not_fail_without_baseline():
    summary = summary_of(make_rows({"case3": ["a ticket was issued but none was expected"]}))
    assert gate(summary) == []


# ---- case-level regression -------------------------------------------------
def test_case_regression_vs_baseline_fails():
    baseline = summary_of(make_rows())
    current = summary_of(make_rows({"case3": ["ticket amount 55 != expected 20.0"]}))
    assert newly_failing(current, baseline) == ["case3"]
    assert any("passed in the baseline" in p for p in gate(current, baseline))


def test_case_failing_in_baseline_is_not_a_new_regression():
    baseline = summary_of(make_rows({"case3": ["x"]}))
    current = summary_of(make_rows({"case3": ["x"]}))
    assert newly_failing(current, baseline) == []


def test_old_baseline_without_case_checks_still_works():
    old = {"averages": dict(GOOD), "check_pass_rate": 1.0, "cases": 10}
    assert gate(summary_of(make_rows()), old) == []


# ---- mutants ---------------------------------------------------------------
def test_mutant_names_unique_and_prompts_extend_the_base_prompt():
    names = [m.name for m in MUTANTS]
    assert len(names) == len(set(names))
    for m in MUTANTS:
        assert m.prompt().startswith(SYSTEM_PROMPT)
        assert OVERRIDE_HEADER in m.prompt() and m.override in m.prompt()


def test_mutant_expected_signals_are_valid():
    valid = (set(config.THRESHOLDS) - ADVISORY) | {"checks"}
    for m in MUTANTS:
        assert m.expect and set(m.expect) <= valid, m.name


def test_assess_detects_a_broken_agent_and_ignores_a_clean_one():
    control = summary_of(make_rows())
    broken = summary_of(make_rows({"case1": ["a"], "case2": ["b"], "case3": ["c"]}))
    mutant = next(m for m in MUTANTS if m.name == "never_issues_ticket")

    caught = assess(mutant, broken, control)
    assert caught["detected"] and caught["expected_signal_hit"]

    clean = assess(mutant, summary_of(make_rows()), control)
    assert not clean["detected"]


def test_report_shows_detection_rate_and_control_status():
    control = summary_of(make_rows())
    broken = summary_of(make_rows({"case1": ["a"]}))
    mutant = MUTANTS[0]
    results = [assess(mutant, broken, control)]
    report = build_report("full", control, [], results)
    assert "Detection rate: 1/1" in report
    assert "Control (normal agent): clean" in report
    assert "NOT clean" in build_report("full", control, ["something"], results)
