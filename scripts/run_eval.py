"""Run the evaluation suite, compare with the baseline, write reports, set exit code.

Usage:
    uv run python -m scripts.run_eval                    # full dataset, all 7 metrics
    uv run python -m scripts.run_eval --smoke            # small subset (cheap, for PRs)
    uv run python -m scripts.run_eval --no-trace         # skip trace metrics (3 metrics only)
    uv run python -m scripts.run_eval --save-baseline    # store this run as the baseline

Gate rules (any one failing fails the run):
    1. a gating metric's average is below its threshold
    2. a gating metric dropped more than MAX_REGRESSION versus the baseline
    3. the share of cases passing all deterministic checks is below MIN_CHECK_PASS_RATE
    4. any CRITICAL deterministic rule was broken, in even one case
    5. a case that passed in the baseline fails now (case-level regression)
Advisory metrics (config.ADVISORY_METRICS) are reported but never fail the run.
"""

import argparse
import json
import sys
from pathlib import Path

import config
from agent.agent import build_agent
from evals.checks import CRITICAL_PREFIX, run_checks
from evals.judge import build_judge
from evals.metrics import build_metrics
from evals.runner import argument_test_case, load_cases
from evals.trace_eval import (
    TRACE_METRIC_KEYS,
    CaseRun,
    build_trace_metrics,
    run_plain,
    run_traced,
)

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "evals" / "baselines"
REPORT_DIR = ROOT / "reports"
ADVISORY = set(getattr(config, "ADVISORY_METRICS", ()))


def evaluate_case(metrics: dict, trace_keys: list[str], case: dict, run: CaseRun) -> dict:
    """Score one finished agent run: output metrics + deterministic checks + trace scores."""
    all_names = list(metrics) + trace_keys
    row = {"id": case["id"], "scores": {}, "reasons": {}, "check_failures": []}

    if run.result is None:
        row["error"] = run.error or "agent did not run"
        row["scores"] = {name: 0.0 for name in all_names}
        row["check_failures"] = ["agent run failed"]
        return row

    row["answer"] = run.result.answer
    row["tools"] = [c["name"] for c in run.result.tool_calls]
    row["check_failures"] = run_checks(case, run.result.tool_calls)

    # Output metrics (the argument metric ignores no-parameter tools like get_today_date)
    for name, metric in metrics.items():
        tc = argument_test_case(run.test_case) if name == "argument_correctness" else run.test_case
        try:
            metric.measure(tc, _show_indicator=False)
            row["scores"][name] = float(metric.score)
            row["reasons"][name] = getattr(metric, "reason", "") or ""
        except Exception as e:
            row["scores"][name] = 0.0
            row["reasons"][name] = f"ERROR: {e}"

    # Trace metrics (scored by DeepEval while the run was being captured)
    for key in trace_keys:
        info = run.trace_scores.get(key)
        row["scores"][key] = info["score"] if info else 0.0
        row["reasons"][key] = info["reason"] if info else "ERROR: no trace score was produced"
    return row


def aggregate(rows: list[dict], metric_names: list[str]) -> dict:
    n = len(rows)
    averages = {m: sum(r["scores"][m] for r in rows) / n for m in metric_names}
    return {
        "averages": averages,
        "check_pass_rate": sum(1 for r in rows if not r["check_failures"]) / n,
        "cases": n,
        # per-case results, used for case-level regression and critical-rule gating
        "case_checks": {r["id"]: not r["check_failures"] for r in rows},
        "critical_cases": [
            r["id"]
            for r in rows
            if any(f.startswith(CRITICAL_PREFIX) for f in r["check_failures"])
        ],
    }


def load_baseline(mode: str) -> dict | None:
    path = BASELINE_DIR / f"baseline_{mode}.json"
    return json.loads(path.read_text()) if path.exists() else None


def newly_failing(summary: dict, baseline: dict | None) -> list[str]:
    """Cases whose deterministic checks passed in the baseline but fail now."""
    if not baseline:
        return []
    base_cases = baseline.get("case_checks", {})
    return [cid for cid, ok in summary.get("case_checks", {}).items() if base_cases.get(cid) and not ok]


def metric_problems(summary: dict, baseline: dict | None) -> dict[str, str]:
    """Gating metrics that are below threshold or regressed vs the baseline."""
    found = {}
    for name, avg in summary["averages"].items():
        if name in ADVISORY:
            continue
        if avg < config.THRESHOLDS[name]:
            found[name] = f"{name} average {avg:.2f} < threshold {config.THRESHOLDS[name]:.2f}"
        elif baseline and name in baseline["averages"]:
            drop = baseline["averages"][name] - avg
            if drop > config.MAX_REGRESSION:
                found[name] = (
                    f"{name} dropped {drop:.2f} vs baseline "
                    f"({baseline['averages'][name]:.2f} -> {avg:.2f})"
                )
    return found


def gate(summary: dict, baseline: dict | None = None, stage_error: str | None = None) -> list[str]:
    problems = list(metric_problems(summary, baseline).values())
    if summary["check_pass_rate"] < config.MIN_CHECK_PASS_RATE:
        problems.append(
            f"deterministic check pass rate {summary['check_pass_rate']:.0%} "
            f"< {config.MIN_CHECK_PASS_RATE:.0%}"
        )
    if summary.get("critical_cases"):
        problems.append("CRITICAL rule broken in: " + ", ".join(summary["critical_cases"]))
    regressed = newly_failing(summary, baseline)
    if regressed:
        problems.append(f"{len(regressed)} case(s) passed in the baseline but fail now: " + ", ".join(regressed))
    if stage_error:
        problems.append(stage_error)
    return problems


def failed_signals(summary: dict, baseline: dict | None = None) -> set[str]:
    """Which parts of the evaluator flagged this run: metric names and/or 'checks'."""
    signals = set(metric_problems(summary, baseline))
    if (
        summary["check_pass_rate"] < config.MIN_CHECK_PASS_RATE
        or summary.get("critical_cases")
        or newly_failing(summary, baseline)
    ):
        signals.add("checks")
    return signals


def markdown_report(mode, summary, baseline, rows, problems, trace: bool) -> str:
    lines = [
        "## Agent evaluation report",
        f"Mode: **{mode}** | trace metrics: **{'on' if trace else 'off'}** | "
        f"cases: **{summary['cases']}** | agent: `{config.AGENT_MODEL}` | "
        f"judge: `{config.JUDGE_MODEL}`",
        "",
        "| Metric | Baseline | Current | Change | Threshold |",
        "|---|---|---|---|---|",
    ]
    for name, avg in summary["averages"].items():
        base = baseline["averages"].get(name) if baseline else None
        base_s = f"{base:.2f}" if base is not None else "-"
        delta = f"{avg - base:+.2f}" if base is not None else "-"
        thr = f"{config.THRESHOLDS[name]:.2f}" + (" (advisory)" if name in ADVISORY else "")
        lines.append(f"| {name} | {base_s} | {avg:.2f} | {delta} | {thr} |")
    base_c = baseline.get("check_pass_rate") if baseline else None
    base_c_s = f"{base_c:.0%}" if base_c is not None else "-"
    delta_c = f"{(summary['check_pass_rate'] - base_c) * 100:+.0f} pts" if base_c is not None else "-"
    lines.append(
        f"| deterministic checks | {base_c_s} | {summary['check_pass_rate']:.0%} | "
        f"{delta_c} | {config.MIN_CHECK_PASS_RATE:.0%} |"
    )

    failing = [r for r in rows if r["check_failures"] or r.get("error")]
    if failing:
        lines += ["", "### Failing cases"]
        for r in failing:
            detail = r.get("error") or "; ".join(r["check_failures"])
            lines.append(f"- `{r['id']}`: {detail}")

    low = []
    for r in rows:
        for name, score in r["scores"].items():
            if name in ADVISORY:
                continue
            if score < config.THRESHOLDS[name]:
                reason = " ".join(r["reasons"].get(name, "").split())[:160]
                low.append(f"- `{r['id']}` {name} {score:.2f}: {reason}")
    if low:
        lines += ["", "### Low-scoring metrics (below threshold, gating metrics only)"] + low[:12]
        if len(low) > 12:
            lines.append(f"- ... and {len(low) - 12} more (see reports/latest.json)")

    lines += ["", "**Result: " + ("FAILED**" if problems else "PASSED**")]
    lines += [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run only smoke cases")
    parser.add_argument("--no-trace", action="store_true", help="skip trace metrics")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    use_trace = not args.no_trace

    cases = load_cases(smoke_only=args.smoke)
    judge = build_judge()
    agent = build_agent()
    output_metrics = build_metrics(judge)

    if use_trace:
        runs, stage_error = run_traced(agent, cases, build_trace_metrics(judge))
        trace_keys = list(TRACE_METRIC_KEYS.values())
    else:
        runs, stage_error, trace_keys = run_plain(agent, cases), None, []

    print("\nScoring output metrics...", flush=True)
    rows = [evaluate_case(output_metrics, trace_keys, c, runs[c["id"]]) for c in cases]

    summary = aggregate(rows, list(output_metrics) + trace_keys)
    baseline = load_baseline(mode)
    problems = gate(summary, baseline, stage_error)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(
        json.dumps({"mode": mode, "trace": use_trace, "summary": summary, "rows": rows}, indent=2)
    )
    report = markdown_report(mode, summary, baseline, rows, problems, use_trace)
    (REPORT_DIR / "latest.md").write_text(report)
    print("\n" + report)

    if args.save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        (BASELINE_DIR / f"baseline_{mode}.json").write_text(json.dumps(summary, indent=2))
        print(f"Baseline saved to evals/baselines/baseline_{mode}.json")
        return 0  # saving a new baseline never fails the run
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
