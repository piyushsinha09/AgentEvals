"""Test the evaluator itself: run it on a normal agent (control) and on deliberately broken
agents (mutants), and check that it flags every broken one without flagging the control.

Usage:
    uv run python -m scripts.run_mutants                 # full dataset (slow, costs API calls)
    uv run python -m scripts.run_mutants --smoke         # 7 cases per variant (cheaper)
    uv run python -m scripts.run_mutants --only skips_date_check,obeys_injection

Output: reports/mutants.md and reports/mutants.json. Exit code 0 only if the control is clean
and every mutant is flagged. Output metrics only (no trace metrics), to keep cost down.
"""

import argparse
import json
import sys

import config
from agent.agent import build_agent
from agent.prompts import SYSTEM_PROMPT
from evals.judge import build_judge
from evals.metrics import build_metrics
from evals.mutants import MUTANTS, Mutant
from evals.runner import load_cases
from evals.trace_eval import run_plain
from scripts.run_eval import REPORT_DIR, aggregate, evaluate_case, failed_signals, gate, newly_failing


def run_variant(system_prompt: str, cases: list[dict], metrics: dict) -> tuple[list[dict], dict]:
    agent = build_agent(system_prompt=system_prompt)
    runs = run_plain(agent, cases)
    rows = [evaluate_case(metrics, [], c, runs[c["id"]]) for c in cases]
    return rows, aggregate(rows, list(metrics))


def assess(mutant: Mutant, summary: dict, control: dict) -> dict:
    """Did the evaluator flag this mutant, and by the signal we expected?"""
    signals = failed_signals(summary, control)
    return {
        "name": mutant.name,
        "description": mutant.description,
        "expected": list(mutant.expect),
        "signals": sorted(signals),
        "detected": bool(signals),
        "expected_signal_hit": bool(signals & set(mutant.expect)),
        "newly_failing": newly_failing(summary, control),
        "problems": gate(summary, control),
        "averages": summary["averages"],
        "check_pass_rate": summary["check_pass_rate"],
    }


def build_report(mode: str, control: dict, control_problems: list[str], results: list[dict]) -> str:
    detected = sum(r["detected"] for r in results)
    hit = sum(r["expected_signal_hit"] for r in results)
    lines = [
        "## Evaluator test: can it catch broken agents?",
        f"Mode: **{mode}** | cases per variant: **{control['cases']}** | "
        f"agent: `{config.AGENT_MODEL}` | judge: `{config.JUDGE_MODEL}`",
        "",
        f"**Detection rate: {detected}/{len(results)}** mutants flagged "
        f"({hit}/{len(results)} by an expected signal).",
        "",
        "| Broken agent | What was broken | Caught? | Flagged by | Expected | Newly failing cases |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['description']} | {'YES' if r['detected'] else '**NO**'} | "
            f"{', '.join(r['signals']) or '-'} | {', '.join(r['expected'])} | {len(r['newly_failing'])} |"
        )
    lines += [""]
    if control_problems:
        lines.append("**Control (normal agent): NOT clean.** The evaluator flagged the unmodified agent:")
        lines += [f"- {p}" for p in control_problems]
        lines.append("Fix this first: mutant results are only meaningful when the control is clean.")
    else:
        lines.append("**Control (normal agent): clean.** No false alarm on the unmodified agent.")
    missed = [r["name"] for r in results if not r["detected"]]
    if missed:
        lines += [
            "",
            "Not caught: " + ", ".join(missed) + ". Either the evaluator has a blind spot here, "
            "or the mutant did not change the agent's behavior. Read reports/mutants.json to tell which.",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="use only smoke cases")
    parser.add_argument("--only", default="", help="comma-separated mutant names")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    mutants = [m for m in MUTANTS if not wanted or m.name in wanted]
    unknown = wanted - {m.name for m in MUTANTS}
    if unknown:
        print(f"Unknown mutant(s): {', '.join(sorted(unknown))}. Known: {', '.join(m.name for m in MUTANTS)}")
        return 2

    cases = load_cases(smoke_only=args.smoke)
    metrics = build_metrics(build_judge())

    print("== control (unmodified agent)", flush=True)
    _, control = run_variant(SYSTEM_PROMPT, cases, metrics)
    control_problems = gate(control)

    results, details = [], {}
    for m in mutants:
        print(f"\n== mutant: {m.name}", flush=True)
        rows, summary = run_variant(m.prompt(), cases, metrics)
        results.append(assess(m, summary, control))
        details[m.name] = [
            {"id": r["id"], "check_failures": r["check_failures"], "tools": r.get("tools", [])}
            for r in rows
        ]

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "mutants.json").write_text(
        json.dumps({"mode": mode, "control": control, "results": results, "cases": details}, indent=2)
    )
    report = build_report(mode, control, control_problems, results)
    (REPORT_DIR / "mutants.md").write_text(report)
    print("\n" + report)

    all_caught = all(r["detected"] for r in results)
    return 0 if (all_caught and not control_problems) else 1


if __name__ == "__main__":
    sys.exit(main())
