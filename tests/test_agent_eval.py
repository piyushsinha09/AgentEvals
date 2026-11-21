"""Per-case LLM evaluation with DeepEval + pytest. Costs money, so it is skipped by default.

Run:  uv run pytest -m llm -v
"""

import pytest
from deepeval import assert_test

import config
from agent.agent import build_agent
from evals.checks import run_checks
from evals.judge import build_judge
from evals.metrics import build_metrics
from evals.runner import argument_test_case, load_cases, run_case

pytestmark = pytest.mark.llm

CASES = load_cases()


@pytest.fixture(scope="module")
def agent():
    return build_agent()


@pytest.fixture(scope="module")
def metrics():
    # Looser per-case threshold; the strict average gate lives in scripts/run_eval.py
    per_case = {k: config.CASE_THRESHOLD for k in config.THRESHOLDS}
    return build_metrics(build_judge(), threshold=per_case)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_agent_case(agent, metrics, case):
    result, test_case = run_case(agent, case)

    failures = run_checks(case, result.tool_calls)
    assert not failures, f"deterministic checks failed: {failures}"

    others = [m for k, m in metrics.items() if k != "argument_correctness"]
    assert_test(test_case, others)
    assert_test(argument_test_case(test_case), [metrics["argument_correctness"]])
