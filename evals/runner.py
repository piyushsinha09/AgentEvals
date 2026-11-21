"""Loads the dataset and turns one agent run into a DeepEval test case."""

import json
from pathlib import Path

from deepeval.test_case import LLMTestCase, ToolCall

from agent.agent import AgentResult, run_agent
from agent.tools import TOOLS

DATASET_PATH = Path(__file__).parent / "dataset.json"

# Tools that legitimately take no arguments (e.g. get_today_date).
NO_ARG_TOOLS = {t.name for t in TOOLS if not t.args}


def load_cases(smoke_only: bool = False) -> list[dict]:
    cases = json.loads(DATASET_PATH.read_text())
    return [c for c in cases if c.get("smoke")] if smoke_only else cases


def to_test_case(case: dict, result: AgentResult) -> LLMTestCase:
    return LLMTestCase(
        name=case["id"],
        input=case["input"],
        actual_output=result.answer,
        expected_output=case["expected_output"],
        tools_called=[
            ToolCall(
                name=c["name"], input_parameters=c["args"], output=c.get("output", "")
            )
            for c in result.tool_calls
        ],
        expected_tools=[ToolCall(name=n) for n in case["expected_tools"]],
    )


def run_case(
    agent, case: dict, callbacks: list | None = None
) -> tuple[AgentResult, LLMTestCase]:
    result = run_agent(
        agent, case["input"], thread_id=f"eval-{case['id']}", callbacks=callbacks
    )
    return result, to_test_case(case, result)


def argument_test_case(test_case: LLMTestCase) -> LLMTestCase:
    """Copy of the test case without no-parameter tools. The argument-correctness
    judge wrongly penalizes calls that correctly have no arguments."""
    kept = [c for c in test_case.tools_called if c.name not in NO_ARG_TOOLS]
    return test_case.model_copy(update={"tools_called": kept})
