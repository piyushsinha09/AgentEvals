"""DeepEval metrics. Keys are the names used in config.THRESHOLDS and in reports."""

from deepeval.metrics import ArgumentCorrectnessMetric, GEval, ToolCorrectnessMetric
from deepeval.test_case import SingleTurnParams

import config

ANSWER_CRITERIA = (
    "Compare the actual output with the expected output for the customer's request. "
    "The actual output must reach the same decision (refund approved, denied, partial, "
    "question answered, or information requested) and mention the same refund amount "
    "when the expected output gives one. Penalize any contradiction, a wrong amount, "
    "or a refund granted when the expected output says it must be denied. "
    "Differences in wording or politeness do not matter."
)


def build_metrics(judge, threshold: dict[str, float] | None = None) -> dict:
    t = threshold or config.THRESHOLDS
    return {
        # Did it call the expected tools? (names only; ordering not enforced)
        "tool_correctness": ToolCorrectnessMetric(
            threshold=t["tool_correctness"], model=judge
        ),
        # Were the arguments sensible for the request? (LLM judged)
        "argument_correctness": ArgumentCorrectnessMetric(
            threshold=t["argument_correctness"], model=judge
        ),
        # Does the final answer reach the expected conclusion? (LLM judged)
        "answer_correctness": GEval(
            name="Answer Correctness",
            criteria=ANSWER_CRITERIA,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=t["answer_correctness"],
            model=judge,
        ),
    }
