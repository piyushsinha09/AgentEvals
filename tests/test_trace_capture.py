"""Offline test (no API key, free): proves DeepEval's CallbackHandler captures the agent's
full execution trace and that the trace-evaluation plumbing returns per-case scores.

It uses a scripted fake model and a probe metric instead of a real LLM judge."""

from deepeval.metrics import BaseMetric
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS
from evals.trace_eval import run_traced


class ScriptedModel(BaseChatModel):
    """Fake LLM: first call requests two tools, second call answers."""

    step: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.step += 1
        if self.step == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_user_purchase_history", "args": {"username": "user101"}, "id": "t1"},
                    {"name": "get_today_date", "args": {}, "id": "t2"},
                ],
            )
        else:
            msg = AIMessage(content="Your headphones are eligible.")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class ProbeMetric(BaseMetric):
    """Trace metric that only checks it was handed a non-empty trace."""

    requires_trace = True
    trace_seen = None

    def __init__(self):
        self.threshold = 0.5
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self):
        return "Probe"

    def measure(self, test_case, *args, **kwargs):
        ProbeMetric.trace_seen = getattr(test_case, "_trace_dict", None)
        self.score = 1.0 if ProbeMetric.trace_seen else 0.0
        self.success = bool(ProbeMetric.trace_seen)
        self.reason = "trace captured" if self.success else "NO TRACE"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return bool(self.success)


def test_callback_handler_captures_trace_and_scores():
    agent = create_agent(
        model=ScriptedModel(), tools=TOOLS, system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    cases = [{
        "id": "case_a",
        "input": "I'm user101, refund my headphones",
        "expected_output": "eligible",
        "expected_tools": ["get_user_purchase_history", "get_today_date"],
        "expect_ticket": False,
        "expected_amount": None,
    }]

    runs, stage_error = run_traced(agent, cases, [ProbeMetric()], key_map={"Probe": "probe"})

    assert stage_error is None
    run = runs["case_a"]
    assert run.error is None
    # the agent really ran and its tool calls were extracted
    assert [c["name"] for c in run.result.tool_calls] == ["get_user_purchase_history", "get_today_date"]
    # the trace was captured by the CallbackHandler and handed to the metric
    assert ProbeMetric.trace_seen, "CallbackHandler did not capture a trace"
    assert "tools_called" in ProbeMetric.trace_seen
    # and the score came back into our results
    assert run.trace_scores["probe"]["score"] == 1.0
