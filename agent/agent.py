"""Builds the refund agent and runs it on one question."""

import uuid
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver

from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS
from config import AGENT_MODEL


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)  # {name, args, output}


def build_agent(
    model: str | None = None,
    system_prompt: str | None = None,
    tools: list | None = None,
):
    """system_prompt and tools can be overridden to build deliberately broken agents
    (see evals/mutants.py)."""
    llm = ChatLiteLLM(model=model or AGENT_MODEL, temperature=0)
    return create_agent(
        model=llm,
        tools=tools or TOOLS,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def _text(content) -> str:
    """Message content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def extract_result(messages: list) -> AgentResult:
    """Pull the final answer and the ordered tool calls out of the message list."""
    outputs = {
        m.tool_call_id: _text(m.content) for m in messages if isinstance(m, ToolMessage)
    }
    tool_calls: list[dict] = []
    answer = ""
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                tool_calls.append(
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "output": outputs.get(call["id"], ""),
                    }
                )
            text = _text(m.content)
            if text:
                answer = text  # keep the last non-empty AI text
    return AgentResult(answer=answer, tool_calls=tool_calls)


def run_agent(
    agent, question: str, thread_id: str | None = None, callbacks: list | None = None
) -> AgentResult:
    """Run one question in a fresh conversation (new thread id = no shared memory).
    Pass callbacks (e.g. DeepEval's CallbackHandler) to capture the execution trace."""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    if callbacks:
        config["callbacks"] = callbacks
    state = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    return extract_result(state["messages"])
