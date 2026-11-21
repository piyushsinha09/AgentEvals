"""The LLM that scores the agent (the "judge"), served through LiteLLM."""

from deepeval.models import LiteLLMModel

from config import JUDGE_MODEL


def build_judge(model: str | None = None) -> LiteLLMModel:
    return LiteLLMModel(model=model or JUDGE_MODEL, temperature=0)
