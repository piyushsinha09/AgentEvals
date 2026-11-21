"""Central configuration. Change models here (or via environment variables)."""

import os

import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True

# LiteLLM uses "provider/model" names. Override with env vars to compare models:
#   AGENT_MODEL=anthropic/claude-sonnet-5-5 uv run python -m scripts.run_eval
AGENT_MODEL = os.getenv("AGENT_MODEL", "anthropic/claude-haiku-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5-5")

# Minimum AVERAGE score per metric for the run to pass.
THRESHOLDS = {
    "tool_correctness": 0.90,
    "argument_correctness": 0.80,
    "answer_correctness": 0.80,
    # Trace-based metrics (judge the whole execution, not just the output).
    # Starting guesses: tune after you have seen a real baseline.
    "task_completion": 0.80,
    "step_efficiency": 0.70,
    "plan_quality": 0.60,
    "plan_adherence": 0.70,
}
ADVISORY_METRICS = {"plan_quality", "plan_adherence"}
# Minimum share of cases that must pass ALL deterministic checks
# (ticket issued or not, correct refund amount, forbidden tools).
MIN_CHECK_PASS_RATE = 0.85

# Fail if any metric average drops more than this vs the saved baseline.
MAX_REGRESSION = 0.05

# Per-metric threshold used for each individual case (pytest mode).
CASE_THRESHOLD = 0.7
