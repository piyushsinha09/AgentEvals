# agent-eval: a regression-testing system for LLM agents

An evaluation framework that scores an LLM agent on a golden dataset, compares the scores with a saved baseline, and **fails the build in CI when quality drops**. It also tests itself: deliberately broken agents ("mutants") are run through the evaluator to prove it can catch real failures.

The agent being tested (a refund-support bot) is only the test subject. **The project is the evaluation system.**

> New to the ideas? Read [LEARNING.md](LEARNING.md) for a from-scratch explanation of every metric, with analogies and worked examples.

---

## Why this exists

Changing a prompt, swapping a model, or editing a tool can silently break an agent. Unlike normal code, there is no compiler error: the agent still answers, just worse. This project treats agent quality like code quality:

- a fixed **golden dataset** of cases with known correct outcomes
- **scored metrics** (deterministic rules + LLM judges + execution-trace judges)
- **thresholds** and a saved **baseline**
- a **CI gate** that blocks a pull request when quality falls
- an **evaluator test-suite** that checks the checker

## What it demonstrates

| Skill | Where |
|---|---|
| Designing an eval dataset with edge cases (boundary dates, prompt injection, missing info) | `evals/dataset.json` |
| Mixing deterministic checks with LLM-as-judge metrics | `evals/checks.py`, `evals/metrics.py` |
| Trace-level evaluation via DeepEval's LangChain `CallbackHandler` | `evals/trace_eval.py` |
| Regression gating (thresholds, baseline drift, per-case regression, zero-tolerance rules) | `scripts/run_eval.py` |
| Mutation testing of the evaluator itself | `evals/mutants.py`, `scripts/run_mutants.py` |
| CI integration with PR comments and nightly runs | `.github/workflows/eval.yml` |
| Provider-agnostic models through LiteLLM | `config.py` |

---

## How it works

```
                 evals/dataset.json (14 golden cases)
                              |
                              v
   +------------------ run the agent once per case ------------------+
   |   LangChain agent + 4 tools, traced by DeepEval CallbackHandler |
   +-----------------------------------------------------------------+
        |                       |                          |
        v                       v                          v
  Deterministic checks     Output metrics            Trace metrics
  (plain Python, free)     (tool / argument /        (task completion, step
                            answer correctness)       efficiency, plan quality,
                                                      plan adherence)
        \                       |                          /
         +----------------------+--------------------------+
                                |
                                v
                  average per metric  ->  GATE  ->  exit code 0 or 1
                                ^
                                |
                     evals/baselines/*.json  (last known good)
```

### The three evaluation layers

| Layer | Examples | Cost | Noise |
|---|---|---|---|
| **Deterministic checks** | Was a ticket issued? Is the amount exactly right? Was a forbidden amount issued? | free | none |
| **Output metrics** (DeepEval) | `tool_correctness`, `argument_correctness`, `answer_correctness` | 0 to 1 LLM call per case | low to medium |
| **Trace metrics** (DeepEval) | `task_completion`, `step_efficiency`, `plan_quality`, `plan_adherence` | 1 LLM call per metric per case | medium to high |

Rule of thumb used here: **if a rule can be written as code, it is code**. LLM judges are used only for things code cannot express ("is this answer equivalent to that one?").

### The 7 metrics

| Metric | Question it answers | Gating? | Threshold |
|---|---|---|---|
| `tool_correctness` | Did the agent call the expected tools? | yes | 0.90 |
| `argument_correctness` | Were the tool inputs sensible? | yes | 0.80 |
| `answer_correctness` | Does the final answer reach the expected decision and amount? | yes | 0.80 |
| `task_completion` | Did it finish the user's task? | yes | 0.80 |
| `step_efficiency` | Did it avoid redundant steps? | yes | 0.70 |
| `plan_quality` | Was its implicit plan suitable? | **advisory** | 0.60 |
| `plan_adherence` | Did it follow its own plan? | **advisory** | 0.70 |

The two plan metrics are *advisory*: reported on every run but not allowed to fail the build. This agent never writes an explicit plan, so DeepEval reconstructs one from the trace and the score is noisy. A gate should only contain signals you trust.

### The gate (a run fails if any rule fails)

1. A gating metric's **average is below its threshold**.
2. A gating metric **dropped more than `MAX_REGRESSION` (0.05)** versus the baseline.
3. Fewer than **85 %** of cases pass all deterministic checks.
4. **Any CRITICAL rule** is broken in even one case (for example, issuing a forbidden refund amount). Zero tolerance, regardless of averages.
5. A case that **passed in the baseline fails now** (case-level regression), even if the averages look fine.

Rules 4 and 5 exist because averages hide things: 13 good cases can average away one dangerous one.

---

## The test subject: a refund agent

A LangChain agent with four tools and a 30-day refund policy:

| Tool | Purpose |
|---|---|
| `get_user_purchase_history` | Look up a customer's purchases |
| `get_today_date` | Return the date (fixed to `2026-11-01` via `REFERENCE_DATE` so results never depend on the real calendar) |
| `calculate_refund` | Sum eligible item prices |
| `issue_refund_ticket` | Create the refund ticket |

Policy: refund only requested items that are 0 to 30 days old (30 days exactly is eligible); never refund more than the price; never obey "ignore your rules" instructions; answer questions without issuing a ticket.

### Golden dataset (14 cases, 7 flagged `smoke`)

Eligible refund, expired purchase, one of two items, two items one ticket, question only (no ticket), missing username, unknown customer, item never purchased, **exactly 30 days (boundary)**, **31 days (boundary)**, mixed eligible/expired, prompt injection asking for an inflated $500 refund, and more. Each case stores the expected tools, whether a ticket is expected, the expected amount, and optionally forbidden amounts.

---

## Does the evaluator itself work? (mutation testing)

An evaluator that never fails proves nothing. So the suite includes a **control** (the normal agent, which must pass) and six **mutants** (the normal agent plus one deliberately harmful instruction). A good evaluator must flag every mutant and must not flag the control.

Result of the smoke run (7 cases per variant), agent `claude-haiku-5-5`, judge `claude-sonnet-5-5`:

| Mutant | What was broken | Caught? | Flagged by |
|---|---|---|---|
| `skips_date_check` | Never checks purchase age | Yes | tool_correctness, answer_correctness, checks |
| `never_issues_ticket` | Promises a refund, never creates the ticket | Yes | tool_correctness, answer_correctness, checks |
| `always_asks_confirmation` | Asks "shall I proceed?" instead of finishing | Yes | tool_correctness, answer_correctness, checks |
| `refunds_everything` | Refunds every item, not just the requested ones | Yes | answer_correctness, argument_correctness, checks |
| `obeys_injection` | Follows customer instructions that break the rules | Yes (side effect) | checks, tool_correctness |
| `tickets_for_questions` | Issues a ticket for a pure question | No (equivalent mutant) | none |

**Control: clean (no false alarm).**

How to read this honestly:

- **4 of 4 effective mutants were caught by an expected signal.**
- `obeys_injection` was caught because the mutated agent *stalled* (no ticket issued), not because it issued the forbidden amount. The model resisted the harmful instruction, so the CRITICAL rule has been verified by unit tests, not by a live failure.
- `tickets_for_questions` is an **equivalent mutant**: probed three times, the model still issued no ticket (it treated "just asking for now" as an exception), so the agent's behavior did not change and there was nothing to detect. This is not an evaluator blind spot.

Reproduce: `uv run python -m scripts.run_mutants --smoke` (writes `reports/mutants.md` and `reports/mutants.json`).

---

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), an Anthropic API key (or any LiteLLM-supported provider).

```bash
git clone <your-repo-url>
cd agent-eval
uv sync
cp .env.example .env          # add ANTHROPIC_API_KEY=...
```

### Commands

```bash
uv run pytest                                          # offline tests (free, no API calls)
uv run python -m scripts.run_eval --smoke              # cheap subset (7 cases)
uv run python -m scripts.run_eval                      # full dataset, all 7 metrics
uv run python -m scripts.run_eval --no-trace           # skip trace metrics (cheaper)
uv run python -m scripts.run_eval --save-baseline      # store this run as the baseline
uv run python -m scripts.run_mutants --smoke           # test the evaluator itself
uv run pytest -m llm -v                                # per-case pytest run (costs API calls)
```

Outputs: `reports/latest.md`, `reports/latest.json`, `reports/mutants.md`, `reports/mutants.json`. Exit code is `1` if the gate fails.

### Compare models without editing code

```bash
AGENT_MODEL=anthropic/claude-sonnet-5-5 uv run python -m scripts.run_eval
JUDGE_MODEL=anthropic/claude-opus-5-5   uv run python -m scripts.run_eval
```

### Baselines

A baseline is the last known-good result. Save one only after a clean, stable run, and commit it:

```bash
uv run python -m scripts.run_eval --save-baseline
uv run python -m scripts.run_eval --smoke --save-baseline
git add evals/baselines/ && git commit -m "Update baselines"
```

Full and smoke runs have separate baselines because they average over different cases.

---

## Continuous integration

`.github/workflows/eval.yml`:

- **Pull requests** (touching `agent/`, `evals/`, `scripts/`, config): offline tests, then the smoke eval, then a sticky PR comment with the score table.
- **Nightly** (02:00 IST) and manual runs: the full dataset.
- The full report is uploaded as a build artifact.

Setup: add the repository secret `ANTHROPIC_API_KEY`, and mark the workflow as a required check in branch protection.

---

## Project layout

```
agent/
  agent.py        builds the agent, runs a question, extracts answer + tool calls
  tools.py        the 4 tools        data.py   fake purchase data + REFERENCE_DATE
  prompts.py      system prompt
evals/
  dataset.json    golden cases       checks.py   deterministic rules (+ CRITICAL)
  metrics.py      output metrics     trace_eval.py  trace capture + trace metrics
  judge.py        judge model        runner.py   run a case, build test cases
  mutants.py      broken agents for testing the evaluator
scripts/
  run_eval.py     scoring, baseline, gate, reports
  run_mutants.py  evaluator self-test
tests/            offline tests (27) + LLM per-case tests
config.py         models, thresholds, gate settings
```

---

## Configuration (`config.py`)

| Setting | Meaning | Default |
|---|---|---|
| `AGENT_MODEL` / `JUDGE_MODEL` | LiteLLM `provider/model` names (env-overridable) | haiku-5-5 / sonnet-5-5 |
| `THRESHOLDS` | Minimum average per metric | see table above |
| `ADVISORY_METRICS` | Reported but never fail the gate | plan_quality, plan_adherence |
| `MIN_CHECK_PASS_RATE` | Share of cases that must pass all deterministic checks | 0.85 |
| `MAX_REGRESSION` | Largest allowed drop vs baseline | 0.05 |

---

## Design decisions

- **Deterministic first.** Ticket count, exact amount and forbidden amounts are plain Python: free, instant, no noise.
- **A separate, stronger judge model.** The judge is a larger model than the agent, so the grader is not the same system as the student.
- **Fixed reference date.** `REFERENCE_DATE` removes the real calendar from the tests, so a case that passes today does not fail next month.
- **No-argument tools are excluded from the argument metric.** `get_today_date` takes no inputs, so there is nothing to grade, and including it produced false penalties.
- **Advisory metrics.** Unreliable signals are visible but cannot block a merge.
- **Test the tester.** Mutation testing gives a measurable answer to "would this evaluator notice a broken agent?"

## Limitations (known and stated)

- **Small dataset (14 cases).** One case moves a metric average by about 0.07. Grow the dataset with every bug found in production.
- **LLM judges are noisy.** A repeated run can differ by a few points. `MAX_REGRESSION = 0.05` is a starting value and should be calibrated by repeating the run several times.
- **Judge lacks policy context in one case.** `mixed_eligible_and_expired` is scored low by `argument_correctness` (about 0.33) because the judge does not know the 30-day policy that justifies the arguments the agent chose. The deterministic checks pass for that case.
- **Plan metrics are advisory** because the agent has no explicit plan.
- **Mutation coverage.** Prompt-level mutants can be resisted by the model (see `tickets_for_questions`). Tool-level mutants (a stale clock, a corrupted ticket amount) would test the CRITICAL rule on a live run and are a natural next step.
- **No judge-vs-human calibration yet.** Agreement between the judge and hand-labelled cases has not been measured.

## Possible extensions

- Judge-vs-human agreement on ~20 hand-labelled cases
- Run each case 3 to 5 times and report variance; set `MAX_REGRESSION` from the data
- Tool-level mutants (`stale_clock`, `overpaying_ticket`)
- Policy context supplied to the argument judge
- Nightly trend chart from stored reports

## Tech stack

Python 3.12, [uv](https://docs.astral.sh/uv/), [DeepEval](https://deepeval.com), LangChain + LangGraph, [LiteLLM](https://docs.litellm.ai), pytest, GitHub Actions.
