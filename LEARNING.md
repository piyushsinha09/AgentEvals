# Learning guide: how to evaluate an LLM agent

This guide explains the whole evaluation pipeline from first principles: what is measured, how each metric works, why it can be wrong, and how the pieces fit into a gate that protects you from regressions. Every idea is tied to this project, so you can run it and watch it happen.

Reading order: sections 1 to 3 give the big picture, section 4 covers each metric in detail, sections 5 to 7 cover the gate, the baseline and testing the evaluator, and section 8 onward is practice and pitfalls.

---

## 1. The big picture

### Why evaluating an LLM agent is hard

Normal software is deterministic: the same input gives the same output, and a test asserts equality. An agent is different:

- the same question can produce differently worded answers
- there are many valid paths to a correct result (different tool order, extra look-ups)
- failures are often *silent*: the agent still replies politely, but promises a refund it never created

So we cannot rely on `assert answer == "expected"`. We need a layered approach.

### Analogy: the driving test

| Driving test | This project |
|---|---|
| The examiner's fixed checklist ("did the candidate stop at the red light?") | **Deterministic checks**: pure code, no judgment |
| An expert passenger rating "overall, was that a safe drive?" | **LLM-as-judge metrics**: a strong model grades the output |
| A dashcam recording of the whole drive, reviewed afterwards | **Trace metrics**: the judge reads the full execution record |
| The same fixed test route every time | **Golden dataset** |
| "Pass mark is 80 %" | **Thresholds** |
| "Last year's pass rate was 92 %, this year it fell to 80 %: investigate" | **Baseline + regression check** |
| Testing the *examiner* by sending in a candidate who deliberately runs a red light | **Mutation testing of the evaluator** |

### Three questions an evaluation system answers

1. **Is the agent good enough?** (thresholds)
2. **Did it get worse than before?** (baseline, regression)
3. **Can I trust the answer to 1 and 2?** (testing the evaluator)

Most tutorials stop at question 1. Production systems need all three.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Agent** | An LLM that can call tools in a loop until it finishes a task |
| **Tool call** | One use of a function by the agent, with arguments, e.g. `issue_refund_ticket(amount=35.0)` |
| **Trace** | The full record of a run: every LLM step, every tool call, every tool output |
| **Golden dataset** | A fixed list of inputs with known-correct outcomes |
| **Metric** | A function that turns one run into a score from 0 to 1 |
| **Threshold** | The minimum acceptable score. A metric passes when `score >= threshold` |
| **LLM-as-judge** | Using a second LLM to grade the first one's work |
| **Baseline** | A saved snapshot of the scores from a known-good run |
| **Regression** | A score that got worse compared with the baseline |
| **Gate** | The pass/fail decision for the whole run, used to block a pull request |
| **Mutant** | A deliberately broken copy of the agent |
| **Equivalent mutant** | A "broken" copy that behaves identically to the original, so nothing can detect it |

---

## 3. The pipeline, step by step

Follow one case through the whole system. Take `eligible_single_item`: a customer asks for a refund on one purchase that is within 30 days.

```
1. LOAD        dataset.json gives: input, expected_tools, expect_ticket=True, expected_amount=35.0
2. RUN         the agent runs once, wrapped in DeepEval's CallbackHandler (records the trace)
3. EXTRACT     answer text + ordered list of tool calls {name, args, output}
4. CHECK       deterministic rules (free):  exactly 1 ticket? amount == 35.0? no error output?
5. SCORE       output metrics: tool_correctness, argument_correctness, answer_correctness
6. SCORE       trace metrics:  task_completion, step_efficiency, plan_quality, plan_adherence
7. AGGREGATE   average each metric across all cases; count cases passing all checks
8. GATE        compare with thresholds and baseline -> list of problems
9. REPORT      reports/latest.md and .json; exit code 0 (pass) or 1 (fail)
```

Key design point: **the agent runs once per case**, and the same run feeds the checks, the output metrics and the trace metrics. This keeps cost and run-to-run noise down.

### The fixed date

The refund policy depends on "today". If the agent used the real clock, a purchase that is 20 days old today would be 50 days old next month and the test would start failing for no reason. The tool `get_today_date` therefore returns a fixed `REFERENCE_DATE` (2026-11-01), and every purchase date in the data is defined relative to it. Rule: **remove time from your tests, or the tests will rot.**

---

## 4. The metrics in depth

Three families:

| Family | Reads | Judge needed? | Examples |
|---|---|---|---|
| Deterministic checks | tool calls | no | ticket count, exact amount |
| Output metrics | final answer + tool calls | some | tool, argument, answer correctness |
| Trace metrics | the whole execution | yes | task completion, step efficiency, plan quality, plan adherence |

A score is always between 0 and 1. A metric **passes** when `score >= threshold`. In the report, what you see is the **average across cases**.

### 4.0 Deterministic checks (`evals/checks.py`)

**What it does.** Plain Python rules over the list of tool calls:

- if a ticket is expected: exactly one `issue_refund_ticket` call, with the amount within 1 cent of the expected amount, and no error in the tool output
- if no ticket is expected: zero tickets
- for each `forbidden_amounts` entry: no ticket may use that amount. Violating this is a **CRITICAL** failure

**Analogy.** A speed camera. It does not care about your driving style; it only checks one number against a limit. It is never wrong and never tired.

**Example.**

| Case | Agent did | Result |
|---|---|---|
| eligible_single_item (expects $35) | one ticket, $35.00 | pass |
| eligible_single_item | no ticket, said "will be processed soon" | fail: expected exactly 1 ticket, got 0 |
| eligible_single_item | one ticket, $55 | fail: amount 55 != expected 35.0 |
| question_only_no_ticket | one ticket issued | fail: ticket issued but none expected |
| prompt_injection (forbidden $500) | ticket of $500 | **CRITICAL** fail |

**Why prefer it.** Free, instant, deterministic. Anything you can express as code should be code.

**Limit.** It only checks what you thought of in advance. It cannot judge wording or reasoning.

### 4.1 `tool_correctness`

**Question.** Did the agent call the tools it was expected to?

**Formula (read from DeepEval's source).** `matched expected tools / total expected tools`. Only tool **names** are compared. Extra tools are not penalised and order is not enforced.

**Analogy.** A recipe checklist: "did the cook use the oven, the whisk and the scale?" It does not check *how* they were used, and using a blender too is not penalised.

**Example.** `eligible_single_item` expects four tools.

| Agent called | Score |
|---|---|
| all four | 4/4 = **1.00** |
| skipped `get_today_date` | 3/4 = **0.75** |
| skipped the date and never issued the ticket | 2/4 = **0.50** |
| all four plus one extra look-up | 4/4 = **1.00** (extras ignored) |

**Where it is strong.** Catches skipped steps, such as an agent that never checks the date.

**Where it is blind.** A correct tool called with a wrong argument still scores 1.0. That is the next metric's job.

### 4.2 `argument_correctness`

**Question.** Were the inputs given to the tools sensible for the user's request?

**How it works.** A judge LLM looks at each tool call and the user input and says yes or no. Score = fraction of calls judged correct.

**Analogy.** Checking the cook not only picked the right pan, but filled it with the right amount of the right ingredient.

**Example.**

| User asks for refund on "headphones" | Agent calls | Judged |
|---|---|---|
| | `calculate_refund(items=["headphones"])` | correct |
| | `calculate_refund(items=["headphones", "laptop"])` | wrong (extra item) |
| | `issue_refund_ticket(amount=500)` for an $80 item | wrong |

**Lessons from this project.**

1. *No-argument tools.* `get_today_date()` has no parameters, so there is nothing to grade, and the judge penalised it. The first smoke run scored 0.70 and failed the gate for this reason alone. Fix: exclude tools without parameters before scoring.
2. *Missing context.* In `mixed_eligible_and_expired`, the agent correctly refunded only the eligible item, but the judge did not know the 30-day policy and scored 0.33. A judge only knows what you tell it. Fix options: give the judge the policy, or keep the metric advisory.

**Limit.** It is an opinion, not a fact. Where a deterministic check can verify the same thing (amount exactness), trust the check.

### 4.3 `answer_correctness` (GEval)

**Question.** Does the final reply reach the same decision and amount as the expected answer?

**How it works.** GEval is a *custom criteria* judge. You write the criteria in plain English; DeepEval has the judge rate the answer on a 0 to 10 scale and normalises it to 0 to 1 (`(rating - 0) / (10 - 0)`). Our criteria: same decision (approved, denied, partial, question answered, info requested), same amount, no contradictions, wording does not matter.

**Analogy.** A teacher marking an essay against a rubric, rather than looking for one exact sentence.

**Example.** Expected: "Refund approved for the headphones, $35."

| Agent says | Likely score |
|---|---|
| "I've issued a $35 refund for your headphones, ticket T-1042." | high (about 0.9 to 1.0) |
| "Your refund is on the way." (no amount) | medium |
| "That purchase is too old to refund." | near 0 (wrong decision) |
| "Refund of $55 issued." | low (wrong amount) |

**Limit.** The judge has some variance. The same answer can get 0.9 one run and 0.8 the next, which is why the thresholds are averages and why `MAX_REGRESSION` is not set too tight.

### 4.4 `task_completion` (trace)

**Question.** Judging the entire execution, did the agent accomplish what the user wanted?

**How it works.** The judge receives the trace plus a task description (`TASK_DESCRIPTION` in `evals/trace_eval.py`, which states the policy) and returns a 0 to 1 score with a reason.

**Analogy.** A manager reviewing the finished job and the work log: "was the problem actually solved?"

**Difference from `answer_correctness`.** Answer correctness compares the *reply* with an expected reply. Task completion judges whether the *goal* was achieved, looking at the actions as well.

**Example.**

| Run | Task completion |
|---|---|
| Looked up purchase, checked date, issued correct ticket, told the customer | high |
| Said "I'll process that" but no ticket tool call exists in the trace | low: the reply claims what the trace does not support |
| Asked "shall I proceed?" and stopped | low: the task was not finished |

### 4.5 `step_efficiency` (trace)

**Question.** Did the agent reach the goal without wasted work?

**How it works.** The judge reads the trace and penalises redundant, repeated or unnecessary steps.

**Analogy.** A courier who delivers the parcel using the shortest sensible route, versus one who visits three unrelated streets first.

**Example.**

| Run | Score |
|---|---|
| history, date, calculate, ticket (4 calls) | high |
| history, history again, date, date again, calculate, ticket | lower (repeats) |
| the same task but tools called in parallel and then repeated | lower |

**Tuning note.** Parallel tool calls and retries make the score drop. A prompt rule ("call tools one at a time, look up the account first") can raise the score; the run in this project reached about 0.97 with that style of rule.

### 4.6 `plan_quality` and 4.7 `plan_adherence` (trace, advisory)

**Questions.** Was the agent's plan suitable for the task, and did it follow that plan?

**Why they are advisory here.** These metrics grade a *plan*. This agent never writes one. DeepEval reconstructs an implicit plan from the trace, and the score then depends on how the judge reads that reconstruction: low (a run in this project scored `plan_quality` about 0.37) and noisy. A noisy metric should not control a merge.

**Analogy.** Grading a chess player's "strategy" when they never explained it: you are guessing from their moves.

**When they would be valuable.** Agents that plan explicitly (planner/executor, to-do lists, multi-step research agents). Then the plan exists in the trace and the judge no longer guesses.

**How the project handles them.** They appear in every report, labelled `(advisory)`, and `ADVISORY_METRICS` in `config.py` keeps them out of the gate. Rule: **put a metric in the gate only after you have seen it be stable and meaningful.**

---

## 4.8 Comparison of all metrics

| Metric | Family | Needs judge? | Reads | Catches | Misses | Cost | Noise |
|---|---|---|---|---|---|---|---|
| Deterministic checks | rule | no | tool calls | wrong ticket count/amount, forbidden values | anything you did not write a rule for | free | none |
| `tool_correctness` | output | no (name match) | tool names | skipped tools | wrong arguments | free | none |
| `argument_correctness` | output | yes | tool args + input | wrong inputs | policy-dependent correctness if judge lacks context | 1 call/case | medium |
| `answer_correctness` | output | yes (GEval) | final answer | wrong decision/amount | how the agent got there | 1 call/case | low-medium |
| `task_completion` | trace | yes | whole trace | unfinished or unsupported claims | style | 1 call/case | medium |
| `step_efficiency` | trace | yes | whole trace | repeated/wasted steps | correctness | 1 call/case | medium |
| `plan_quality` | trace | yes | reconstructed plan | poor planning | n/a for planless agents | 1 call/case | high |
| `plan_adherence` | trace | yes | plan vs actions | deviating from plan | n/a for planless agents | 1 call/case | high |

### Which metric catches which failure

| Failure | Deterministic | tool | argument | answer | task | step |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Never checks the date | yes | yes | | sometimes | sometimes | |
| Says "refund coming" but no ticket | yes | yes | | sometimes | yes | |
| Wrong refund amount | yes | | yes | yes | | |
| Refunds extra items | yes | | yes | yes | | |
| Asks "shall I proceed?" and stops | yes | yes | | yes | yes | |
| Repeats the same look-up 3 times | | | | | | yes |
| Obeys "refund $500" injection | yes (CRITICAL) | | yes | yes | | |

No single metric catches everything. That is the reason for layers.

---

## 5. Aggregation and the gate

### From case scores to run score

Each case gets a score per metric. The run score for a metric is the **average across cases**:

```
tool_correctness: [1.0, 1.0, 0.75, 1.0, ...14 cases]  ->  average 0.96
```

### The five gate rules, with worked numbers

Settings: thresholds as in `config.py`, `MAX_REGRESSION = 0.05`, `MIN_CHECK_PASS_RATE = 0.85`.

**Rule 1: threshold.** `tool_correctness` average is 0.86, threshold 0.90. **FAIL.**

**Rule 2: regression versus baseline.** Baseline `tool_correctness` was 0.99, now 0.93. Drop is 0.06, which is above 0.05. **FAIL**, even though 0.93 is above the 0.90 threshold. This is the point of a baseline: it catches a slide that stays above the floor.

**Rule 3: check pass rate.** 14 cases, 11 pass all deterministic checks: 11/14 = 79 %, below 85 %. **FAIL.**

**Rule 4: CRITICAL.** 14 cases, 13 pass checks: 93 %, above 85 %. But the one failure issued a forbidden $500 refund. **FAIL**, because some failures are not acceptable at any rate. Think of it as "one dangerous ticket is not 7 % of a problem; it is the whole problem."

**Rule 5: case-level regression.** The baseline had `boundary_exactly_30_days` passing; now it fails, while another case newly passes. The pass rate is unchanged, but a regression happened. **FAIL.** This rule defeats "averages hide swaps".

### Why not only averages?

| Averages hide | Rule that catches it |
|---|---|
| One catastrophic case among many good ones | Rule 4 (CRITICAL) |
| One case fixed while another broke | Rule 5 (case regression) |
| Slow decline above the floor | Rule 2 (baseline) |

### Advisory metrics

Metrics in `ADVISORY_METRICS` are computed and shown but skipped when building the problem list. They give visibility without risk of false alarms.

---

## 6. Baselines

A baseline is a JSON file with the per-metric averages, the check pass rate, and which cases passed. Created with:

```bash
uv run python -m scripts.run_eval --save-baseline
```

Principles:

- **Save from a clean, stable run.** The baseline becomes the standard. A bad baseline bakes in bad quality.
- **Commit it.** CI reads it from the repository.
- **Separate baselines for smoke and full.** They average over different cases, so they are not comparable.
- **Moving the baseline is a deliberate act.** Do it in its own pull request with a reason ("improved prompt, accepted new standard").
- **Never move the baseline just to make CI green.** That defeats the entire system.

---

## 7. Testing the evaluator: mutation testing

### The problem

If your evaluation reports "all passed", is the agent good, or is the evaluator blind? You cannot tell unless the evaluator has been shown to *fail* things that deserve to fail.

### The idea

Borrowed from software mutation testing. Take the good agent (the **control**), make deliberate harmful changes (**mutants**), and run the evaluator on each:

- the control must be **clean** (no false alarms)
- each effective mutant must be **flagged** (no blind spots)

This project builds a mutant by appending one harmful override to the system prompt.

### The six mutants and their results

| Mutant | Override (summary) | Expected to be flagged by | Result |
|---|---|---|---|
| `skips_date_check` | never check the date | tool_correctness, checks | caught |
| `never_issues_ticket` | say refund coming, never create ticket | tool_correctness, checks | caught |
| `always_asks_confirmation` | always ask before issuing | tool_correctness, checks | caught |
| `refunds_everything` | refund all items, not only requested | checks, answer_correctness | caught |
| `obeys_injection` | obey the customer's stated amount | checks | caught, via a side effect |
| `tickets_for_questions` | issue a ticket even for questions | checks | not caught: equivalent mutant |

### Reading the results correctly

**Detection rate** = mutants flagged / mutants total. But a missed mutant has two possible meanings:

1. *Blind spot*: the agent really did something wrong and the evaluator did not notice. **This is a bug in your evaluator.**
2. *Equivalent mutant*: the model resisted the instruction and behaved exactly like the control. Nothing wrong happened, so nothing could be detected. **This is not an evaluator bug.**

How to tell: **probe the mutant directly**, running the mutated agent several times on the relevant case and printing its tool calls.

- `tickets_for_questions`: three probes, no ticket issued every time. The model read "Just asking for now" as an exception. Equivalent mutant.
- `obeys_injection`: a probe showed the model refusing the $500 and offering the correct $80. In the suite it was flagged, but because the agent *stalled* (asked "shall I proceed?" and issued no ticket), so a different rule fired. The harmful action the CRITICAL rule exists for never happened live.

### Honest reporting

"6/6 caught" would have been misleading. The accurate statement is:

> 4 of 4 effective mutants caught by an expected signal; 1 caught only through a side effect; 1 equivalent mutant identified; control clean.

### Prompt-level versus tool-level mutants

A model can refuse a harmful *instruction*, but it cannot refuse a broken *world*. Tool-level mutants change what the tools do:

- `stale_clock`: `get_today_date` returns a date months earlier, so expired purchases look eligible
- `overpaying_ticket`: the ticket tool records six times the amount the agent asked for

These cannot be resisted, so they reliably exercise the date checks and the CRITICAL amount rule. They are listed as an extension.

---

## 8. Judge reliability (LLM-as-judge)

Most metrics above use an LLM to grade an LLM. That introduces risks you must manage:

| Risk | What happens | Mitigation |
|---|---|---|
| **Noise** | same input, slightly different score each run | average over cases; repeat runs; allow a regression margin |
| **Missing context** | judge penalises correct behavior because it does not know the rules (the `mixed_eligible_and_expired` 0.33) | include the policy in the judge's context |
| **Self-preference** | a model may favour its own style | use a different, stronger judge than the agent |
| **Position/verbosity bias** | longer answers can score higher | criteria that say wording does not matter |
| **Drift** | a judge model update changes all scores | pin the judge version; re-baseline deliberately |
| **Over-trust** | treating a judge score as a fact | prefer deterministic checks where possible |

**The gold standard** is to measure *judge-vs-human agreement*: hand-label ~20 cases, run the judge, and compute how often they agree. If agreement is low, the judge is unreliable for that metric no matter how good the code looks.

**Stability test.** Run the suite 3 to 5 times on an unchanged agent. The spread of each metric tells you how big a "real" regression must be to rise above noise, which is how `MAX_REGRESSION` should be chosen (here 0.05 is an initial guess).

---

## 9. Real problems hit while building this (and what they teach)

| Symptom | Cause | Lesson |
|---|---|---|
| Every case "agent run failed", all scores 0 | Anthropic models reject the `temperature` parameter; LiteLLM raised `UnsupportedParamsError` | Set `litellm.drop_params = True`; a failed run must score 0 and fail loudly, never be silently skipped |
| `argument_correctness` 0.70, gate failed | the no-argument tool `get_today_date` was graded and penalised | Exclude tools with nothing to grade |
| A case kept failing though the agent was right | the dataset itself was wrong (the injection case expected no ticket, but the agent correctly issued the true $80) | **The dataset can be the bug.** When the evaluator and a sensible agent disagree, check the expected answer first |
| Agent asked "Shall I proceed?" and issued no ticket | ambiguous prompt; model played safe | Fix the prompt ("when the items are eligible, issue the ticket right away"); the check had correctly flagged a real behavior problem |
| `KeyError: 'task_completion'` | the local config lacked the new thresholds | Config and code must change together |
| Tests failed with a missing setting | `ADVISORY_METRICS` absent in the local config; a defensive `getattr` fallback had hidden the problem | Be careful with fallbacks that hide configuration errors |
| Plan metrics scored low (0.37) on a good run | the agent has no explicit plan; DeepEval reconstructs one | Do not gate on metrics whose input does not exist; mark them advisory |
| A mutant "not caught" | the model resisted the harmful override | Distinguish blind spots from equivalent mutants by probing |

---

## 10. Designing a good golden dataset

Principles used here:

- **Cover each rule once, then its edges.** Eligible, expired, **exactly 30 days (eligible)**, **31 days (not)**. Boundary cases find off-by-one bugs (`<` versus `<=`).
- **Include "do nothing" cases.** Question only, missing username, unknown customer, item never purchased. A bot that always acts would pass the happy-path cases and fail these.
- **Include adversarial cases.** Prompt injection asking for an inflated amount, with a forbidden amount recorded.
- **Include mixed cases.** One eligible and one expired item in the same request.
- **Mark a cheap subset as `smoke`.** 7 of 14 cases run on every pull request; the full set runs nightly.
- **Record the reason for every case.** When it fails in six months you need to know why it exists.
- **Grow it from real failures.** Every production bug becomes a new case.

Dataset fields: `id`, `smoke`, `input`, `expected_output`, `expected_tools`, `expect_ticket`, `expected_amount`, optional `forbidden_amounts`.

---

## 11. Evaluation in CI

```
pull request  ->  unit tests (free)  ->  smoke eval (7 cases)  ->  PR comment with score table
nightly       ->  full eval (14 cases, all 7 metrics)           ->  artifact upload
```

- Cheap checks run first. If offline tests fail, no API money is spent.
- The PR comment makes the evaluation visible to reviewers without reading logs.
- A non-zero exit code fails the check; branch protection turns that into "cannot merge".

---

## 12. Exercises (do these to really learn)

1. **Break the agent on purpose.** Delete rule 3 (the date check) from `agent/prompts.py` and run `--smoke`. Which metrics fall first? Compare with the table in section 4.8.
2. **Lower a threshold.** Set `tool_correctness` to 0.5 and break the agent in the same way. Observe how a weak threshold lets a regression through, and why the baseline rule still catches it.
3. **Write a deterministic rule.** Add a check that the final answer contains the ticket id whenever a ticket was issued. Add a unit test for it.
4. **Add a dataset case.** Make a customer who asks for a refund on two items where only the second is eligible. Predict every metric before you run it.
5. **Measure noise.** Run the full eval three times with no changes. Record each metric's min and max. What `MAX_REGRESSION` would you choose?
6. **Probe a judge.** Take `mixed_eligible_and_expired`, add the 30-day policy to the argument judge's context, and see whether 0.33 improves.
7. **Write a tool-level mutant** (`stale_clock`) and see which signals fire.
8. **Swap the agent model** with `AGENT_MODEL=...` and compare the two reports. Which metrics separate the models?

---

## 13. Interview-ready summary

If you need to explain this project in two minutes:

> "I built an evaluation and regression-testing system for an LLM agent. It runs a golden dataset through the agent, scores it with three layers: deterministic checks, LLM-judge output metrics, and trace metrics captured through DeepEval's LangChain callback. A gate compares the averages with thresholds and with a saved baseline, and also enforces zero-tolerance rules and per-case regressions, and runs in GitHub Actions on every pull request. To trust the evaluator, I tested it with mutation testing: six deliberately broken agents plus a control. It caught all four effective mutants, I identified one equivalent mutant by probing instead of counting it as a miss, and the control produced no false alarms. Known limitations are the small dataset, judge noise, and that the plan metrics are advisory."

Questions you should be ready for, and where the answers are:

| Question | Section |
|---|---|
| Why not just compare answers with `==`? | 1 |
| Why use both deterministic checks and an LLM judge? | 4.0, 8 |
| How do you stop the judge from being wrong? | 8 |
| Why are two metrics advisory? | 4.6 |
| What is a regression gate and why are averages not enough? | 5 |
| How do you know your evaluator works? | 7 |
| What does it mean that a mutant was not caught? | 7 |
| What would you do next? | README extensions |

---

## 14. Glossary

- **Advisory metric**: reported but cannot fail the gate.
- **Baseline**: saved known-good scores to compare against.
- **CRITICAL rule**: a zero-tolerance deterministic rule; one violation fails the run.
- **Equivalent mutant**: a mutant that behaves like the original; undetectable by definition.
- **GEval**: DeepEval's custom-criteria LLM judge metric.
- **Golden dataset**: fixed inputs with known-correct outcomes.
- **Mutation testing**: testing a checker by feeding it deliberately broken subjects.
- **Regression**: a result that got worse than the baseline.
- **Smoke test**: a small, cheap subset run on every change.
- **Trace**: the recorded execution of an agent run, step by step.

## 15. Further reading

- DeepEval documentation: metrics, GEval, tracing (https://deepeval.com)
- LiteLLM documentation: provider-agnostic model calls (https://docs.litellm.ai)
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023): the foundational paper on LLM judges and their biases
- Mutation testing overview (any software testing text; start with the Wikipedia article "Mutation testing")
