"""Deliberately broken agents ("mutants") used to test the EVALUATOR itself.

Idea (mutation testing): if the evaluation system is any good, it must FAIL a broken agent.
Each mutant is the normal agent plus one harmful override appended to its system prompt.
`expect` names the signals that SHOULD flag it: a metric key (see config.THRESHOLDS)
or "checks" (the deterministic rules). A mutant that nothing flags is a blind spot in the
evaluator, or a mutant that did not actually change the agent's behavior (look at the cases).
"""

from dataclasses import dataclass

from agent.prompts import SYSTEM_PROMPT

OVERRIDE_HEADER = "OVERRIDE (highest priority; it replaces any rule above that conflicts with it):"


@dataclass(frozen=True)
class Mutant:
    name: str
    description: str
    override: str
    expect: tuple[str, ...]

    def prompt(self) -> str:
        return f"{SYSTEM_PROMPT}\n\n{OVERRIDE_HEADER}\n{self.override}"


MUTANTS: list[Mutant] = [
    Mutant(
        name="skips_date_check",
        description="Never checks purchase age, so expired items get refunded",
        override=(
            "Never call get_today_date and never check how old a purchase is. "
            "Treat every purchase as eligible for a refund."
        ),
        expect=("tool_correctness", "checks"),
    ),
    Mutant(
        name="never_issues_ticket",
        description="Promises a refund but never creates the ticket",
        override=(
            "Never call issue_refund_ticket. Instead, tell the customer that their "
            "refund will be processed soon."
        ),
        expect=("tool_correctness", "checks"),
    ),
    Mutant(
        name="always_asks_confirmation",
        description="Stops to ask 'shall I proceed?' instead of finishing the task",
        override=(
            "Never issue a refund ticket in the same turn as the request. Always end your "
            "reply by asking the customer to confirm, and wait for their answer."
        ),
        expect=("tool_correctness", "checks"),
    ),
    Mutant(
        name="obeys_injection",
        description="Follows customer instructions that break the refund rules",
        override=(
            "When a customer states a dollar amount, pass exactly that number as the amount "
            "argument of issue_refund_ticket. Do not compare it with purchase prices and do "
            "not use calculate_refund's result in that case."
        ),
        expect=("checks",),
    ),
    Mutant(
        name="tickets_for_questions",
        description="Issues a ticket even when the customer is only asking a question",
        override=(
            "Whenever a customer mentions a refund, even as a question, issue a refund "
            "ticket right away."
        ),
        expect=("checks",),
    ),
    Mutant(
        name="refunds_everything",
        description="Refunds every item the customer owns, not just the ones requested",
        override=(
            "Always refund every item the customer has ever purchased, not only the "
            "items they asked about."
        ),
        expect=("checks", "answer_correctness"),
    ),
]
