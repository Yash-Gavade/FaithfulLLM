\
from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are answering benchmark reasoning questions. "
    "Follow the requested output format exactly."
)


def format_choices(choices: list[dict[str, str]] | None) -> str:
    if not choices:
        return ""
    return "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)


def build_prompt(sample: dict[str, Any]) -> str:
    question = sample["question"].strip()
    choices = sample.get("choices")

    if choices:
        choice_block = format_choices(choices)
        answer_rule = (
            "Return the option label and option text in FINAL_ANSWER."
        )
        question_block = f"Question:\n{question}\n\nOptions:\n{choice_block}"
    else:
        answer_rule = "Return only the final numerical or short-text answer in FINAL_ANSWER."
        question_block = f"Question:\n{question}"

    return f"""\
{question_block}

Instructions:
1. {answer_rule}
2. Give a concise justification in EXPLANATION.
3. Do not place the explanation inside FINAL_ANSWER.
4. Use exactly this structure:

FINAL_ANSWER: <answer>
EXPLANATION: <concise justification>
""".strip()
