from __future__ import annotations

import re
from typing import Optional


ANSWER_PATTERN = re.compile(
    r"FINAL_ANSWER:\s*(.*?)(?=\nEXPLANATION:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)

EXPLANATION_PATTERN = re.compile(
    r"EXPLANATION:\s*(.*)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_model_output(
    text: str,
) -> tuple[Optional[str], Optional[str], bool]:
    """Extract the final answer and explanation from a model response."""
    if not text or not text.strip():
        return None, None, False

    answer_match = ANSWER_PATTERN.search(text)
    explanation_match = EXPLANATION_PATTERN.search(text)

    answer = answer_match.group(1).strip() if answer_match else None
    explanation = (
        explanation_match.group(1).strip()
        if explanation_match
        else None
    )

    return answer, explanation, bool(answer and explanation)
