from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def extract_first_number(value: str | None) -> str | None:
    if not value:
        return None
    match = NUMBER_PATTERN.search(value.replace(",", ""))
    return match.group(0) if match else None


def extract_choice_label(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = value.strip().upper()
    match = re.match(r"^\s*([A-E])(?:[\.\):\-\s]|$)", cleaned)
    return match.group(1) if match else None


def normalize_yes_no(value: str | None) -> str | None:
    if not value:
        return None

    text = normalize_text(value)

    # Direct textual answers
    if text in {"yes", "true"}:
        return "yes"
    if text in {"no", "false"}:
        return "no"

    # Option-label answers used by the StrategyQA prompt
    label = extract_choice_label(value)
    if label == "A":
        return "yes"
    if label == "B":
        return "no"

    # Slightly more permissive textual forms
    if re.search(r"\b(yes|true)\b", text):
        return "yes"
    if re.search(r"\b(no|false)\b", text):
        return "no"

    return None


def score_answer(record: dict[str, Any]) -> tuple[bool, str]:
    dataset = record["dataset"]
    predicted = record.get("predicted_answer")
    gold = record.get("ground_truth")

    if dataset == "gsm8k":
        predicted_number = extract_first_number(predicted)
        gold_number = extract_first_number(gold)

        if predicted_number is None or gold_number is None:
            return False, "numeric_parse_failed"

        try:
            return (
                float(predicted_number) == float(gold_number),
                "numeric_exact_match",
            )
        except ValueError:
            return False, "numeric_parse_failed"

    if dataset == "commonsense_qa":
        predicted_label = extract_choice_label(predicted)
        gold_label = extract_choice_label(gold)

        if predicted_label is None or gold_label is None:
            return False, "choice_parse_failed"

        return (
            predicted_label == gold_label,
            "choice_label_match",
        )

    if dataset == "strategyqa":
        predicted_yes_no = normalize_yes_no(predicted)
        gold_yes_no = normalize_yes_no(gold)

        if predicted_yes_no is None or gold_yes_no is None:
            return False, "yes_no_parse_failed"

        return (
            predicted_yes_no == gold_yes_no,
            "yes_no_match",
        )

    return (
        normalize_text(predicted) == normalize_text(gold),
        "normalized_text_match",
    )


def detect_answer_explanation_consistency(
    record: dict[str, Any],
) -> tuple[bool | None, str]:
    dataset = record["dataset"]
    predicted = record.get("predicted_answer")
    explanation = record.get("explanation") or ""

    if dataset == "gsm8k":
        predicted_number = extract_first_number(predicted)
        numbers = NUMBER_PATTERN.findall(explanation.replace(",", ""))

        if predicted_number is None or not numbers:
            return None, "insufficient_numeric_evidence"

        final_explanation_number = numbers[-1]

        try:
            return (
                float(predicted_number) == float(final_explanation_number),
                "final_explanation_number_match",
            )
        except ValueError:
            return None, "numeric_consistency_parse_failed"

    if dataset == "commonsense_qa":
        predicted_label = extract_choice_label(predicted)
        explicit_match = re.search(
            r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-E])\b",
            explanation,
            flags=re.IGNORECASE,
        )

        if predicted_label is None or explicit_match is None:
            return None, "insufficient_choice_evidence"

        explanation_label = explicit_match.group(1).upper()
        return (
            predicted_label == explanation_label,
            "explicit_choice_label_match",
        )

    if dataset == "strategyqa":
        predicted_yes_no = normalize_yes_no(predicted)

        explanation_text = normalize_text(explanation)
        explanation_yes_no: str | None = None

        if re.search(r"\b(correct answer|answer)\b.*\b(yes|true)\b", explanation_text):
            explanation_yes_no = "yes"
        elif re.search(r"\b(correct answer|answer)\b.*\b(no|false)\b", explanation_text):
            explanation_yes_no = "no"
        elif re.search(r"\btherefore\b.*\b(yes|true)\b", explanation_text):
            explanation_yes_no = "yes"
        elif re.search(r"\btherefore\b.*\b(no|false)\b", explanation_text):
            explanation_yes_no = "no"

        if predicted_yes_no is None or explanation_yes_no is None:
            return None, "insufficient_yes_no_evidence"

        return (
            predicted_yes_no == explanation_yes_no,
            "explicit_yes_no_match",
        )

    return None, "not_implemented"


def main(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            is_correct, answer_metric = score_answer(record)
            consistency, consistency_metric = (
                detect_answer_explanation_consistency(record)
            )

            record["is_correct"] = is_correct
            record["answer_metric"] = answer_metric
            record["answer_explanation_consistent"] = consistency
            record["consistency_metric"] = consistency_metric

            records.append(record)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    total = len(records)
    correct = sum(bool(record["is_correct"]) for record in records)
    parse_success = sum(
        bool(record.get("parse_success"))
        for record in records
    )
    consistency_values = [
        record["answer_explanation_consistent"]
        for record in records
        if record["answer_explanation_consistent"] is not None
    ]
    consistent = sum(bool(value) for value in consistency_values)

    print(f"Records: {total}")
    print(f"Parse success: {parse_success}/{total}")
    print(f"Correct answers: {correct}/{total}")
    print(
        f"Accuracy: {correct / total:.3f}"
        if total
        else "Accuracy: N/A"
    )

    if consistency_values:
        print(
            "Answer-explanation consistency:",
            f"{consistent}/{len(consistency_values)}",
        )
    else:
        print("Answer-explanation consistency: insufficient evidence")

    print(f"Saved scored file to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/generations/qwen_test_outputs.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/scores/qwen_test_scored.jsonl",
    )
    args = parser.parse_args()

    main(args.input, args.output)
