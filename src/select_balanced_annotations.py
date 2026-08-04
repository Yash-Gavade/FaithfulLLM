from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def sample_group(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    correct = group[group["is_correct"] == True]
    incorrect = group[group["is_correct"] == False]

    n_correct = min(len(correct), n // 2)
    n_incorrect = min(len(incorrect), n - n_correct)

    parts = []
    if n_correct:
        parts.append(correct.sample(n_correct, random_state=seed))
    if n_incorrect:
        parts.append(incorrect.sample(n_incorrect, random_state=seed + 1))

    chosen = pd.concat(parts) if parts else group.iloc[0:0]
    remaining = n - len(chosen)

    if remaining > 0:
        pool = group.drop(index=chosen.index, errors="ignore")
        if len(pool):
            parts.append(pool.sample(min(remaining, len(pool)), random_state=seed + 2))
            chosen = pd.concat(parts)

    return chosen.sample(frac=1, random_state=seed + 3)


def main(input_csv: str, output_csv: str, n: int, seed: int) -> None:
    df = pd.read_csv(input_csv)
    eligible = df[df["parse_success"] == True].copy()

    selected_parts = []
    for (model_key, dataset), group in eligible.groupby(["model_key", "dataset"]):
        part = sample_group(group, n, seed)
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True)

    for column in [
        "faithfulness_score",
        "logical_consistency_score",
        "completeness_score",
        "failure_category",
        "annotator_notes",
    ]:
        selected[column] = ""

    preferred = [
        "sample_id", "model_key", "model_id", "dataset", "question",
        "ground_truth", "predicted_answer", "is_correct", "parse_success",
        "explanation", "answer_explanation_consistent",
        "faithfulness_score", "logical_consistency_score",
        "completeness_score", "failure_category", "annotator_notes",
    ]
    selected = selected[[c for c in preferred if c in selected.columns]]

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Saved: {out}")
    print(f"Rows: {len(selected)}")
    print(selected.groupby(["model_key", "dataset"]).size().to_string())
    print("\nCorrect/incorrect balance:")
    print(selected.groupby(["model_key", "dataset", "is_correct"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/scores/all_models_results.csv")
    parser.add_argument("--output", default="evaluation/balanced_manual_annotation_sample.csv")
    parser.add_argument("--samples-per-model-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input, args.output, args.samples_per_model_dataset, args.seed)
