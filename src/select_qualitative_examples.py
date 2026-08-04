from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "qwen_2_5_3b": "Qwen 2.5 3B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "deepseek_r1_qwen_1_5b": "DeepSeek-R1 Distill 1.5B",
}

DATASET_LABELS = {
    "gsm8k": "GSM8K",
    "commonsense_qa": "CommonsenseQA",
    "strategyqa": "StrategyQA",
}

TARGET_CASES = [
    ("Strong correct explanation", "none"),
    ("Answer–explanation mismatch", "answer_explanation_mismatch"),
    ("Unsupported justification", "unsupported_justification"),
    ("Factual hallucination", "factual_hallucination"),
    ("Logical or arithmetic error", "logical_or_arithmetic"),
]


def escape_latex(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def shorten(text: object, max_chars: int = 180) -> str:
    value = "" if pd.isna(text) else str(text)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError("Could not parse all values in is_correct.")
    return parsed.astype(bool)


def pick_one(
    df: pd.DataFrame,
    case_name: str,
    category: str,
    used_indices: set[int],
) -> pd.Series | None:
    available = df.loc[~df.index.isin(used_indices)].copy()

    if category == "none":
        candidates = available[
            (available["failure_category"] == "none")
            & (available["is_correct"])
        ].copy()
        candidates["_rank"] = (
            candidates["faithfulness_score"] * 3
            + candidates["logical_consistency_score"] * 2
            + candidates["completeness_score"]
        )
        candidates = candidates.sort_values("_rank", ascending=False)

    elif category == "logical_or_arithmetic":
        candidates = available[
            available["failure_category"].isin(
                ["logical_error", "arithmetic_error"]
            )
        ].copy()
        candidates["_rank"] = (
            candidates["faithfulness_score"]
            + candidates["logical_consistency_score"]
        )
        candidates = candidates.sort_values("_rank", ascending=True)

    else:
        candidates = available[
            available["failure_category"] == category
        ].copy()
        candidates["_rank"] = (
            candidates["faithfulness_score"]
            + candidates["logical_consistency_score"]
        )
        candidates = candidates.sort_values("_rank", ascending=True)

    if candidates.empty:
        return None

    # Prefer diversity across models and datasets.
    selected_so_far = df.loc[list(used_indices)] if used_indices else pd.DataFrame()
    used_models = set(selected_so_far.get("model_key", []))
    used_datasets = set(selected_so_far.get("dataset", []))

    candidates["_diversity"] = (
        (~candidates["model_key"].isin(used_models)).astype(int) * 2
        + (~candidates["dataset"].isin(used_datasets)).astype(int)
    )

    candidates = candidates.sort_values(
        ["_diversity", "_rank"],
        ascending=[False, category == "none"],
    )

    row = candidates.iloc[0].copy()
    row["case_type"] = case_name
    return row


def build_qualitative_sample(df: pd.DataFrame) -> pd.DataFrame:
    used_indices: set[int] = set()
    selected_rows: list[pd.Series] = []

    for case_name, category in TARGET_CASES:
        row = pick_one(df, case_name, category, used_indices)
        if row is None:
            print(f"[WARNING] No example found for: {case_name}")
            continue

        used_indices.add(int(row.name))
        selected_rows.append(row)

    if not selected_rows:
        raise ValueError("No qualitative examples could be selected.")

    result = pd.DataFrame(selected_rows).copy()
    result["model"] = result["model_key"].map(
        lambda value: MODEL_LABELS.get(value, value)
    )
    result["dataset_label"] = result["dataset"].map(
        lambda value: DATASET_LABELS.get(value, value)
    )

    result["question_excerpt"] = result["question"].map(
        lambda value: shorten(value, 120)
    )
    result["explanation_excerpt"] = result["explanation"].map(
        lambda value: shorten(value, 180)
    )

    preferred_columns = [
        "case_type",
        "model",
        "dataset_label",
        "sample_id",
        "question_excerpt",
        "ground_truth",
        "predicted_answer",
        "explanation_excerpt",
        "is_correct",
        "faithfulness_score",
        "logical_consistency_score",
        "completeness_score",
        "failure_category",
        "annotator_notes",
    ]

    return result[
        [column for column in preferred_columns if column in result.columns]
    ]


def write_latex_table(df: pd.DataFrame, output_path: Path) -> None:
    rows = []

    for _, row in df.iterrows():
        model = escape_latex(row.get("model", ""))
        dataset = escape_latex(row.get("dataset_label", ""))
        case_type = escape_latex(row.get("case_type", ""))
        prediction = escape_latex(row.get("predicted_answer", ""))
        truth = escape_latex(row.get("ground_truth", ""))
        explanation = escape_latex(row.get("explanation_excerpt", ""))
        failure = escape_latex(row.get("failure_category", ""))

        rows.append(
            f"{model} & {dataset} & {case_type} & "
            f"{truth} & {prediction} & "
            f"\\parbox[t]{{4.4cm}}{{{explanation}}} & "
            f"{failure} \\\\"
        )

    latex = r"""\begin{table*}[t]
\centering
\small
\caption{Representative qualitative examples from the reviewed manual-analysis sample. Excerpts are shortened for readability.}
\label{tab:qualitative_examples}
\begin{tabular}{llllllp{2.7cm}}
\toprule
\textbf{Model} &
\textbf{Dataset} &
\textbf{Case} &
\textbf{Gold} &
\textbf{Prediction} &
\textbf{Explanation excerpt} &
\textbf{Primary failure} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    output_path.write_text(latex, encoding="utf-8")


def validate(df: pd.DataFrame) -> None:
    required = {
        "model_key",
        "dataset",
        "question",
        "ground_truth",
        "predicted_answer",
        "explanation",
        "is_correct",
        "faithfulness_score",
        "logical_consistency_score",
        "completeness_score",
        "failure_category",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["is_correct"] = parse_bool(df["is_correct"])

    for column in [
        "faithfulness_score",
        "logical_consistency_score",
        "completeness_score",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            raise ValueError(f"Column {column} contains invalid values.")


def main(input_file: str, output_dir: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate(df)

    selected = build_qualitative_sample(df)

    csv_path = output_path / "qualitative_examples.csv"
    tex_path = output_path / "qualitative_examples_table.tex"

    selected.to_csv(csv_path, index=False)
    write_latex_table(selected, tex_path)

    print("\nQualitative examples selected:")
    display_columns = [
        column for column in [
            "case_type",
            "model",
            "dataset_label",
            "sample_id",
            "failure_category",
            "faithfulness_score",
        ]
        if column in selected.columns
    ]
    print(selected[display_columns].to_string(index=False))

    print("\nGenerated files:")
    print(f"- {csv_path}")
    print(f"- {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/balanced_manual_annotation_sample_final.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qualitative_analysis",
    )

    args = parser.parse_args()
    main(args.input, args.output_dir)
