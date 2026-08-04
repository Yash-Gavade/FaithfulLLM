from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCORE_COLUMNS = [
    "faithfulness_score",
    "logical_consistency_score",
    "completeness_score",
]

MODEL_LABELS = {
    "deepseek_r1_qwen_1_5b": "DeepSeek-R1 Distill 1.5B",
    "llama_3_2_3b": "Llama 3.2 3B",
    "qwen_2_5_3b": "Qwen 2.5 3B",
}

DATASET_LABELS = {
    "commonsense_qa": "CommonsenseQA",
    "gsm8k": "GSM8K",
    "strategyqa": "StrategyQA",
}

FAILURE_LABELS = {
    "answer_explanation_mismatch": "Answer–explanation mismatch",
    "arithmetic_error": "Arithmetic error",
    "contradictory_explanation": "Contradictory explanation",
    "factual_hallucination": "Factual hallucination",
    "incomplete_reasoning": "Incomplete reasoning",
    "instruction_following_failure": "Instruction-following failure",
    "generation_or_instruction_failure": "Generation/instruction failure",
    "logical_error": "Logical error",
    "none": "No primary failure",
    "other": "Other",
    "unsupported_justification": "Unsupported justification",
}


def validate_annotations(df: pd.DataFrame) -> None:
    required = {
        "model_key",
        "dataset",
        "is_correct",
        "failure_category",
        *SCORE_COLUMNS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            bad_rows = df.index[df[column].isna()].tolist()
            raise ValueError(
                f"Column '{column}' contains missing/non-numeric values "
                f"at rows: {bad_rows[:10]}"
            )

        invalid = ~df[column].between(0, 3)
        if invalid.any():
            bad_values = df.loc[invalid, column].tolist()
            raise ValueError(
                f"Column '{column}' contains values outside 0–3: "
                f"{bad_values[:10]}"
            )

    if df["failure_category"].fillna("").eq("").any():
        raise ValueError("Some rows have an empty failure_category.")


def save_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    by_model = (
        df.groupby("model_key")[SCORE_COLUMNS]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    by_model.to_csv(output_dir / "manual_scores_by_model.csv")

    by_model_dataset = (
        df.groupby(["model_key", "dataset"])[SCORE_COLUMNS]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    by_model_dataset.to_csv(
        output_dir / "manual_scores_by_model_dataset.csv"
    )

    correct_vs_incorrect = (
        df.groupby(["model_key", "is_correct"])[SCORE_COLUMNS]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    correct_vs_incorrect.to_csv(
        output_dir / "manual_scores_correct_vs_incorrect.csv"
    )

    failure_counts = (
        df.groupby(["model_key", "failure_category"])
        .size()
        .rename("count")
        .reset_index()
    )
    failure_counts.to_csv(
        output_dir / "failure_category_counts.csv",
        index=False,
    )

    failure_percentages = failure_counts.copy()
    totals = failure_percentages.groupby("model_key")["count"].transform("sum")
    failure_percentages["percentage"] = (
        failure_percentages["count"] / totals * 100
    ).round(2)
    failure_percentages.to_csv(
        output_dir / "failure_category_percentages.csv",
        index=False,
    )


def plot_mean_scores_by_model(df: pd.DataFrame, output_dir: Path) -> None:
    means = df.groupby("model_key")[SCORE_COLUMNS].mean()
    means.index = [MODEL_LABELS.get(model, model) for model in means.index]

    ax = means.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Mean Manual Explanation Scores by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Score (0–3)")
    ax.set_ylim(0, 3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(
        ["Faithfulness", "Logical consistency", "Completeness"],
        title="Metric",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "mean_scores_by_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_faithfulness_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(
        index="model_key",
        columns="dataset",
        values="faithfulness_score",
        aggfunc="mean",
    )

    pivot.index = [MODEL_LABELS.get(model, model) for model in pivot.index]
    pivot.columns = [
        DATASET_LABELS.get(dataset, dataset)
        for dataset in pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    image = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Explanation-Support Faithfulness by Model and Dataset")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax, label="Mean score (0–3)")
    plt.tight_layout()
    plt.savefig(
        output_dir / "faithfulness_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_correct_vs_incorrect(df: pd.DataFrame, output_dir: Path) -> None:
    pivot = df.pivot_table(
        index="model_key",
        columns="is_correct",
        values="faithfulness_score",
        aggfunc="mean",
    ).rename(columns={False: "Incorrect", True: "Correct"})

    pivot.index = [MODEL_LABELS.get(model, model) for model in pivot.index]

    ax = pivot.plot(kind="bar", figsize=(9, 5.5))
    ax.set_title("Explanation-Support Faithfulness for Correct vs. Incorrect Answers")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Faithfulness Score (0–3)")
    ax.set_ylim(0, 3)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Answer correctness")
    plt.tight_layout()
    plt.savefig(
        output_dir / "faithfulness_correct_vs_incorrect.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_failure_categories(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()

    plot_df["model_key"] = plot_df["model_key"].map(
        lambda value: MODEL_LABELS.get(value, value)
    )
    plot_df["failure_category"] = plot_df["failure_category"].map(
        lambda value: FAILURE_LABELS.get(value, value)
    )

    counts = (
        plot_df.groupby(["model_key", "failure_category"])
        .size()
        .unstack(fill_value=0)
    )

    ax = counts.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )
    ax.set_title("Manual Explanation Failure Categories by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of annotated responses\n(n = 30 per model)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(
        title="Failure category",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "failure_categories_by_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def print_console_summary(df: pd.DataFrame) -> None:
    print("\nMean scores by model:")
    print(
        df.groupby("model_key")[SCORE_COLUMNS]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nMean faithfulness by model and dataset:")
    print(
        df.pivot_table(
            index="model_key",
            columns="dataset",
            values="faithfulness_score",
            aggfunc="mean",
        )
        .round(2)
        .to_string()
    )

    print("\nFaithfulness for correct vs incorrect answers:")
    print(
        df.pivot_table(
            index="model_key",
            columns="is_correct",
            values="faithfulness_score",
            aggfunc="mean",
        )
        .rename(columns={False: "Incorrect", True: "Correct"})
        .round(2)
        .to_string()
    )

    print("\nFailure-category counts:")
    print(
        df.groupby(["model_key", "failure_category"])
        .size()
        .to_string()
    )


def main(input_file: str, output_dir: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_annotations(df)

    save_summary_tables(df, output_path)
    plot_mean_scores_by_model(df, output_path)
    plot_faithfulness_heatmap(df, output_path)
    plot_correct_vs_incorrect(df, output_path)
    plot_failure_categories(df, output_path)
    print_console_summary(df)

    print(f"\nAnalysis completed. Outputs saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/balanced_manual_annotation_sample_annotated.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manual_analysis",
    )
    args = parser.parse_args()
    main(args.input, args.output_dir)
