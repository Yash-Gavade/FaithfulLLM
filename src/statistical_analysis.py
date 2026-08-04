from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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

MODEL_ORDER = [
    "qwen_2_5_3b",
    "llama_3_2_3b",
    "deepseek_r1_qwen_1_5b",
]

DATASET_ORDER = [
    "gsm8k",
    "commonsense_qa",
    "strategyqa",
]


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to real booleans."""
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

    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if converted.isna().any():
        bad_values = sorted(series[converted.isna()].astype(str).unique())
        raise ValueError(
            "Could not interpret some boolean values: "
            f"{bad_values[:10]}"
        )

    return converted.astype(bool)


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return math.nan, math.nan

    p = successes / total
    denominator = 1 + (z**2 / total)
    centre = p + (z**2 / (2 * total))
    adjustment = z * math.sqrt(
        (p * (1 - p) / total)
        + (z**2 / (4 * total**2))
    )

    lower = (centre - adjustment) / denominator
    upper = (centre + adjustment) / denominator
    return lower, upper


def cliffs_delta(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Cliff's delta for group_a versus group_b.

    Positive values mean group_a tends to have larger scores.
    """
    greater = 0
    smaller = 0

    for value_a in group_a:
        greater += np.sum(value_a > group_b)
        smaller += np.sum(value_a < group_b)

    return float((greater - smaller) / (len(group_a) * len(group_b)))


def cliffs_delta_magnitude(delta: float) -> str:
    """Interpret absolute Cliff's delta using common thresholds."""
    value = abs(delta)

    if value < 0.147:
        return "negligible"
    if value < 0.330:
        return "small"
    if value < 0.474:
        return "medium"
    return "large"


def bootstrap_mean_difference(
    group_a: np.ndarray,
    group_b: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """
    Bootstrap the mean difference group_a - group_b.

    Returns observed difference and percentile 95% CI.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(group_a) - np.mean(group_b))

    differences = np.empty(iterations, dtype=float)

    for index in range(iterations):
        sample_a = rng.choice(group_a, size=len(group_a), replace=True)
        sample_b = rng.choice(group_b, size=len(group_b), replace=True)
        differences[index] = np.mean(sample_a) - np.mean(sample_b)

    lower, upper = np.percentile(differences, [2.5, 97.5])
    return observed, float(lower), float(upper)


def validate_results(df: pd.DataFrame) -> None:
    required = {"model_key", "dataset", "is_correct"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Results file is missing columns: {sorted(missing)}"
        )

    df["is_correct"] = parse_bool_series(df["is_correct"])

    duplicated = df.duplicated(subset=["model_key", "sample_id"])
    if "sample_id" in df.columns and duplicated.any():
        count = int(duplicated.sum())
        raise ValueError(
            f"Results contain {count} duplicate model/sample rows."
        )


def validate_annotations(df: pd.DataFrame) -> None:
    required = {
        "model_key",
        "dataset",
        "is_correct",
        "faithfulness_score",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Annotation file is missing columns: {sorted(missing)}"
        )

    df["is_correct"] = parse_bool_series(df["is_correct"])
    df["faithfulness_score"] = pd.to_numeric(
        df["faithfulness_score"],
        errors="coerce",
    )

    if df["faithfulness_score"].isna().any():
        raise ValueError(
            "Some faithfulness scores are missing or non-numeric."
        )

    invalid = ~df["faithfulness_score"].between(0, 3)
    if invalid.any():
        raise ValueError(
            "Faithfulness scores must be between 0 and 3."
        )


def create_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    model_keys = [
        key for key in MODEL_ORDER
        if key in set(df["model_key"])
    ]

    for model_key in model_keys:
        model_df = df[df["model_key"] == model_key]

        for dataset in DATASET_ORDER:
            subset = model_df[model_df["dataset"] == dataset]
            if subset.empty:
                continue

            correct = int(subset["is_correct"].sum())
            total = int(len(subset))
            lower, upper = wilson_interval(correct, total)

            rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS.get(model_key, model_key),
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(
                        dataset,
                        dataset,
                    ),
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total,
                    "accuracy_percent": correct / total * 100,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci95_lower_percent": lower * 100,
                    "ci95_upper_percent": upper * 100,
                }
            )

        correct = int(model_df["is_correct"].sum())
        total = int(len(model_df))
        lower, upper = wilson_interval(correct, total)

        rows.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS.get(model_key, model_key),
                "dataset": "overall",
                "dataset_label": "Overall",
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "accuracy_percent": correct / total * 100,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci95_lower_percent": lower * 100,
                "ci95_upper_percent": upper * 100,
            }
        )

    return pd.DataFrame(rows)


def compare_faithfulness(
    annotations: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    scopes: list[tuple[str, pd.DataFrame]] = [
        ("All models", annotations)
    ]

    for model_key in MODEL_ORDER:
        subset = annotations[
            annotations["model_key"] == model_key
        ]
        if not subset.empty:
            scopes.append(
                (
                    MODEL_LABELS.get(model_key, model_key),
                    subset,
                )
            )

    for scope_name, subset in scopes:
        correct = subset.loc[
            subset["is_correct"],
            "faithfulness_score",
        ].to_numpy(dtype=float)

        incorrect = subset.loc[
            ~subset["is_correct"],
            "faithfulness_score",
        ].to_numpy(dtype=float)

        if len(correct) == 0 or len(incorrect) == 0:
            continue

        test = mannwhitneyu(
            correct,
            incorrect,
            alternative="two-sided",
            method="auto",
        )

        delta = cliffs_delta(correct, incorrect)

        difference, lower, upper = bootstrap_mean_difference(
            correct,
            incorrect,
            iterations=bootstrap_iterations,
            seed=seed,
        )

        rows.append(
            {
                "scope": scope_name,
                "correct_n": len(correct),
                "incorrect_n": len(incorrect),
                "correct_mean": np.mean(correct),
                "incorrect_mean": np.mean(incorrect),
                "correct_median": np.median(correct),
                "incorrect_median": np.median(incorrect),
                "mean_difference_correct_minus_incorrect": difference,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "mann_whitney_u": float(test.statistic),
                "p_value_two_sided": float(test.pvalue),
                "cliffs_delta": delta,
                "effect_magnitude": cliffs_delta_magnitude(delta),
            }
        )

    return pd.DataFrame(rows)


def create_accuracy_figure(
    accuracy_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_df = accuracy_table[
        accuracy_table["dataset"] != "overall"
    ].copy()

    pivot = figure_df.pivot(
        index="model",
        columns="dataset_label",
        values="accuracy_percent",
    )

    lower = figure_df.pivot(
        index="model",
        columns="dataset_label",
        values="ci95_lower_percent",
    )

    upper = figure_df.pivot(
        index="model",
        columns="dataset_label",
        values="ci95_upper_percent",
    )

    desired_models = [
        MODEL_LABELS[key]
        for key in MODEL_ORDER
        if MODEL_LABELS[key] in pivot.index
    ]
    desired_datasets = [
        DATASET_LABELS[key]
        for key in DATASET_ORDER
        if DATASET_LABELS[key] in pivot.columns
    ]

    pivot = pivot.reindex(
        index=desired_models,
        columns=desired_datasets,
    )
    lower = lower.reindex(
        index=desired_models,
        columns=desired_datasets,
    )
    upper = upper.reindex(
        index=desired_models,
        columns=desired_datasets,
    )

    yerr_lower = pivot - lower
    yerr_upper = upper - pivot
    yerr = np.stack(
        [yerr_lower.to_numpy(), yerr_upper.to_numpy()]
    )
    # Pandas expects asymmetric bar errors as:
    # (number_of_series, 2, number_of_categories).
    yerr = np.transpose(yerr, (2, 0, 1))

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
        yerr=yerr,
        capsize=4,
    )

    ax.set_title(
        "Answer Accuracy with 95% Wilson Confidence Intervals"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Dataset")

    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_with_confidence_intervals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def write_summary(
    accuracy_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("STATISTICAL ANALYSIS SUMMARY")
    lines.append("=" * 32)
    lines.append("")
    lines.append("Accuracy with 95% Wilson confidence intervals")
    lines.append("-" * 49)

    for _, row in accuracy_table.iterrows():
        lines.append(
            f"{row['model']} | {row['dataset_label']}: "
            f"{int(row['correct'])}/{int(row['total'])} = "
            f"{row['accuracy_percent']:.2f}% "
            f"[95% CI: {row['ci95_lower_percent']:.2f}%, "
            f"{row['ci95_upper_percent']:.2f}%]"
        )

    lines.append("")
    lines.append(
        "Faithfulness: correct versus incorrect answers"
    )
    lines.append("-" * 50)

    for _, row in comparison_table.iterrows():
        lines.append(
            f"{row['scope']}: correct mean={row['correct_mean']:.2f}, "
            f"incorrect mean={row['incorrect_mean']:.2f}, "
            f"mean difference={row['mean_difference_correct_minus_incorrect']:.2f} "
            f"[bootstrap 95% CI: {row['bootstrap_ci95_lower']:.2f}, "
            f"{row['bootstrap_ci95_upper']:.2f}], "
            f"Mann-Whitney U={row['mann_whitney_u']:.1f}, "
            f"p={format_p_value(row['p_value_two_sided'])}, "
            f"Cliff's delta={row['cliffs_delta']:.3f} "
            f"({row['effect_magnitude']} effect)."
        )

    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 14)
    lines.append(
        "A positive mean difference and positive Cliff's delta mean "
        "that explanations attached to correct answers tend to receive "
        "higher faithfulness scores than explanations attached to "
        "incorrect answers."
    )
    lines.append(
        "The Mann-Whitney U test evaluates whether the score "
        "distributions differ. The bootstrap interval quantifies "
        "uncertainty in the mean-score difference."
    )
    lines.append(
        "These analyses concern explanation-support faithfulness and "
        "do not establish causal access to the model's hidden internal "
        "reasoning process."
    )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(
    results_file: str,
    annotations_file: str,
    output_dir: str,
    bootstrap_iterations: int,
    seed: int,
) -> None:
    results_path = Path(results_file)
    annotations_path = Path(annotations_file)
    output_path = Path(output_dir)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotations_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(results_path)
    annotations = pd.read_csv(annotations_path)

    validate_results(results)
    validate_annotations(annotations)

    accuracy_table = create_accuracy_table(results)
    comparison_table = compare_faithfulness(
        annotations,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    accuracy_table.round(6).to_csv(
        output_path / "accuracy_confidence_intervals.csv",
        index=False,
    )

    comparison_table.round(6).to_csv(
        output_path / "faithfulness_correct_vs_incorrect.csv",
        index=False,
    )

    create_accuracy_figure(
        accuracy_table,
        output_path,
    )

    write_summary(
        accuracy_table,
        comparison_table,
        output_path / "statistical_summary.txt",
    )

    print("\nStatistical analysis completed.")
    print(f"Results file: {results_path}")
    print(f"Annotation file: {annotations_path}")
    print(f"Output folder: {output_path}")
    print("\nGenerated files:")

    for file_path in sorted(output_path.iterdir()):
        print(f"- {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute accuracy confidence intervals and compare "
            "faithfulness scores for correct versus incorrect answers."
        )
    )
    parser.add_argument(
        "--results",
        default="outputs/scores/all_models_results.csv",
    )
    parser.add_argument(
        "--annotations",
        default=(
            "evaluation/"
            "balanced_manual_annotation_sample_final.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/statistical_analysis",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(
        results_file=args.results,
        annotations_file=args.annotations,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
