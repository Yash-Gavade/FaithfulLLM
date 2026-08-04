from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
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


def create_accuracy_chart(results_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(results_csv)

    accuracy = (
        df.groupby(["model_key", "dataset"])["is_correct"]
        .mean()
        .mul(100)
        .unstack()
    )

    model_order = [
        "qwen_2_5_3b",
        "llama_3_2_3b",
        "deepseek_r1_qwen_1_5b",
    ]
    dataset_order = [
        "gsm8k",
        "commonsense_qa",
        "strategyqa",
    ]

    accuracy = accuracy.reindex(
        index=[m for m in model_order if m in accuracy.index],
        columns=[d for d in dataset_order if d in accuracy.columns],
    )

    accuracy.index = [
        MODEL_LABELS.get(model, model)
        for model in accuracy.index
    ]
    accuracy.columns = [
        DATASET_LABELS.get(dataset, dataset)
        for dataset in accuracy.columns
    ]

    ax = accuracy.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Answer Accuracy by Model and Dataset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Dataset")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_by_model_dataset.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_pipeline_diagram(output_dir: Path) -> None:
    steps_top = [
        "Dataset Selection",
        "Fixed Sampling\n(100 per dataset)",
        "Standardized Prompt",
        "LLM Inference",
    ]
    steps_bottom = [
        "Comparative Analysis",
        "Balanced Manual Annotation\n(90 responses)",
        "Automatic Evaluation",
        "Answer + Explanation Parsing",
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.12, 0.37, 0.62, 0.87]
    top_y = 0.68
    bottom_y = 0.28

    for i, (step, x) in enumerate(zip(steps_top, x_positions)):
        ax.text(
            x,
            top_y,
            step,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "linewidth": 1.5},
        )
        if i < len(steps_top) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.07, top_y),
                xytext=(x + 0.07, top_y),
                arrowprops={"arrowstyle": "->", "linewidth": 1.5},
            )

    ax.annotate(
        "",
        xy=(x_positions[-1], bottom_y + 0.08),
        xytext=(x_positions[-1], top_y - 0.08),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
    )

    for i, (step, x) in enumerate(zip(steps_bottom, x_positions)):
        ax.text(
            x,
            bottom_y,
            step,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "linewidth": 1.5},
        )

    for i in range(len(steps_bottom) - 1, 0, -1):
        ax.annotate(
            "",
            xy=(x_positions[i - 1] + 0.07, bottom_y),
            xytext=(x_positions[i] - 0.07, bottom_y),
            arrowprops={"arrowstyle": "->", "linewidth": 1.5},
        )

    ax.set_title("Experimental Pipeline", fontsize=16, pad=18)

    plt.tight_layout()
    plt.savefig(
        output_dir / "experimental_pipeline.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def copy_existing_figures(
    source_dir: Path,
    output_dir: Path,
) -> None:
    figure_names = [
        "faithfulness_heatmap.png",
        "faithfulness_correct_vs_incorrect.png",
        "failure_categories_by_model.png",
        "mean_scores_by_model.png",
    ]

    for name in figure_names:
        source = source_dir / name
        destination = output_dir / name

        if source.exists():
            shutil.copy2(source, destination)
            print(f"[COPIED] {source} -> {destination}")
        else:
            print(f"[WARNING] Missing: {source}")


def main(
    results_csv: str,
    manual_figure_dir: str,
    output_dir: str,
) -> None:
    results_path = Path(results_csv)
    manual_path = Path(manual_figure_dir)
    output_path = Path(output_dir)

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    output_path.mkdir(parents=True, exist_ok=True)

    create_accuracy_chart(results_path, output_path)
    create_pipeline_diagram(output_path)
    copy_existing_figures(manual_path, output_path)

    print("\nPaper figures are ready:")
    for figure in sorted(output_path.glob("*.png")):
        print(f"- {figure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/scores/all_models_results.csv",
    )
    parser.add_argument(
        "--manual-figures",
        default="outputs/manual_analysis",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures",
    )
    args = parser.parse_args()

    main(
        results_csv=args.results,
        manual_figure_dir=args.manual_figures,
        output_dir=args.output_dir,
    )
