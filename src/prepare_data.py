from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from src.common import load_config, set_seed, write_jsonl


STANDARD_COLUMNS = [
    "sample_id",
    "dataset",
    "task_type",
    "question",
    "choices",
    "ground_truth",
    "source_index",
]


def normalize_gsm8k(row: dict[str, Any], index: int) -> dict[str, Any]:
    answer_text = str(row["answer"])
    final_answer = answer_text.split("####")[-1].strip().replace(",", "")
    return {
        "sample_id": f"gsm8k_{index:04d}",
        "dataset": "gsm8k",
        "task_type": "mathematical_reasoning",
        "question": str(row["question"]).strip(),
        "choices": None,
        "ground_truth": final_answer,
        "source_index": index,
    }


def normalize_commonsense_qa(row: dict[str, Any], index: int) -> dict[str, Any]:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    choices = [
        {"label": str(label), "text": str(text)}
        for label, text in zip(labels, texts)
    ]

    answer_key = str(row["answerKey"]).strip()
    matching = [choice for choice in choices if choice["label"] == answer_key]
    answer_text = matching[0]["text"] if matching else ""

    return {
        "sample_id": f"commonsense_qa_{index:04d}",
        "dataset": "commonsense_qa",
        "task_type": "commonsense_reasoning",
        "question": str(row["question"]).strip(),
        "choices": choices,
        "ground_truth": f"{answer_key}. {answer_text}".strip(),
        "source_index": index,
    }


def normalize_strategyqa(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = row.get("question") or row.get("input")
    answer = row.get("answer")

    if question is None or answer is None:
        raise ValueError(
            "StrategyQA schema was not recognized. "
            f"Available columns: {list(row.keys())}"
        )

    if isinstance(answer, bool):
        answer = "yes" if answer else "no"

    return {
        "sample_id": f"strategyqa_{index:04d}",
        "dataset": "strategyqa",
        "task_type": "multi_hop_reasoning",
        "question": str(question).strip(),
        "choices": [
            {"label": "A", "text": "Yes"},
            {"label": "B", "text": "No"},
        ],
        "ground_truth": str(answer).strip().lower(),
        "source_index": index,
    }


NORMALIZERS = {
    "gsm8k": normalize_gsm8k,
    "commonsense_qa": normalize_commonsense_qa,
    "strategyqa": normalize_strategyqa,
}


def load_split(dataset_name: str, settings: dict[str, Any]) -> Dataset:
    path = settings["path"]
    config = settings.get("config")
    split = settings["split"]

    try:
        if config:
            return load_dataset(path, config, split=split)
        return load_dataset(path, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load dataset '{dataset_name}' from '{path}', "
            f"config={config!r}, split='{split}'. Original error: {exc}"
        ) from exc


def sample_dataset(
    dataset_name: str,
    settings: dict[str, Any],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset = load_split(dataset_name, settings)

    if len(dataset) < sample_size:
        raise ValueError(
            f"{dataset_name} contains only {len(dataset)} rows, "
            f"but {sample_size} were requested."
        )

    # Add the original row number before shuffling so that source_index remains stable.
    dataset = dataset.add_column("source_index", list(range(len(dataset))))
    selected = dataset.shuffle(seed=seed).select(range(sample_size))

    normalizer = NORMALIZERS[dataset_name]
    records: list[dict[str, Any]] = []

    for row in selected:
        row_dict = dict(row)
        source_index = int(row_dict.pop("source_index"))
        records.append(normalizer(row_dict, source_index))

    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()

    for record in records:
        missing = [
            column for column in STANDARD_COLUMNS
            if column not in record
        ]
        if missing:
            raise ValueError(f"Missing columns {missing} in record: {record}")

        if record["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        seen_ids.add(record["sample_id"])

        if not str(record["question"]).strip():
            raise ValueError(f"Empty question: {record['sample_id']}")

        if not str(record["ground_truth"]).strip():
            raise ValueError(f"Empty ground truth: {record['sample_id']}")


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    sample_size = int(config["data"]["samples_per_dataset"])
    set_seed(seed)

    all_records: list[dict[str, Any]] = []

    for dataset_name, settings in config["datasets"].items():
        if not settings.get("enabled", False):
            print(f"[SKIP] {dataset_name} is disabled in config.yaml")
            continue

        print(f"[LOAD] {dataset_name}")
        records = sample_dataset(
            dataset_name=dataset_name,
            settings=settings,
            sample_size=sample_size,
            seed=seed,
        )
        all_records.extend(records)
        print(f"[OK]   {dataset_name}: {len(records)} samples")

    validate_records(all_records)

    csv_path = Path(config["data"]["output_csv"])
    jsonl_path = Path(config["data"]["output_jsonl"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(all_records, columns=STANDARD_COLUMNS)
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    write_jsonl(all_records, jsonl_path)

    print("\nDataset preparation completed.")
    print(f"CSV:   {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print("\nCounts by dataset:")
    print(dataframe["dataset"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare FaithfulLLM benchmark samples."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
