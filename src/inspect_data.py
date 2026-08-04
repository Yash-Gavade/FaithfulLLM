\
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from src.prompts import build_prompt


def restore_choices(value):
    if pd.isna(value) or value in ("", "None"):
        return None
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)


def main(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python -m src.prepare_data"
        )

    dataframe = pd.read_csv(path)
    print("Shape:", dataframe.shape)
    print("\nColumns:", list(dataframe.columns))
    print("\nCounts:")
    print(dataframe["dataset"].value_counts())
    print("\nMissing values:")
    print(dataframe.isna().sum())

    print("\nExample prompt from each dataset:")
    for dataset_name, group in dataframe.groupby("dataset"):
        row = group.iloc[0].to_dict()
        row["choices"] = restore_choices(row.get("choices"))
        print("\n" + "=" * 80)
        print(dataset_name)
        print("=" * 80)
        print(build_prompt(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="datasets/processed/benchmark_samples.csv",
    )
    args = parser.parse_args()
    main(args.csv)
