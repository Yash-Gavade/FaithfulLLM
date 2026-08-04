from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(path_string: str) -> None:
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    print(f"Records: {len(records)}")
    print(
        "Parse success:",
        sum(bool(record.get("parse_success")) for record in records),
        "/",
        len(records),
    )
    print(
        "Errors:",
        sum(bool(record.get("error")) for record in records),
    )

    for index, record in enumerate(records, start=1):
        print("\n" + "=" * 80)
        print(f"RECORD {index}")
        print("=" * 80)
        print("Sample ID:", record["sample_id"])
        print("Dataset:", record["dataset"])
        print("Ground truth:", record["ground_truth"])
        print("Predicted answer:", record["predicted_answer"])
        print("Parse success:", record["parse_success"])
        print("Generation seconds:", record["generation_seconds"])
        print("\nExplanation:")
        print(record["explanation"])
        if record.get("error"):
            print("\nError:")
            print(record["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="outputs/generations/qwen_test_outputs.jsonl",
    )
    args = parser.parse_args()
    main(args.file)
