from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main(input_dir: str, model_key: str, output_csv: str):
    files = sorted(Path(input_dir).glob(f"{model_key}_*_scored.jsonl"))
    if not files:
        raise FileNotFoundError(f"No scored files found for {model_key}")

    rows = []
    for file in files:
        part = read_jsonl(file)
        print(f"[LOAD] {file.name}: {len(part)}")
        rows.extend(part)

    df = pd.DataFrame(rows)
    if df.duplicated(subset=["sample_id", "model_key"]).any():
        raise ValueError("Duplicate sample/model rows found.")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nRows: {len(df)}")
    print(f"Saved: {out}")
    print("\nCounts:")
    print(df["dataset"].value_counts().to_string())
    print("\nAccuracy:")
    print((df.groupby("dataset")["is_correct"].mean()*100).round(2).astype(str)+"%")
    print("\nParse success:")
    print((df.groupby("dataset")["parse_success"].mean()*100).round(2).astype(str)+"%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="outputs/scores")
    p.add_argument("--model", default="qwen_2_5_3b")
    p.add_argument("--output", default="outputs/scores/qwen_2_5_3b_all_results.csv")
    a = p.parse_args()
    main(a.input_dir, a.model, a.output)
