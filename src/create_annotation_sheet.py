from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    keep = [c for c in [
        "sample_id","dataset","model_key","model_id","question",
        "ground_truth","predicted_answer","is_correct","explanation",
        "answer_explanation_consistent"
    ] if c in df.columns]
    out_df = df[keep].copy()
    for col in [
        "faithfulness_score","logical_consistency_score",
        "completeness_score","failure_category","annotator_notes"
    ]:
        out_df[col] = ""
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Created: {out}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/scores/qwen_2_5_3b_all_results.csv")
    p.add_argument("--output", default="evaluation/qwen_2_5_3b_manual_annotations.csv")
    a = p.parse_args()
    main(a.input, a.output)
