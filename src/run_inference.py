from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.common import load_config
from src.parse_outputs import parse_model_output
from src.prompts import SYSTEM_PROMPT, build_prompt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def build_quantization_config(model_config: dict[str, Any]):
    if not model_config.get("load_in_4bit", False):
        return None

    compute_dtype_name = model_config.get(
        "bnb_4bit_compute_dtype",
        "float16",
    )
    compute_dtype = getattr(torch, compute_dtype_name)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.get(
            "bnb_4bit_quant_type",
            "nf4",
        ),
        bnb_4bit_use_double_quant=bool(
            model_config.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    input_length = model_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][input_length:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def main(
    config_path: str,
    model_key: str,
    dataset_name: str | None,
    limit: int | None,
    overwrite: bool,
) -> None:
    config = load_config(config_path)
    model_config = config["models"][model_key]
    inference_config = config["inference"]

    model_id = model_config["model_id"]
    input_path = Path(inference_config["input_file"])

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    samples = read_jsonl(input_path)

    if dataset_name:
        samples = [
            sample
            for sample in samples
            if sample["dataset"] == dataset_name
        ]

    requested_limit = (
        limit
        if limit is not None
        else int(inference_config.get("limit", 5))
    )
    samples = samples[:requested_limit]

    if not samples:
        raise ValueError(
            f"No samples available for dataset={dataset_name!r}"
        )

    dataset_suffix = dataset_name if dataset_name else "all"
    output_path = Path(
        f"outputs/generations/"
        f"{model_key}_{dataset_suffix}_outputs.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_completed_ids(output_path)
    remaining = [
        sample
        for sample in samples
        if sample["sample_id"] not in completed_ids
    ]

    quantization_config = build_quantization_config(model_config)
    use_4bit = quantization_config is not None

    print(f"Model:     {model_id}")
    print(f"Dataset:   {dataset_suffix}")
    print(f"CUDA:      {torch.cuda.is_available()}")
    print(f"4-bit:     {use_4bit}")
    print(f"Limit:     {requested_limit}")
    print(f"Completed: {len(completed_ids)}")
    print(f"Remaining: {len(remaining)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(
            model_config.get("trust_remote_code", False)
        ),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(
            model_config.get("trust_remote_code", False)
        ),
    }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.eval()

    max_new_tokens = int(
        model_config.get(
            "max_new_tokens",
            inference_config.get("max_new_tokens", 256),
        )
    )

    with output_path.open("a", encoding="utf-8") as output_file:
        for sample in tqdm(remaining, desc="Generating"):
            prompt = build_prompt(sample)
            started = time.perf_counter()

            try:
                raw_output = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                predicted_answer, explanation, parse_success = (
                    parse_model_output(raw_output)
                )
                error = None
            except Exception as exc:
                raw_output = ""
                predicted_answer = None
                explanation = None
                parse_success = False
                error = f"{type(exc).__name__}: {exc}"

            result = {
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "task_type": sample["task_type"],
                "model_key": model_key,
                "model_id": model_id,
                "quantization": "bnb_nf4_4bit" if use_4bit else "none",
                "question": sample["question"],
                "choices": sample.get("choices"),
                "ground_truth": sample["ground_truth"],
                "prompt": prompt,
                "raw_output": raw_output,
                "predicted_answer": predicted_answer,
                "explanation": explanation,
                "parse_success": parse_success,
                "generation_seconds": round(
                    time.perf_counter() - started,
                    4,
                ),
                "error": error,
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

    print("\nInference completed.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=[
            "gsm8k",
            "commonsense_qa",
            "strategyqa",
        ],
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        config_path=args.config,
        model_key=args.model,
        dataset_name=args.dataset,
        limit=args.limit,
        overwrite=args.overwrite,
    )
