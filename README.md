# FaithfulLLM

Empirical evaluation of self-generated explanations from open-source large
language models across mathematical, commonsense, and multi-hop reasoning tasks.

## Phase 1: prepare benchmark data

### 1. Clone the repository

```bash
git clone https://github.com/Yash-Gavade/FaithfulLLM.git
cd FaithfulLLM
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the first two datasets

```bash
python -m src.prepare_data
```

Generated files:

- `datasets/processed/benchmark_samples.csv`
- `datasets/processed/benchmark_samples.jsonl`

The initial configuration enables GSM8K and CommonsenseQA. StrategyQA is kept
disabled until its exact reproducible source and schema are verified.

### 5. Inspect the prepared data and prompts

```bash
python -m src.inspect_data
```

## Standard sample schema

| Field | Meaning |
|---|---|
| `sample_id` | Stable local identifier |
| `dataset` | Dataset name |
| `task_type` | Reasoning category |
| `question` | Input question |
| `choices` | Multiple-choice options or null |
| `ground_truth` | Gold answer |
| `source_index` | Original sampled row index |

## Output format required from every model

```text
FINAL_ANSWER: <answer>
EXPLANATION: <concise justification>
```

## Current repository structure

```text
FaithfulLLM/
├── config.yaml
├── requirements.txt
├── datasets/
│   ├── raw/
│   └── processed/
├── src/
│   ├── common.py
│   ├── prepare_data.py
│   ├── inspect_data.py
│   └── prompts.py
├── outputs/
├── evaluation/
├── metrics/
├── notebooks/
├── figures/
├── experiments/
└── paper/
```

## Reproducibility

The sampling seed is fixed to `42` in `config.yaml`. Do not change the sampled
benchmark after model inference begins.
