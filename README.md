# LLM Fine-Tuning Harness — Natural Language to Chart Specs

A focused, demo-oriented toolkit for fine-tuning a small open-weights LLM to translate natural-language questions into [Vega-Lite](https://vega.github.io/vega-lite/) chart specifications, paired with a structured evaluation suite and a Streamlit demo.

The project covers the full lifecycle — **curate → fine-tune → evaluate → demo** — and runs end-to-end on free infrastructure: free-tier Colab GPU for training, local CPU or Apple Silicon (MPS) for evaluation and the demo, open-licensed models and datasets throughout.

## Why this project

Most fine-tuning examples stop at "the loss curve went down." This harness emphasises what comes after: a decoupled eval suite that scores any model — base, fine-tuned, or an alternative — against the same task, so the lift from fine-tuning is measurable rather than asserted.

The chosen task — natural-language → Vega-Lite — is a clean structured-output target: every prediction must parse, validate against a published JSON schema, and reference real columns from the supplied table. That makes correctness, hallucination, and tool-use accuracy all computable automatically.

## Pipeline

The project is split into four independently runnable stages:

| Stage | Package | Role |
|-------|---------|------|
| 1. Curate | `curate/` | Stream [nvBench](https://github.com/TsinghuaDatabaseGroup/nvBench), normalise gold answers to Vega-Lite v5, validate against the official JSON schema, and write deterministic train/test JSONL splits. |
| 2. Fine-tune | `train/` + `notebooks/train_colab.ipynb` | LoRA/QLoRA supervised fine-tuning of [Gemma 2 2B Instruct](https://huggingface.co/google/gemma-2-2b-it) on the free Colab T4 tier, using `peft` + `trl` + `bitsandbytes`. |
| 3. Evaluate | `evals/` | Score any predictor against the curated test set across structural validity, mark accuracy, encoding-field accuracy, hallucination rate, and latency. |
| 4. Demo | `app/` | Streamlit app rendering base vs. fine-tuned outputs side by side using `st.vega_lite_chart`. |

The eval suite is intentionally decoupled from the training code: it accepts any `predict_fn(question, schema) -> dict` and produces a JSON summary, so base-vs-tuned comparisons stay fair and the same harness can score later iterations or different base models.

## Eval metrics

| Metric | Definition |
|--------|------------|
| **Validity rate** | Fraction of predictions that pass the official Vega-Lite v5 JSON schema — the correctness floor. |
| **Mark accuracy** | Predicted mark type matches the gold mark. This is the project's tool-use proxy: in Vega-Lite, `mark` selects the chart-rendering primitive (the "tool"). |
| **Encoding-field accuracy** | Per-channel match of `field` values against gold — the tool arguments. |
| **Hallucination rate** | Fraction of encoded fields that don't exist in the question's schema. |
| **Latency** | Mean / p50 / p95 prediction time in milliseconds. |

Cost is zero by construction — see [Constraints](#constraints).

## Stack

| Stage | Tools |
|-------|-------|
| Curation | `datasets`, `jsonschema`, `pandas` |
| Training | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate` (CUDA, Colab) |
| Evaluation | `jsonschema`, `pandas`, custom metrics |
| Demo | `streamlit`, `transformers` (CPU / MPS inference) |

- **Base model:** [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)
- **Dataset:** nvBench via the [`TianqiLuo/nvBench2.0`](https://huggingface.co/datasets/TianqiLuo/nvBench2.0) Hugging Face mirror (MIT-licensed)
- **Env manager:** [`uv`](https://docs.astral.sh/uv/) with `pyproject.toml` + `uv.lock`
- **Lint / test:** `ruff`, `pytest`

## Constraints

Everything runs on free tiers — no paid APIs, no paid model hosting, no paid datasets:

- Training uses the free Colab T4 GPU; 4-bit QLoRA fits comfortably in 16 GB.
- The dataset and base model are openly licensed.
- Evaluation and the Streamlit demo run locally on CPU or Apple Silicon (MPS).
- The fine-tuned LoRA adapter is small enough to publish to a free Hugging Face Hub repo.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # core deps (curate + eval + inference)
uv sync --group train       # add CUDA-only training deps (Colab)
uv sync --group demo        # add Streamlit
uv sync --group dev         # add pytest + ruff

cp .env.example .env        # populate any keys you need
```

## Usage

### 1. Curate the dataset

```bash
uv run python -m curate.prepare
```

Writes `data/nvbench_train.jsonl` and `data/nvbench_test.jsonl`. Each line carries a question, an inferred column schema, and a validated Vega-Lite v5 gold spec — see [`curate/README.md`](curate/README.md) for the full output contract.

### 2. Fine-tune (Colab)

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab, point it at the curated train split (Drive mount or upload), and run the cells. The notebook:

- 4-bit-quantises Gemma 2 2B with `bitsandbytes`,
- attaches LoRA adapters via `peft`,
- runs supervised fine-tuning with `trl.SFTTrainer`,
- exports the adapter to Drive or the Hugging Face Hub.

### 3. Evaluate

```python
from evals.runner import score_model

summary = score_model(
    predict_fn=my_model.predict,        # (question, schema) -> spec dict
    test_path="data/nvbench_test.jsonl",
    output_path="runs/base.json",
)
```

Run the same call against the base and fine-tuned models to compare them on identical inputs.

### 4. Run the demo

```bash
uv run streamlit run app/streamlit_app.py
```

Side-by-side chart and JSON-spec rendering for a base and a fine-tuned model on a handful of sample questions, with a free-form override box.

## Project layout

```
curate/      Stage 1 — dataset preparation and Vega-Lite schema validation
train/       Stage 2 — fine-tuning code (consumed by the Colab notebook)
evals/       Stage 3 — scoring suite (named `evals` to avoid shadowing `eval`)
app/         Stage 4 — Streamlit demo
notebooks/   Colab notebook for training
tests/       pytest suite
data/        Curated splits + cached Vega-Lite schema (gitignored)
```

## Status

| Stage | State |
|-------|-------|
| Curate | Implemented; `uv run python -m curate.prepare` produces validated splits end-to-end. |
| Train | Notebook scaffold in place; full training loop in progress. |
| Evaluate | Implemented and tested; ready to score arbitrary predictors. |
| Demo | Implemented with a heuristic placeholder predictor; will swap in the trained adapter once available. |

Run the test suite with `uv run pytest -q`.

## License

[MIT](LICENSE). The underlying nvBench dataset is also MIT-licensed.
