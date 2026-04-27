# Curate (Stage 1)

Loads nvBench (NL-to-visualization), normalises it to Vega-Lite v5, validates
each gold spec against the official JSON schema, and writes the train/test
JSONL splits consumed by the downstream training and eval modules.

## Source

We use the Hugging Face mirror [`TianqiLuo/nvBench2.0`](https://huggingface.co/datasets/TianqiLuo/nvBench2.0).
It packages the upstream [TsinghuaDatabaseGroup/nvBench](https://github.com/TsinghuaDatabaseGroup/nvBench)
data as parquet with the three fields we need (`nl_query`, `table_schema`,
`gold_answer`) and downloads in seconds versus cloning the multi-MB upstream
repo. License of the underlying data is MIT (per the upstream repository).

If the HF mirror ever becomes unavailable the upstream GitHub repo can be
cloned into a temp dir and parsed instead — see `prepare.py` for the
field shapes we rely on.

## Output contract

`data/nvbench_train.jsonl` and `data/nvbench_test.jsonl`. Each line is:

```json
{
  "id": "nvbench-<split>-<index>",
  "question": "natural language question text",
  "schema": [{"name": "column", "type": "quantitative|nominal|temporal|ordinal"}],
  "spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "...",
    "encoding": { ... }
  }
}
```

The persisted spec deliberately omits any inline `data` values; the validator
in `validate.py` injects a stub `{"name": "table"}` data clause at validation
time so the spec validates without leaking data into the training corpus.

## Run

```bash
uv run python -m curate.prepare
```
