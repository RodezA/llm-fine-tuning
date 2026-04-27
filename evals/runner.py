"""End-to-end eval runner: invoke a predict_fn over a JSONL test set and write a summary."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.metrics import (
    encoding_field_accuracy,
    hallucination_rate,
    is_valid_spec,
    mark_accuracy,
    time_call,
)

PredictFn = Callable[[str, list[dict]], dict]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def score_model(
    predict_fn: PredictFn,
    test_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Score a predict function against a JSONL gold set; write & return a summary dict.

    Each line in `test_path` is expected to look like:
        {"id": "...", "question": "...",
         "schema": [{"name": "col", "type": "quantitative|nominal|temporal|ordinal"}],
         "spec": { /* gold vega-lite v5 spec */ }}
    """
    test_path = Path(test_path)
    output_path = Path(output_path)
    examples = _load_jsonl(test_path)

    per_example: list[dict[str, Any]] = []
    validity_flags: list[bool] = []
    mark_flags: list[bool] = []
    encoding_means: list[float] = []
    hallucination_values: list[float] = []
    latencies: list[float] = []

    for ex in examples:
        ex_id = ex.get("id")
        question = ex.get("question", "")
        schema = ex.get("schema", []) or []
        gold_spec = ex.get("spec", {}) or {}
        allowed_columns = [
            col["name"] for col in schema if isinstance(col, dict) and "name" in col
        ]

        pred_spec, latency_ms = time_call(predict_fn, question, schema)
        if not isinstance(pred_spec, dict):
            pred_spec = {}

        valid, err = is_valid_spec(pred_spec)
        mark_ok = mark_accuracy(pred_spec, gold_spec)
        enc_acc = encoding_field_accuracy(pred_spec, gold_spec)
        enc_mean = _mean([1.0 if v else 0.0 for v in enc_acc.values()]) if enc_acc else 1.0
        hallu = hallucination_rate(pred_spec, allowed_columns)

        validity_flags.append(valid)
        mark_flags.append(mark_ok)
        encoding_means.append(enc_mean)
        hallucination_values.append(hallu)
        latencies.append(latency_ms)

        per_example.append(
            {
                "id": ex_id,
                "valid": valid,
                "validity_error": err,
                "mark_accuracy": mark_ok,
                "encoding_field_accuracy": enc_acc,
                "encoding_field_accuracy_mean": enc_mean,
                "hallucination_rate": hallu,
                "latency_ms": latency_ms,
            }
        )

    n = len(examples)
    summary: dict[str, Any] = {
        "n_examples": n,
        "validity_rate": _mean([1.0 if v else 0.0 for v in validity_flags]),
        "mark_accuracy_rate": _mean([1.0 if m else 0.0 for m in mark_flags]),
        "mean_encoding_field_accuracy": _mean(encoding_means),
        "mean_hallucination_rate": _mean(hallucination_values),
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
        },
        "per_example": per_example,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary
