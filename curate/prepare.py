"""Prepare nvBench training data for natural-language to Vega-Lite generation.

Pipeline
--------
1. Stream the ``TianqiLuo/nvBench2.0`` Hugging Face mirror of nvBench. We use
   that mirror because it is permissively packaged as parquet and exposes the
   exact fields we need (``nl_query``, ``table_schema``, ``gold_answer``)
   without requiring a clone of the upstream repository.
2. For each row we pick the first candidate spec from ``gold_answer`` (an
   ordered list of equally-acceptable visualizations), normalise it into a
   minimal Vega-Lite v5 document, and infer encoding-channel types from the
   table-schema column examples.
3. Validate every normalised spec against the official Vega-Lite v5 JSON
   schema. Examples whose gold spec fails validation are skipped and counted.
4. Write a deterministic 90/10 train/test split as JSONL files under
   ``data/`` matching the contract documented in ``curate/__init__.py`` and
   the project README.

Run via::

    uv run python -m curate.prepare
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset

from curate.validate import validate_vega_lite_v5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_DATASET_ID = "TianqiLuo/nvBench2.0"
HF_SPLITS: tuple[str, ...] = ("train", "dev", "test")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data"
TRAIN_PATH = _OUTPUT_DIR / "nvbench_train.jsonl"
TEST_PATH = _OUTPUT_DIR / "nvbench_test.jsonl"

# The schema example arrays in nvBench may use these primitive Python types.
_INT_TYPES = (int,)
_FLOAT_TYPES = (float,)

# Heuristics for temporal columns: name match OR ISO-ish value match.
_TEMPORAL_NAME_RE = re.compile(r"(?:^|_)(date|time|year|month|day|datetime)(?:$|_)", re.IGNORECASE)
_TEMPORAL_VALUE_RE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{4}-\d{2}-\d{2}T|^\d{1,2}[-/]\d{1,2}[-/]\d{4}"
)

# Mark-name fixups: nvBench uses Vega-Lite v3-ish names that need translation
# to the v5 primitives accepted by the official schema.
_MARK_REWRITES: dict[str, str] = {"pie": "arc"}

# Encoding channels that should always carry an explicit ``type`` if a field
# is present. Validating against the v5 schema is permissive without ``type``,
# but the eval suite expects normalized rows, so we set it where we can.
_TYPED_CHANNELS = {"x", "y", "x2", "y2", "color", "size", "shape", "theta", "row", "column"}


# ---------------------------------------------------------------------------
# Schema-type inference
# ---------------------------------------------------------------------------


def infer_column_type(column_name: str, examples: list[Any]) -> str:
    """Return a Vega-Lite type string for a single column.

    Order: temporal (by name or value), quantitative (numeric), nominal otherwise.
    """
    if _TEMPORAL_NAME_RE.search(column_name):
        return "temporal"

    string_examples = [e for e in examples if isinstance(e, str)]
    if string_examples and all(_TEMPORAL_VALUE_RE.match(e) for e in string_examples):
        return "temporal"

    if examples and all(isinstance(e, _INT_TYPES + _FLOAT_TYPES) and not isinstance(e, bool)
                        for e in examples):
        return "quantitative"

    return "nominal"


def build_schema(table_schema: dict[str, Any]) -> list[dict[str, str]]:
    """Convert nvBench's ``table_schema`` blob to the project schema list shape."""
    columns = table_schema.get("table_columns", []) or []
    examples = table_schema.get("column_examples", {}) or {}
    return [
        {"name": col, "type": infer_column_type(col, examples.get(col, []))}
        for col in columns
    ]


# ---------------------------------------------------------------------------
# Spec normalization
# ---------------------------------------------------------------------------


def _column_type_lookup(schema_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["name"]: row["type"] for row in schema_rows}


def normalize_spec(raw_spec: dict[str, Any], schema_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Return a Vega-Lite v5 spec built from an nvBench gold answer.

    - Adds the ``$schema`` URL.
    - Rewrites legacy mark names (``pie`` -> ``arc``).
    - Annotates encoding channels with an explicit ``type`` derived from the
      table schema, falling back to the channel's natural default.
    - Carries forward optional clauses (``transform``, ``aggregate``, etc.).
    - Does NOT include any inline data values.
    """
    spec: dict[str, Any] = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json"}

    # Mark
    mark = raw_spec.get("mark")
    if isinstance(mark, str):
        spec["mark"] = _MARK_REWRITES.get(mark, mark)
    elif isinstance(mark, dict):
        m = dict(mark)
        if isinstance(m.get("type"), str):
            m["type"] = _MARK_REWRITES.get(m["type"], m["type"])
        spec["mark"] = m
    else:
        # No mark -> not a chartable spec; the caller will drop it on validate.
        spec["mark"] = mark

    # Encoding
    encoding = raw_spec.get("encoding") or {}
    type_lookup = _column_type_lookup(schema_rows)
    new_encoding: dict[str, Any] = {}
    for channel, channel_def in encoding.items():
        if not isinstance(channel_def, dict):
            new_encoding[channel] = channel_def
            continue
        cd = dict(channel_def)
        if channel in _TYPED_CHANNELS and "type" not in cd:
            field = cd.get("field")
            aggregate = cd.get("aggregate")
            if isinstance(field, str) and field in type_lookup:
                cd["type"] = type_lookup[field]
            elif aggregate in {"count", "sum", "mean", "average", "min", "max", "median"}:
                cd["type"] = "quantitative"
            else:
                # Sensible default for positional channels without a field.
                cd["type"] = "nominal"
        new_encoding[channel] = cd
    spec["encoding"] = new_encoding

    # Pass through transform if present (nvBench filter blocks etc.).
    if "transform" in raw_spec:
        spec["transform"] = raw_spec["transform"]

    return spec


# ---------------------------------------------------------------------------
# Source iteration
# ---------------------------------------------------------------------------


def _iter_hf_examples() -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(split, example)`` pairs from every HF split of nvBench."""
    for split in HF_SPLITS:
        ds = load_dataset(HF_DATASET_ID, split=split)
        for idx, row in enumerate(ds):
            yield split, {"_idx": idx, **row}


def _parse_json_field(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def build_examples(rows: Iterable[tuple[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    """Convert raw nvBench rows into project-shape examples and report skip count."""
    out: list[dict[str, Any]] = []
    skipped = 0
    for split, row in rows:
        question = row.get("nl_query")
        if not isinstance(question, str) or not question.strip():
            skipped += 1
            continue

        table_schema = _parse_json_field(row.get("table_schema"), default={})
        gold_specs = _parse_json_field(row.get("gold_answer"), default=[])
        if not isinstance(gold_specs, list) or not gold_specs:
            skipped += 1
            continue

        schema_rows = build_schema(table_schema)
        raw_spec = gold_specs[0]
        if not isinstance(raw_spec, dict):
            skipped += 1
            continue

        spec = normalize_spec(raw_spec, schema_rows)
        ok, _err = validate_vega_lite_v5(spec)
        if not ok:
            skipped += 1
            continue

        out.append({
            "id": f"nvbench-{split}-{row['_idx']}",
            "question": question.strip(),
            "schema": schema_rows,
            "spec": spec,
        })
    return out, skipped


# ---------------------------------------------------------------------------
# Splitting and writing
# ---------------------------------------------------------------------------


def split_examples(
    examples: list[dict[str, Any]],
    *,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    return shuffled[n_test:], shuffled[:n_test]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[curate] loading {HF_DATASET_ID} from Hugging Face ...")
    raw_rows = list(_iter_hf_examples())
    print(f"[curate] raw rows: {len(raw_rows)}")

    examples, skipped = build_examples(raw_rows)
    print(f"[curate] kept {len(examples)} after Vega-Lite v5 validation "
          f"(skipped {skipped})")

    train, test = split_examples(examples, test_fraction=0.1, seed=42)
    n_train = write_jsonl(TRAIN_PATH, train)
    n_test = write_jsonl(TEST_PATH, test)

    print(f"[curate] wrote train: {n_train} -> {TRAIN_PATH}")
    print(f"[curate] wrote test:  {n_test} -> {TEST_PATH}")


if __name__ == "__main__":
    main()
