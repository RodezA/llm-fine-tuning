"""Pure metric functions for scoring predicted Vega-Lite specs against gold references."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError

VEGA_LITE_V5_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"
SCHEMA_CACHE_PATH = Path("data/.schema_cache/vega-lite-v5.json")

# Module-level cache so repeated validations within a process don't re-parse from disk.
_SCHEMA_CACHE: dict[str, Any] | None = None
_VALIDATOR_CACHE: Draft7Validator | None = None


def _load_schema(cache_path: Path | None = None) -> dict[str, Any]:
    """Load the Vega-Lite v5 JSON schema, fetching to disk cache on first use."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    if cache_path is None:
        cache_path = SCHEMA_CACHE_PATH

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VEGA_LITE_V5_SCHEMA_URL, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
        cache_path.write_text(payload)

    _SCHEMA_CACHE = json.loads(cache_path.read_text())
    return _SCHEMA_CACHE


def _get_validator(cache_path: Path | None = None) -> Draft7Validator:
    """Build (and memoize) a Draft7Validator for the Vega-Lite v5 schema."""
    global _VALIDATOR_CACHE
    if _VALIDATOR_CACHE is not None:
        return _VALIDATOR_CACHE
    schema = _load_schema(cache_path)
    _VALIDATOR_CACHE = Draft7Validator(schema)
    return _VALIDATOR_CACHE


def is_valid_spec(spec: dict) -> tuple[bool, str | None]:
    """Validate a spec against the Vega-Lite v5 JSON schema.

    Returns (True, None) if valid, else (False, error_message).
    """
    if not isinstance(spec, dict):
        return False, f"spec must be a dict, got {type(spec).__name__}"
    try:
        validator = _get_validator()
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        return False, f"schema unavailable: {exc}"

    errors = sorted(validator.iter_errors(spec), key=lambda e: e.path)
    if not errors:
        return True, None
    first = errors[0]
    path = ".".join(str(p) for p in first.absolute_path) or "<root>"
    return False, f"{path}: {first.message}"


def _mark_type(mark: Any) -> str | None:
    """Extract the mark type from either string ("bar") or object ({"type": "bar", ...}) form."""
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        t = mark.get("type")
        return t if isinstance(t, str) else None
    return None


def mark_accuracy(pred: dict, gold: dict) -> bool:
    """True iff pred's mark type matches gold's mark type."""
    pred_mark = _mark_type(pred.get("mark"))
    gold_mark = _mark_type(gold.get("mark"))
    if pred_mark is None or gold_mark is None:
        return False
    return pred_mark == gold_mark


def encoding_field_accuracy(pred: dict, gold: dict) -> dict[str, bool]:
    """Per-channel correctness of the encoded `field` value.

    For each channel found in either pred or gold:
    - Both missing: correct (not represented in output, since the channel isn't there).
    - Channel in gold but missing in pred: incorrect.
    - Channel in pred but missing in gold: incorrect.
    - Both present: correct iff `field` values match (None==None counts as match).
    """
    pred_enc = pred.get("encoding", {}) or {}
    gold_enc = gold.get("encoding", {}) or {}
    channels = set(pred_enc.keys()) | set(gold_enc.keys())
    result: dict[str, bool] = {}
    for ch in channels:
        in_pred = ch in pred_enc
        in_gold = ch in gold_enc
        if in_pred and in_gold:
            pred_field = (pred_enc.get(ch) or {}).get("field")
            gold_field = (gold_enc.get(ch) or {}).get("field")
            result[ch] = pred_field == gold_field
        else:
            # Asymmetric presence is always wrong for that channel.
            result[ch] = False
    return result


def hallucination_rate(spec: dict, allowed_columns: list[str]) -> float:
    """Fraction of encoded `field` values that aren't in `allowed_columns`.

    Returns 0.0 if no fields are encoded at all.
    """
    encoding = spec.get("encoding", {}) or {}
    allowed = set(allowed_columns)
    fields: list[str] = []
    for channel_def in encoding.values():
        if not isinstance(channel_def, dict):
            continue
        field = channel_def.get("field")
        if isinstance(field, str):
            fields.append(field)
    if not fields:
        return 0.0
    bad = sum(1 for f in fields if f not in allowed)
    return bad / len(fields)


def time_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run fn(*args, **kwargs); return (result, latency_ms) using time.perf_counter."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
