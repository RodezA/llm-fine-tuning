"""Validate Vega-Lite v5 specs against the official JSON schema.

The schema is fetched once from https://vega.github.io/schema/vega-lite/v5.json
and cached on disk under ``data/.schema_cache/`` so subsequent calls are offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

VEGA_LITE_V5_URL = "https://vega.github.io/schema/vega-lite/v5.json"

# Cache lives under the project ``data/`` directory (already gitignored).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / ".schema_cache"
_CACHE_PATH = _CACHE_DIR / "vega-lite-v5.json"

# Module-level memo so we only build the validator once per process.
_VALIDATOR: Draft7Validator | None = None


def _load_schema() -> dict[str, Any]:
    """Return the Vega-Lite v5 JSON schema, fetching and caching on first call."""
    if _CACHE_PATH.exists():
        with _CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    response = requests.get(VEGA_LITE_V5_URL, timeout=60)
    response.raise_for_status()
    schema = response.json()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(schema, f)
    return schema


def _get_validator() -> Draft7Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft7Validator(_load_schema())
    return _VALIDATOR


def validate_vega_lite_v5(spec: dict) -> tuple[bool, str | None]:
    """Validate ``spec`` against the Vega-Lite v5 JSON schema.

    The Vega-Lite top-level requires a data source. The output contract for this
    project intentionally excludes inline data values, so before validating we
    wrap the spec with a stubbed named data source (``{"name": "table"}``) when
    no ``data`` field is present. The caller's ``spec`` object is not mutated.

    Returns
    -------
    (is_valid, error_message)
        ``error_message`` is ``None`` on success, or the first ``jsonschema``
        validation error message on failure.
    """
    if not isinstance(spec, dict):
        return False, f"spec must be a dict, got {type(spec).__name__}"

    candidate = spec if "data" in spec else {**spec, "data": {"name": "table"}}

    validator = _get_validator()
    try:
        # Surface the *first* error rather than collecting all of them; that's
        # plenty for filtering and keeps the message short.
        error = next(iter(validator.iter_errors(candidate)), None)
    except ValidationError as exc:  # defensive — iter_errors shouldn't raise
        return False, exc.message

    if error is None:
        return True, None
    return False, error.message
