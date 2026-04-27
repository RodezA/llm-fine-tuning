"""Tests for ``curate.validate.validate_vega_lite_v5``.

Verifies positive and negative paths against the official Vega-Lite v5
JSON schema. The first test pulls the schema once (network or cache); the
remaining tests are served entirely from the on-disk cache.
"""

from __future__ import annotations

from curate.validate import validate_vega_lite_v5


def test_minimal_good_spec_is_valid() -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "amount", "type": "quantitative"},
        },
    }
    ok, err = validate_vega_lite_v5(spec)
    assert ok, f"expected valid spec, got error: {err}"
    assert err is None


def test_spec_missing_required_fields_is_rejected() -> None:
    # No mark, no layered/repeat structure -> the schema's anyOf cannot match.
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
        },
    }
    ok, err = validate_vega_lite_v5(spec)
    assert not ok
    assert err is not None and err  # non-empty message


def test_spec_with_invalid_mark_is_rejected() -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "definitely-not-a-mark",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "amount", "type": "quantitative"},
        },
    }
    ok, err = validate_vega_lite_v5(spec)
    assert not ok
    assert err is not None
    # The error message should mention the offending mark value.
    assert "definitely-not-a-mark" in err or "mark" in err.lower()
