"""Unit tests for evals.metrics with synthetic specs constructed inline."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from evals import metrics
from evals.metrics import (
    encoding_field_accuracy,
    hallucination_rate,
    is_valid_spec,
    mark_accuracy,
    time_call,
)


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    """Clear in-memory schema/validator caches before each test for isolation."""
    metrics._SCHEMA_CACHE = None
    metrics._VALIDATOR_CACHE = None
    yield
    metrics._SCHEMA_CACHE = None
    metrics._VALIDATOR_CACHE = None


# ---------- mark_accuracy ----------


def test_mark_accuracy_string_form_match():
    pred = {"mark": "bar"}
    gold = {"mark": "bar"}
    assert mark_accuracy(pred, gold) is True


def test_mark_accuracy_object_form_match():
    pred = {"mark": {"type": "bar", "tooltip": True}}
    gold = {"mark": "bar"}
    assert mark_accuracy(pred, gold) is True


def test_mark_accuracy_mismatch():
    pred = {"mark": {"type": "line"}}
    gold = {"mark": {"type": "bar"}}
    assert mark_accuracy(pred, gold) is False


def test_mark_accuracy_missing_mark():
    assert mark_accuracy({}, {"mark": "bar"}) is False
    assert mark_accuracy({"mark": "bar"}, {}) is False


# ---------- encoding_field_accuracy ----------


def test_encoding_field_accuracy_all_match():
    pred = {
        "encoding": {
            "x": {"field": "year", "type": "temporal"},
            "y": {"field": "sales", "type": "quantitative"},
        }
    }
    gold = {
        "encoding": {
            "x": {"field": "year", "type": "temporal"},
            "y": {"field": "sales", "type": "quantitative"},
        }
    }
    result = encoding_field_accuracy(pred, gold)
    assert result == {"x": True, "y": True}


def test_encoding_field_accuracy_empty_encodings():
    """Empty encoding dicts on both sides => empty result, no false positives."""
    assert encoding_field_accuracy({"encoding": {}}, {"encoding": {}}) == {}
    assert encoding_field_accuracy({}, {}) == {}


def test_encoding_field_accuracy_pred_extra_channel():
    """Channel in pred but not gold counts as incorrect."""
    pred = {"encoding": {"x": {"field": "a"}, "color": {"field": "c"}}}
    gold = {"encoding": {"x": {"field": "a"}}}
    result = encoding_field_accuracy(pred, gold)
    assert result == {"x": True, "color": False}


def test_encoding_field_accuracy_gold_extra_channel():
    """Channel in gold but not pred counts as incorrect."""
    pred = {"encoding": {"x": {"field": "a"}}}
    gold = {"encoding": {"x": {"field": "a"}, "y": {"field": "b"}}}
    result = encoding_field_accuracy(pred, gold)
    assert result == {"x": True, "y": False}


def test_encoding_field_accuracy_field_mismatch():
    pred = {"encoding": {"x": {"field": "year"}}}
    gold = {"encoding": {"x": {"field": "month"}}}
    assert encoding_field_accuracy(pred, gold) == {"x": False}


# ---------- hallucination_rate ----------


def test_hallucination_rate_no_encodings_returns_zero():
    assert hallucination_rate({"mark": "bar"}, ["a", "b"]) == 0.0
    assert hallucination_rate({"encoding": {}}, ["a", "b"]) == 0.0


def test_hallucination_rate_all_allowed():
    spec = {
        "encoding": {
            "x": {"field": "year"},
            "y": {"field": "sales"},
        }
    }
    assert hallucination_rate(spec, ["year", "sales", "region"]) == 0.0


def test_hallucination_rate_partial_hallucination():
    spec = {
        "encoding": {
            "x": {"field": "year"},
            "y": {"field": "sales"},
            "color": {"field": "made_up_column"},
            "size": {"field": "another_fake"},
        }
    }
    # 2 of 4 fields hallucinated.
    assert hallucination_rate(spec, ["year", "sales"]) == 0.5


def test_hallucination_rate_ignores_channels_without_field():
    """Aggregate / value-encoded channels (no `field`) shouldn't count toward the denominator."""
    spec = {
        "encoding": {
            "x": {"field": "year"},
            "y": {"aggregate": "count"},
        }
    }
    assert hallucination_rate(spec, ["year"]) == 0.0


# ---------- is_valid_spec ----------


# A minimal real Vega-Lite v5 schema fragment used for offline tests so we don't depend on the
# network. Only enough structure to differentiate valid from invalid specs in the suite.
_FAKE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["mark"],
    "properties": {
        "mark": {
            "oneOf": [
                {
                    "type": "string",
                    "enum": ["bar", "line", "point", "area", "circle", "tick", "rect"],
                },
                {
                    "type": "object",
                    "required": ["type"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["bar", "line", "point", "area", "circle", "tick", "rect"],
                        }
                    },
                },
            ]
        },
        "encoding": {"type": "object"},
        "data": {"type": "object"},
    },
    "additionalProperties": True,
}


@pytest.fixture
def cached_schema(tmp_path, monkeypatch):
    """Place a fake schema at the cache path so is_valid_spec never hits the network."""
    cache_path = tmp_path / "data" / ".schema_cache" / "vega-lite-v5.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_FAKE_SCHEMA))
    monkeypatch.setattr(metrics, "SCHEMA_CACHE_PATH", cache_path)
    return cache_path


def test_is_valid_spec_valid(cached_schema):
    spec = {"mark": "bar", "encoding": {"x": {"field": "year"}}}
    valid, err = is_valid_spec(spec)
    assert valid is True
    assert err is None


def test_is_valid_spec_missing_mark(cached_schema):
    """Schema-invalid spec: no `mark` key at all."""
    spec = {"encoding": {"x": {"field": "year"}}}
    valid, err = is_valid_spec(spec)
    assert valid is False
    assert isinstance(err, str) and err  # non-empty error message


def test_is_valid_spec_invalid_mark_type(cached_schema):
    """Schema-invalid spec: mark is an unsupported string."""
    spec = {"mark": "definitely_not_a_mark"}
    valid, err = is_valid_spec(spec)
    assert valid is False
    assert isinstance(err, str) and err


def test_is_valid_spec_non_dict(cached_schema):
    valid, err = is_valid_spec("not a dict")  # type: ignore[arg-type]
    assert valid is False
    assert "dict" in (err or "")


def test_is_valid_spec_uses_cache_no_network(cached_schema):
    """If the cache file exists, no urlopen call should happen."""
    spec = {"mark": "bar"}
    with patch("evals.metrics.urllib.request.urlopen") as mock_urlopen:
        valid, _ = is_valid_spec(spec)
    assert valid is True
    mock_urlopen.assert_not_called()


def test_is_valid_spec_fetches_when_cache_missing(tmp_path, monkeypatch):
    """If the cache is absent, the function should fetch and write it once."""
    cache_path = tmp_path / "data" / ".schema_cache" / "vega-lite-v5.json"
    monkeypatch.setattr(metrics, "SCHEMA_CACHE_PATH", cache_path)

    payload = json.dumps(_FAKE_SCHEMA).encode("utf-8")

    class _FakeResponse:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "evals.metrics.urllib.request.urlopen", return_value=_FakeResponse(payload)
    ) as mock_urlopen:
        valid, err = is_valid_spec({"mark": "bar"})

    assert valid is True
    assert err is None
    assert cache_path.exists()
    assert mock_urlopen.call_count == 1


# ---------- time_call ----------


def test_time_call_returns_result_and_positive_latency():
    def adder(a, b, *, c=0):
        return a + b + c

    result, latency_ms = time_call(adder, 2, 3, c=4)
    assert result == 9
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


def test_time_call_propagates_exceptions():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        time_call(boom)


# ---------- runner smoke (uses synthetic specs only) ----------


def test_score_model_smoke(tmp_path, monkeypatch):
    """Smoke-test the runner end-to-end with synthetic specs and a stub predict_fn."""
    # Use cached fake schema so is_valid_spec doesn't hit the network.
    cache_path = tmp_path / "data" / ".schema_cache" / "vega-lite-v5.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_FAKE_SCHEMA))
    monkeypatch.setattr(metrics, "SCHEMA_CACHE_PATH", cache_path)

    from evals.runner import score_model

    test_path = tmp_path / "test.jsonl"
    output_path = tmp_path / "summary.json"

    examples = [
        {
            "id": "ex1",
            "question": "Sales by year",
            "schema": [
                {"name": "year", "type": "temporal"},
                {"name": "sales", "type": "quantitative"},
            ],
            "spec": {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "year"},
                    "y": {"field": "sales"},
                },
            },
        },
        {
            "id": "ex2",
            "question": "Trend over time",
            "schema": [{"name": "year", "type": "temporal"}],
            "spec": {
                "mark": "line",
                "encoding": {"x": {"field": "year"}},
            },
        },
    ]
    test_path.write_text("\n".join(json.dumps(ex) for ex in examples))

    def fake_predict(question: str, schema: list[dict]) -> dict:
        # Returns a perfect prediction for ex1, wrong mark for ex2.
        if "Sales" in question:
            return {
                "mark": "bar",
                "encoding": {"x": {"field": "year"}, "y": {"field": "sales"}},
            }
        return {"mark": "bar", "encoding": {"x": {"field": "year"}}}

    summary = score_model(fake_predict, test_path, output_path)

    assert summary["n_examples"] == 2
    assert summary["validity_rate"] == 1.0
    assert summary["mark_accuracy_rate"] == 0.5  # ex2 mismatched
    assert summary["mean_hallucination_rate"] == 0.0
    assert "p50" in summary["latency_ms"]
    assert "p95" in summary["latency_ms"]
    assert len(summary["per_example"]) == 2
    assert summary["per_example"][0]["id"] == "ex1"
    # Output file should contain the same summary.
    on_disk = json.loads(output_path.read_text())
    assert on_disk["n_examples"] == 2
