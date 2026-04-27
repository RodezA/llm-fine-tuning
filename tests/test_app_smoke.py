"""Smoke tests for the Streamlit demo app and stub model heuristics."""

from __future__ import annotations

import pandas as pd
import pytest


def test_imports_clean() -> None:
    """The demo modules import cleanly without raising."""
    import app.streamlit_app  # noqa: F401
    import app.stub_models  # noqa: F401


@pytest.fixture
def sample_pairs() -> list[tuple[str, pd.DataFrame]]:
    return [
        (
            "How have monthly sales trended over the last year?",
            pd.DataFrame(
                {
                    "month": ["2024-01", "2024-02", "2024-03", "2024-04"],
                    "sales": [100, 120, 140, 160],
                }
            ),
        ),
        (
            "Compare exam scores across class periods.",
            pd.DataFrame(
                {
                    "period": ["P1", "P2", "P3"],
                    "avg_score": [78, 82, 91],
                }
            ),
        ),
        (
            "What's the distribution of product unit prices?",
            pd.DataFrame({"price": [4.99, 9.5, 12.0, 15.25, 18.99, 22.4]}),
        ),
        (
            "Which region has the highest revenue?",
            pd.DataFrame(
                {
                    "region": ["North", "South", "East", "West"],
                    "revenue": [120_000, 98_000, 142_000, 110_000],
                }
            ),
        ),
    ]


def test_stub_predict_returns_required_keys(
    sample_pairs: list[tuple[str, pd.DataFrame]],
) -> None:
    from app.stub_models import stub_predict

    for question, df in sample_pairs:
        for variant in ("base", "tuned"):
            spec = stub_predict(question, df, model=variant)
            assert isinstance(spec, dict)
            assert "mark" in spec, f"missing 'mark' for {variant!r}: {question!r}"
            assert "encoding" in spec, f"missing 'encoding' for {variant!r}: {question!r}"
            assert spec.get("$schema", "").endswith("/v5.json")


def test_tuned_at_least_as_rich_as_base(
    sample_pairs: list[tuple[str, pd.DataFrame]],
) -> None:
    """For at least one query, tuned should expose more encoding channels than base."""
    from app.stub_models import stub_predict

    saw_richer_tuned = False
    for question, df in sample_pairs:
        base = stub_predict(question, df, model="base")
        tuned = stub_predict(question, df, model="tuned")
        base_channels = set(base["encoding"].keys())
        tuned_channels = set(tuned["encoding"].keys())
        # Tuned must never be poorer than base.
        assert base_channels.issubset(tuned_channels) or len(tuned_channels) >= len(
            base_channels
        ), f"tuned regressed below base for {question!r}"
        if len(tuned_channels) > len(base_channels):
            saw_richer_tuned = True

    assert saw_richer_tuned, "expected tuned output to be strictly richer for >=1 query"
