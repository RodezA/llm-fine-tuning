"""Placeholder stub model used in the demo until the fine-tuned LoRA adapter is trained.

Do NOT use these heuristics for evaluation. They exist only so the Streamlit demo can
show a side-by-side "base vs. tuned" rendering of Vega-Lite specs while the real model
is still being curated and trained. The "tuned" branch is intentionally a strict
super-set of the "base" branch so the UI shows a visible improvement.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

_TREND_KEYWORDS = ("trend", "over time", "monthly", "daily", "weekly", "yearly")
_COMPARE_KEYWORDS = ("compare", " by ", "across", "versus", " vs ", "which", "highest", "lowest")
_DISTRIBUTION_KEYWORDS = ("distribution", "spread", "histogram", "range of")


def _infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Map each column to a Vega-Lite type ('temporal', 'quantitative', 'nominal')."""
    types: dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "temporal"
            continue
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "quantitative"
            continue
        # Heuristic: month-name strings or YYYY-MM strings should be treated as temporal.
        if series.dtype == object:
            sample = str(series.iloc[0]) if len(series) else ""
            month_names = {
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec",
            }
            if sample[:3].lower() in month_names or _looks_like_year_month(sample):
                types[col] = "temporal"
                continue
        types[col] = "nominal"
    return types


def _looks_like_year_month(value: str) -> bool:
    if len(value) < 7:
        return False
    return value[:4].isdigit() and value[4] == "-" and value[5:7].isdigit()


def _pick_columns(
    df: pd.DataFrame, col_types: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    """Return (temporal, quantitative, nominal) column names if available."""
    temporal = next((c for c, t in col_types.items() if t == "temporal"), None)
    quantitative = next((c for c, t in col_types.items() if t == "quantitative"), None)
    nominal = next((c for c, t in col_types.items() if t == "nominal"), None)
    return temporal, quantitative, nominal


def _classify_question(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in _TREND_KEYWORDS):
        return "trend"
    if any(kw in q for kw in _DISTRIBUTION_KEYWORDS):
        return "distribution"
    if any(kw in q for kw in _COMPARE_KEYWORDS):
        return "compare"
    return "default"


def stub_predict(
    question: str, df: pd.DataFrame, model: Literal["base", "tuned"]
) -> dict:
    """Return a deterministic Vega-Lite v5 spec for (question, df).

    This is a keyword-driven placeholder. The "tuned" variant is engineered to be
    strictly richer than the "base" variant (extra color encoding when a categorical
    column exists, and a more appropriate mark for ambiguous cases).
    """
    col_types = _infer_column_types(df)
    temporal, quantitative, nominal = _pick_columns(df, col_types)
    intent = _classify_question(question)

    # --- Mark selection ----------------------------------------------------
    if intent == "trend" and temporal and quantitative:
        base_mark = "line"
        tuned_mark = "line"
    elif intent == "distribution" and quantitative:
        base_mark = "bar"
        tuned_mark = "bar"
    elif intent == "compare":
        base_mark = "bar"
        tuned_mark = "bar"
    else:
        # Ambiguous: tuned makes a more thoughtful pick.
        base_mark = "bar"
        tuned_mark = "line" if temporal and quantitative else "bar"

    # --- Encoding selection ------------------------------------------------
    x_field, x_type, y_field, y_type = _pick_axes(
        intent, df, col_types, temporal, quantitative, nominal
    )

    encoding: dict[str, dict] = {}
    if x_field is not None:
        encoding["x"] = {"field": x_field, "type": x_type}
    if y_field is not None:
        encoding["y"] = {"field": y_field, "type": y_type}

    # The "tuned" model adds a color channel when a sensible categorical column
    # is available and isn't already on the x-axis.
    if model == "tuned" and nominal and nominal != x_field:
        encoding["color"] = {"field": nominal, "type": "nominal"}
    if model == "tuned" and intent == "distribution" and quantitative:
        # Bin the quantitative axis for a true histogram look.
        encoding["x"] = {"field": quantitative, "type": "quantitative", "bin": True}
        encoding["y"] = {"aggregate": "count", "type": "quantitative"}

    mark = base_mark if model == "base" else tuned_mark

    spec: dict = {
        "$schema": VEGA_LITE_SCHEMA,
        "mark": mark,
        "encoding": encoding,
    }
    return spec


def _pick_axes(
    intent: str,
    df: pd.DataFrame,
    col_types: dict[str, str],
    temporal: str | None,
    quantitative: str | None,
    nominal: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Choose (x_field, x_type, y_field, y_type) for the spec."""
    if intent == "trend" and temporal and quantitative:
        return temporal, "temporal", quantitative, "quantitative"
    if intent == "distribution" and quantitative:
        return quantitative, "quantitative", None, None
    # compare / default: prefer nominal on x, quantitative on y.
    if nominal and quantitative:
        return nominal, "nominal", quantitative, "quantitative"
    if temporal and quantitative:
        return temporal, "temporal", quantitative, "quantitative"
    # Last-resort fallback: first two columns.
    cols = list(df.columns)
    if len(cols) >= 2:
        return cols[0], col_types[cols[0]], cols[1], col_types[cols[1]]
    if len(cols) == 1:
        return cols[0], col_types[cols[0]], None, None
    return None, None, None, None
