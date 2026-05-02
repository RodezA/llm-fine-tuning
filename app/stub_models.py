"""Stub and real-model backends for the Streamlit demo.

``stub_predict`` is a keyword-driven heuristic used while the fine-tuned adapter
is not yet available.  ``model_predict`` loads a LoRA adapter from the Hugging Face
Hub and calls the model; it is only reachable when ``HF_MODEL_ID`` is set in the
environment and the heavy ML dependencies are installed.

Do NOT use ``stub_predict`` for evaluation — it exists only so the demo renders
something meaningful in the cloud deployment before a trained adapter exists.
"""

from __future__ import annotations

import json
import re
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
    # Base stub: defaults to bar for everything — simulates a model that hasn't
    # learned the temporal-trend → line mapping that fine-tuning teaches.
    # Tuned stub: correct mark for each intent.
    if intent == "trend" and temporal and quantitative:
        base_mark = "bar"
        tuned_mark = "line"
    elif intent == "distribution" and quantitative:
        base_mark = "bar"
        tuned_mark = "bar"
    elif intent == "compare":
        base_mark = "bar"
        tuned_mark = "bar"
    else:
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

    if model == "tuned":
        # Correct chart type for trend: line + point overlay so individual values
        # are visible, plus tooltip for interactivity.
        if intent == "trend" and temporal and quantitative:
            encoding["x"] = {"field": x_field, "type": "temporal", "title": x_field}
            encoding["y"] = {"field": y_field, "type": "quantitative", "title": y_field}
            if x_field and y_field:
                encoding["tooltip"] = [
                    {"field": x_field, "type": "temporal"},
                    {"field": y_field, "type": "quantitative"},
                ]

        # Add color channel when a sensible categorical column exists.
        if nominal and nominal != x_field:
            encoding["color"] = {"field": nominal, "type": "nominal"}

        # Bin quantitative axis for a true histogram look.
        if intent == "distribution" and quantitative:
            encoding["x"] = {"field": quantitative, "type": "quantitative", "bin": True}
            encoding["y"] = {"aggregate": "count", "type": "quantitative"}

    mark = base_mark if model == "base" else (
        {"type": "line", "point": True} if (model == "tuned" and intent == "trend" and temporal and quantitative)
        else tuned_mark
    )

    spec: dict = {
        "$schema": VEGA_LITE_SCHEMA,
        "mark": mark,
        "encoding": encoding,
    }
    return spec


# ---------------------------------------------------------------------------
# Real-model inference (requires HF_MODEL_ID env var + ML deps installed)
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a data visualization assistant. Given a natural-language question and a "
    "table schema, output a minimal Vega-Lite v5 JSON specification that answers the "
    "question. Return only valid JSON — no explanation, no markdown fences."
)


def _schema_to_text(schema: list[dict]) -> str:
    lines = [f"  {col['name']} ({col['type']})" for col in schema]
    return "Table columns:\n" + "\n".join(lines)


def _build_prompt(question: str, schema: list[dict]) -> str:
    return (
        f"{_SYSTEM}\n\n{_schema_to_text(schema)}\n\nQuestion: {question}\n\nVega-Lite spec:"
    )


def _parse_spec(text: str) -> dict:
    """Extract a JSON object from model output, tolerating leading/trailing prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def model_predict(
    question: str,
    schema: list[dict],
    adapter_id: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> dict:
    """Generate a Vega-Lite spec using the trained LoRA adapter.

    The model and tokenizer are loaded once and cached by Streamlit.  Returns an
    empty dict (and lets the caller fall back to stub) if anything goes wrong.
    """
    try:
        import streamlit as st
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        @st.cache_resource(show_spinner="Loading model…")
        def _load(aid: str):
            base = "google/gemma-2-2b-it"
            tok = AutoTokenizer.from_pretrained(aid)
            tok.pad_token = tok.eos_token
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mdl = AutoModelForCausalLM.from_pretrained(
                base,
                torch_dtype=torch.float32,
                device_map=device,
                attn_implementation="eager",
            )
            mdl = PeftModel.from_pretrained(mdl, aid)
            mdl.eval()
            return mdl, tok

        model, tokenizer = _load(adapter_id)

        messages = [{"role": "user", "content": _build_prompt(question, schema)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return _parse_spec(generated)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Axis helpers (used by stub_predict)
# ---------------------------------------------------------------------------


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
