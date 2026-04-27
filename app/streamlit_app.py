"""Streamlit demo: side-by-side base vs. fine-tuned Vega-Lite spec generation.

The demo runs a question + dataframe through two stubbed "models" and renders both
the chart and the JSON spec each one emitted. The fine-tuned branch is currently a
heuristic placeholder (see ``app.stub_models``); once the LoRA adapter is trained
this module should swap the stub for the real model without other UI changes.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

try:
    from app.stub_models import stub_predict
except ModuleNotFoundError:
    from stub_models import stub_predict  # type: ignore[no-redef]


def _build_samples() -> dict[str, dict]:
    """Return ordered mapping of sample-question label -> {question, df}."""
    samples: dict[str, dict] = {
        "Monthly sales trend": {
            "question": "How have monthly sales trended over the last year?",
            "df": pd.DataFrame(
                {
                    "month": [
                        "2024-01", "2024-02", "2024-03", "2024-04",
                        "2024-05", "2024-06", "2024-07", "2024-08",
                        "2024-09", "2024-10", "2024-11", "2024-12",
                    ],
                    "sales": [
                        120, 135, 128, 150, 162, 175, 190, 188, 205, 220, 245, 260,
                    ],
                }
            ),
        },
        "Exam scores by class period": {
            "question": "Compare exam scores across class periods.",
            "df": pd.DataFrame(
                {
                    "period": ["P1", "P2", "P3", "P4", "P5", "P6"],
                    "avg_score": [78, 82, 74, 88, 91, 80],
                }
            ),
        },
        "Distribution of product unit prices": {
            "question": "What's the distribution of product unit prices?",
            "df": pd.DataFrame(
                {
                    "price": [
                        4.99, 9.50, 12.00, 15.25, 18.99, 22.40,
                        24.10, 27.75, 31.00, 35.60,
                    ],
                }
            ),
        },
        "Revenue by region": {
            "question": "Which region has the highest revenue?",
            "df": pd.DataFrame(
                {
                    "region": ["North", "South", "East", "West", "Central"],
                    "revenue": [120_000, 98_000, 142_000, 110_000, 135_000],
                }
            ),
        },
        "Daily active users": {
            "question": "How have daily active users changed over the past two weeks?",
            "df": pd.DataFrame(
                {
                    "day": pd.date_range("2026-04-01", periods=14, freq="D"),
                    "active_users": [
                        1200, 1240, 1190, 1305, 1410, 1380, 1455,
                        1490, 1520, 1610, 1580, 1700, 1755, 1820,
                    ],
                }
            ),
        },
        "Headcount by department": {
            "question": "Compare headcount across departments.",
            "df": pd.DataFrame(
                {
                    "department": [
                        "Engineering", "Sales", "Marketing",
                        "Support", "Operations", "Finance",
                    ],
                    "headcount": [85, 60, 25, 40, 30, 18],
                }
            ),
        },
        "Spread of response times": {
            "question": "Show the distribution of API response times.",
            "df": pd.DataFrame(
                {
                    "latency_ms": [
                        42, 55, 61, 68, 74, 80, 88, 95, 102, 110, 125, 140, 165, 210,
                    ],
                }
            ),
        },
    }
    return samples


def _render_model_column(
    column,
    title: str,
    spec: dict,
    df: pd.DataFrame,
) -> None:
    with column:
        st.subheader(title)
        st.vega_lite_chart(data=df, spec=spec, use_container_width=True)
        st.code(json.dumps(spec, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="NL to Vega-Lite Demo", layout="wide")
    st.title("Natural-language to Vega-Lite chart specs")
    st.caption(
        "Side-by-side comparison of a base model and a fine-tuned model translating a "
        "question + small table into a Vega-Lite v5 chart specification. The "
        "fine-tuned column is currently driven by a documented heuristic stub "
        "(see ``app/stub_models.py``) and will be replaced by the trained LoRA "
        "adapter once available."
    )

    samples = _build_samples()
    sample_labels = list(samples.keys())

    with st.sidebar:
        st.header("Inputs")
        chosen_label = st.selectbox(
            "Sample query",
            options=sample_labels,
            index=0,
            help="Pick a sample question. The bound dataset updates automatically.",
        )
        custom_question = st.text_input(
            "Override question (optional)",
            value="",
            help="If non-empty, this overrides the sample question above. "
            "The bound dataset still comes from the sample selector.",
        )

    sample = samples[chosen_label]
    bound_df: pd.DataFrame = sample["df"]
    resolved_question: str = (
        custom_question.strip() if custom_question.strip() else sample["question"]
    )

    st.markdown(f"**Question:** {resolved_question}")
    st.markdown("**Bound dataset preview**")
    st.dataframe(bound_df, use_container_width=True)

    base_spec = stub_predict(resolved_question, bound_df, model="base")
    tuned_spec = stub_predict(resolved_question, bound_df, model="tuned")

    base_col, tuned_col = st.columns(2)
    _render_model_column(base_col, "Base model", base_spec, bound_df)
    _render_model_column(tuned_col, "Fine-tuned model", tuned_spec, bound_df)


if __name__ not in ("app.streamlit_app", "streamlit_app"):
    main()
