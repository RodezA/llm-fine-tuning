# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

A demo-oriented LLM fine-tuning and evaluation harness covering:
- Dataset curation
- LoRA/QLoRA fine-tuning on an open model (Llama 3, Mistral, or Gemma)
- Structured eval suite: correctness, hallucination, tool-use accuracy, cost/latency benchmarks
- Streamlit app on top for demos
- Intended to be deployed to GitHub with a professional README

See `plan.md` for the original scope statement.

## Current state

Scaffolding stage. Module layout, dependency manifest, README, and a Colab notebook stub are in place. No actual implementation yet — module `__init__.py` files are placeholders.

Top-level layout:
- `curate/`, `train/`, `evals/`, `app/` — the four pipeline stages (see "Architectural intent" below)
- `notebooks/train_colab.ipynb` — Colab training entry point
- `tests/` — pytest suite
- `pyproject.toml` — uv-managed deps; groups: `train` (CUDA-only), `demo` (Streamlit), `dev` (pytest + ruff)
- `.env.example`, `.gitignore`, `README.md`, `plan.md`, this file

This directory is its own git repository (default branch `main`).

## Concrete decisions (keep these unless the user changes them)

- **Base model:** Gemma 2 2B Instruct (`google/gemma-2-2b-it`).
- **Task:** natural-language → Vega-Lite chart specification.
- **Dataset:** nvBench (NL-to-visualization, MIT-licensed).
- **Compute split:** training on free Colab T4 (CUDA, 4-bit QLoRA via `bitsandbytes`); eval and demo run locally on CPU/MPS.
- **Env manager:** `uv` with `pyproject.toml` + `uv.lock`. Python ≥ 3.11.
- **Lint/format:** `ruff`. **Test:** `pytest`.
- **Demo rendering:** `st.vega_lite_chart` directly — no Altair wrapper.
- **"Tool-use accuracy" eval framing:** mark-type selection + encoding-field correctness (Vega-Lite's `mark` is the tool, encodings are the args).

## Hard constraints from the user

- **Do not list Claude as a co-author or co-contributor in commits, the README, or anywhere else in this project.** Omit `Co-Authored-By: Claude` trailers and any "Generated with Claude Code" footer when committing here.
- The README must read as a professional, human-authored project page suitable for a public GitHub portfolio.
- **Everything must be free to run.** No paid APIs, paid model hosting, or paid datasets. Pick training/eval paths that work on free tiers (e.g. Colab free GPU, Hugging Face free inference, local CPU/MPS where feasible) and open-licensed models/datasets.

## Architectural intent (to guide future work)

The four-piece structure — **curate → fine-tune → evaluate → demo** — maps to top-level packages: `curate/`, `train/`, `evals/`, `app/`. The eval package is named `evals` (plural) to avoid shadowing Python's built-in `eval`. Each stage must be runnable independently: the eval suite in particular should score arbitrary models (base, fine-tuned, or different base models) without depending on training code, so base-vs-tuned comparisons are fair.

Don't invent additional architecture, abstractions, or stages without explicit user agreement — the scope above is what's been agreed.

## Secrets and configuration

Project secrets (e.g. `OPENAI_API_KEY`, `QDRANT_API_KEY`) live in a local `.env` file at the project root, loaded at runtime (e.g. `python-dotenv`). `.env` is gitignored; `.env.example` is the committed template — keep it up to date when new keys are introduced. Do not export secrets from the user's shell config.
