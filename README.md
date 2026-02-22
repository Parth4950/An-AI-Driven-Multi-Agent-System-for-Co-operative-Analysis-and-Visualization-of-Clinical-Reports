# Clinical Analyzer

Multi-agent clinical pipeline using CrewAI and Gemini: processes MIMIC-IV discharge notes for diabetes and blood pressure. **Agent 1** extracts structured facts; **Agent 2** analyzes extraction JSON for metabolic/BP risk insights; **Agent 3** produces doctor and patient summaries from extraction + risk analysis. Outputs are validated and written to JSON for downstream use or gold-standard comparison.

## Agents

- **Agent 1 — Clinical Data Extractor:** Extracts only explicitly stated diabetes and hypertension data (type, status, A1C, glucose, BP readings, medications, abnormal markers). No inference; empty when not stated.
- **Agent 2 — Clinical Risk & Insight Analyzer:** Consumes Agent 1’s JSON. Summarizes metabolic and BP status; identifies diabetes and hypertension risk signals and contributing factors supported only by the extracted data. Outputs summary, risk insights, supporting evidence, and confidence level. No medical advice.
- **Agent 3 — Clinical Summarizer:** Consumes extraction (Agent 1) and risk_analysis (Agent 2). Produces a doctor-facing summary (clinical tone), a patient-facing summary (plain language), key_flags (only if supported by Agent 2), and data_gaps. No new facts, no treatment advice, no contradicting earlier agents.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and optional `GEMINI_MODEL`, `DRY_RUN`).

## Data

- Place MIMIC-IV discharge data as `data/discharge.csv.gz` (or adjust paths in config).
- Run the data filter, then main pipeline, then validation as needed.

## Usage

- **Filter notes:** `python src/data_filter.py` — filters discharge notes containing diabetes/hypertension mentions to `data/filtered_discharge_notes.csv`.
- **Run pipeline:** `python src/main.py` — runs Agent 1 (extractor) on each filtered note, validates output, then Agent 2 (risk analyzer), then Agent 3 (summarizer) on each result. Writes:
  - `data/extraction_results.json` — one extraction object per note (patient_id, diabetes, blood_pressure, abnormal_markers).
  - `data/risk_insights.json` — one insight object per note (summary, diabetes_risk_insights, hypertension_risk_insights, supporting_evidence, confidence_level).
  - `data/summaries.json` — one summary per note (patient_id, doctor_summary, patient_summary, key_flags, data_gaps).
- Set `DRY_RUN=true` in `.env` to use mock outputs (no API calls, no billing).
- **Validate vs gold standard:** `python src/validate_extraction.py` — compares `data/extraction_results.json` to `data/gold_standard.json` and prints metrics.

## Project structure

- `config/` — settings and extraction schema
- `src/` — agents (extractor, risk analyzer, summarizer), tasks, extraction runner, risk_analysis runner, summarizer runner, data filter, validator
- `data/` — inputs and outputs (CSV/JSON; large or sensitive files may be gitignored)

## Repository

- **GitHub:** [An-AI-Driven-Multi-Agent-System-for-Co-operative-Analysis-and-Visualization-of-Clinical-Reports](https://github.com/Parth4950/An-AI-Driven-Multi-Agent-System-for-Co-operative-Analysis-and-Visualization-of-Clinical-Reports) — CrewAI + Gemini pipeline for MIMIC-IV discharge summaries: extraction, risk/insight analysis, and clinical summaries into structured JSON for EHR dashboards.
