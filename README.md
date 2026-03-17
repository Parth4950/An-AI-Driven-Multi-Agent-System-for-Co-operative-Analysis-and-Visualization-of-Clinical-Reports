# Clinical Analyzer

Multi-agent clinical pipeline using CrewAI and Gemini: processes MIMIC-IV discharge notes for diabetes and blood pressure. **Agent 1** extracts structured facts; **Agent 2** analyzes extraction for risk insights; **Agent 3** produces doctor and patient summaries; **Agent 4** builds visualization data; **Agent 5 (Orchestrator)** runs the full pipeline. A Streamlit dashboard presents results as a clinical risk analysis UI (no raw JSON).

## Agents

- **Agent 1 — Clinical Data Extractor:** Extracts only explicitly stated diabetes and hypertension data (type, status, A1C, glucose, BP readings, medications, abnormal markers). No inference; empty when not stated.
- **Agent 2 — Clinical Risk & Insight Analyzer:** Consumes Agent 1’s JSON. Summarizes metabolic and BP status; identifies risk signals and contributing factors. Outputs summary, risk insights, supporting evidence, and confidence level. No medical advice.
- **Agent 3 — Clinical Summarizer:** Consumes extraction and risk_analysis. Produces doctor summary, patient summary, key_flags, and data_gaps. No new facts, no treatment advice.
- **Agent 4 — Visualization:** Deterministic transform of extraction + risk + summary into visualization-ready structures (risk levels, scores, severity indicator, evidence chart) for dashboards.
- **Agent 5 — Orchestrator:** Runs the pipeline in order: Extraction → Risk Analysis → Summary → Visualization. Single entry point for the full workflow.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root with:
   ```env
   GEMINI_API_KEY=your_key
   # optional
   # GEMINI_MODEL=gemini-1.5-flash
   # DRY_RUN=true
   # Database password for PostgreSQL storage
   CLINICAL_DB_PASSWORD=your_postgres_password
   # Optional: Tesseract path when not on PATH (for image OCR in dashboard)
   # TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
   ```
3. For the dashboard file upload: PDF/DOCX extraction uses `pypdf`, `pymupdf`, `python-docx`, `docx2txt` (in requirements). For image (PNG/JPG) OCR, install [Tesseract](https://github.com/tesseract-ocr/tesseract) on your system. If Tesseract is not on your PATH, set `TESSERACT_CMD` in `.env` to the full path or containing folder of the executable (e.g. `C:/Program Files/Tesseract-OCR` or `C:/Program Files/Tesseract-OCR/tesseract.exe` on Windows).
4. (Optional but recommended) Initialize the PostgreSQL schema:
   ```bash
   # Ensure Postgres is running and clinical_ai_db exists
   python -m db.init_db
   ```

## Data

- Place MIMIC-IV discharge data as `data/discharge.csv.gz` (or adjust paths in config).
- Run the data filter, then main pipeline, then validation as needed.

## Usage

- **Filter notes:** `python src/data_filter.py` — filters discharge notes to `data/filtered_discharge_notes.csv`.
- **Run pipeline (CLI):** `python src/main.py` — runs the Orchestrator (Agent 5) on each filtered note. Writes `data/extraction_results.json`, `data/risk_insights.json`, `data/summaries.json`, `data/visualizations.json`. With DB configured, each note is also stored in PostgreSQL (`patients`, `reports`, `results`).
- **Run dashboard:** `streamlit run src/app.py` — opens the Clinical Risk Analysis Dashboard.
  - **Pages (sidebar):**
    - **Upload Report**: Provide a patient identifier, then paste clinical text **or** upload a file. Supported formats: **PDF** (pypdf + PyMuPDF fallback), **DOCX** (python-docx with tables + docx2txt + raw ZIP/XML fallback), **PNG/JPG** (OCR via [Tesseract](https://github.com/tesseract-ocr/tesseract) — install Tesseract on your system for image support). The app extracts text, runs the multi‑agent pipeline, and stores inputs/outputs in PostgreSQL for that `patient_id`.
    - **Patient Dashboard**: Admin-style view to browse all patients and their history.
  - From the **Upload Report** page you can:
    - View doctor/patient summaries, risk severity indicator, risk scores chart, evidence table, key clinical flags, and data gaps.
    - Download a formatted PDF report or structured JSON report.
    - For patients with multiple stored reports, see **longitudinal trends**:
      - A trend table with Date, HbA1c, Glucose, BP, Diabetes Risk, Hypertension Risk.
      - Line charts for diabetes and hypertension risk scores over time.
      - Deterministic **overall vs recent trend insights** (e.g. “Overall: Improving ↓, Recent: Worsening ↑”).
  - From the **Patient Dashboard** page you can:
    - See an **All Patients Overview** table with Patient ID, last HbA1c, last BP, and overall risk level.
    - Select any patient to view a **Patient Summary Card** (latest HbA1c, BP, diabetes and hypertension risk with colored indicators).
    - View a full **Patient History** table (all reports) and line charts for HbA1c and risk scores over time, built purely from stored DB data (no re‑runs).
  - Activate the venv first (e.g. `.venv\Scripts\activate`) or run: `.venv\Scripts\python.exe -m streamlit run src/app.py`.
- Set `DRY_RUN=true` in `.env` for mock outputs (no API calls).
- **Validate vs gold standard:** `python src/validate_extraction.py` — compares extraction results to `data/gold_standard.json`.

## Project structure

- `config/` — settings and extraction schema
- `db/` — PostgreSQL connection helper, schema initializer, and query helpers (`patients`, `reports`, `results`, patient history fetch)
- `src/` — agents (extractor, risk analyzer, summarizer, visualizer, orchestrator), tasks, pipeline runners, document parser (PDF: pypdf+PyMuPDF; DOCX: python-docx+docx2txt+ZIP/XML; images: Tesseract OCR), Streamlit app (`app.py`, multimodal input + PDF/JSON report export + longitudinal trends), trend engine (`trends.py`), data filter, validator
- `data/` — inputs and outputs (CSV/JSON; large or sensitive files may be gitignored)
- `.streamlit/` — Streamlit config (e.g. usage stats off)

## Repository

- **GitHub:** [An-AI-Driven-Multi-Agent-System-for-Co-operative-Analysis-and-Visualization-of-Clinical-Reports](https://github.com/Parth4950/An-AI-Driven-Multi-Agent-System-for-Co-operative-Analysis-and-Visualization-of-Clinical-Reports) — CrewAI + Gemini pipeline for MIMIC-IV discharge summaries: extraction, risk/insight analysis, and clinical summaries into structured JSON for EHR dashboards.
