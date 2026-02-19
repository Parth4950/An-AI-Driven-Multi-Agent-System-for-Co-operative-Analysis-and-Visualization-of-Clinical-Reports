# Clinical Analyzer

Multi-agent clinical summarization pipeline using CrewAI: processes MIMIC-IV discharge notes for diabetes and blood pressure extraction, with validation against a gold-standard dataset.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and any other keys).

## Data

- Place MIMIC-IV discharge data as `data/discharge.csv.gz` (or adjust paths in config).
- Run the data filter, then extraction, then validation as needed.

## Usage

- **Filter notes:** `python src/data_filter.py` — filters discharge notes containing diabetes/hypertension/A1C mentions to `data/filtered_discharge_notes.csv`.
- **Run extraction:** `python src/main.py` — runs the Extractor agent on filtered notes and writes `data/extraction_results.json`.
- **Validate:** `python src/validate_extraction.py` — compares `data/extraction_results.json` to `data/gold_standard.json` and prints accuracy/recall metrics.

## Project structure

- `config/` — settings and extraction schema
- `src/` — agents, tasks, extraction runner, data filter, validation script
- `data/` — inputs and outputs (CSV/JSON; large or sensitive files are gitignored)
