"""
Entry point: load filtered_discharge_notes.csv and run the clinical pipeline via the Orchestrator (Agent 5).
The orchestrator runs Extraction → Risk Analysis → Summary → Visualization for each note.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DRY_RUN, FILTERED_NOTES_PATH, validate_settings
from src.orchestrator import run_pipeline
from src.validator import empty_schema, validate_extraction_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _is_429_error(e: Exception) -> bool:
    """True if exception is 429 / quota (suppress stack trace for expected quota errors)."""
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def _is_404_model_error(e: Exception) -> bool:
    """True if exception is 404 / model not available (suppress stack for expected API errors)."""
    s = str(e)
    return "404" in s or "NOT_FOUND" in s or "no longer available" in s

MAX_ROWS = 5
TEXT_COLUMN = "text"
PATIENT_ID_COLUMNS = ("subject_id", "hadm_id", "patient_id")  # first found wins


def _get_patient_id(row: pd.Series, index: int) -> str:
    """Derive patient_id from row; fallback to row index."""
    for col in PATIENT_ID_COLUMNS:
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col]).strip()
    return str(index)


def main() -> None:
    validate_settings()
    if DRY_RUN:
        logger.info("DRY_RUN=true: mock extraction (no Gemini API calls)")

    if not FILTERED_NOTES_PATH.exists():
        raise FileNotFoundError(
            f"Filtered notes file not found: {FILTERED_NOTES_PATH}. Run data_filter.py first."
        )

    df = pd.read_csv(FILTERED_NOTES_PATH, nrows=MAX_ROWS, low_memory=False)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' does not exist. Available: {list(df.columns)}"
        )

    pipeline_results = []
    for idx, row in df.iterrows():
        patient_id = _get_patient_id(row, idx)
        note_text = row[TEXT_COLUMN]
        if pd.isna(note_text) or not str(note_text).strip():
            logger.warning("Row %s: empty note, skipping", idx)
            continue
        try:
            result = run_pipeline(str(note_text), patient_id=patient_id)
            pipeline_results.append(result)
            logger.info("patient_id=%s pipeline OK", patient_id)
        except (ValueError, json.JSONDecodeError, Exception) as e:
            if _is_429_error(e):
                logger.error("patient_id=%s 429/quota: %s", patient_id, str(e))
            elif _is_404_model_error(e):
                logger.error("patient_id=%s model 404/unavailable: %s", patient_id, str(e))
            else:
                logger.exception("patient_id=%s pipeline failed: %s", patient_id, e)
            error_extraction = validate_extraction_output(
                {**empty_schema(patient_id), "error": str(e)}, verbose=True
            )
            pipeline_results.append({
                "extraction": error_extraction,
                "risk_analysis": {
                    "summary": "Insufficient data.",
                    "diabetes_risk_insights": [],
                    "hypertension_risk_insights": [],
                    "supporting_evidence": {"labs": [], "vitals": [], "medications": []},
                    "confidence_level": "low",
                },
                "summary": {
                    "doctor_summary": "",
                    "patient_summary": "",
                    "key_flags": [],
                    "data_gaps": [],
                },
                "visualizations": {
                    "visualizations": {
                        "risk_levels": {"hypertension": "Low", "diabetes": "Low"},
                        "risk_scores": {"hypertension_score": 0.25, "diabetes_score": 0.25},
                        "severity_indicator": {"level": "Low", "color": "green"},
                        "evidence_chart": [],
                    }
                },
            })

    # Write outputs from orchestrator results
    extraction_list = [r["extraction"] for r in pipeline_results]
    risk_list = [
        {"patient_id": r["extraction"].get("patient_id", ""), **r["risk_analysis"]}
        for r in pipeline_results
    ]
    summary_list = [
        {"patient_id": r["extraction"].get("patient_id", ""), **r["summary"]}
        for r in pipeline_results
    ]
    viz_list = [
        {"patient_id": r["extraction"].get("patient_id", ""), **r["visualizations"]}
        for r in pipeline_results
    ]

    out_path = _PROJECT_ROOT / "data" / "extraction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extraction_list, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d results to %s", len(extraction_list), out_path)

    risk_path = _PROJECT_ROOT / "data" / "risk_insights.json"
    with open(risk_path, "w", encoding="utf-8") as f:
        json.dump(risk_list, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d risk insights to %s", len(risk_list), risk_path)

    summary_path = _PROJECT_ROOT / "data" / "summaries.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d summaries to %s", len(summary_list), summary_path)

    viz_path = _PROJECT_ROOT / "data" / "visualizations.json"
    with open(viz_path, "w", encoding="utf-8") as f:
        json.dump(viz_list, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d visualizations to %s", len(viz_list), viz_path)

    # Print final pipeline output for each result
    print("\n===== FINAL PIPELINE OUTPUT =====\n")
    for result in pipeline_results:
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
