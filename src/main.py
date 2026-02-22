"""
Entry point: load first 5 rows from filtered_discharge_notes.csv,
run CrewAI extractor on each note, validate JSON, and handle errors.
429 and quota handling in run_extraction; DRY_RUN for mock mode without billing.
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
from src.extraction import run_extraction
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

    results = []
    for idx, row in df.iterrows():
        patient_id = _get_patient_id(row, idx)
        note_text = row[TEXT_COLUMN]
        if pd.isna(note_text) or not str(note_text).strip():
            logger.warning("Row %s: empty note, skipping", idx)
            continue
        try:
            out = run_extraction(patient_id=patient_id, note_text=str(note_text))
            out = validate_extraction_output(out, verbose=True)
            results.append(out)
            if "error" not in out:
                logger.info("patient_id=%s OK", patient_id)
            else:
                logger.warning("patient_id=%s error: %s", patient_id, out.get("error"))
        except (ValueError, json.JSONDecodeError, Exception) as e:
            if _is_429_error(e):
                logger.error("patient_id=%s 429/quota: %s", patient_id, str(e))
            elif _is_404_model_error(e):
                logger.error("patient_id=%s model 404/unavailable: %s", patient_id, str(e))
            else:
                logger.exception("patient_id=%s failed: %s", patient_id, e)
            # Only validated JSON is written: use full empty schema + error, then validate
            error_payload = {**empty_schema(patient_id), "error": str(e)}
            results.append(validate_extraction_output(error_payload, verbose=True))

    # Optionally write results to data/ (e.g. extraction_results.json)
    out_path = _PROJECT_ROOT / "data" / "extraction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d results to %s", len(results), out_path)


if __name__ == "__main__":
    main()
