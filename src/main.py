"""
Entry point: load first 5 rows from filtered_discharge_notes.csv,
run CrewAI extractor on each note, validate JSON, and handle errors.
Retries on Gemini 429 (quota) with backoff.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on path when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import FILTERED_NOTES_PATH, validate_settings
from src.extraction import run_extraction

# Retry on 429 (quota): wait seconds suggested by API or default, then retry
MAX_429_RETRIES = 3
DEFAULT_429_WAIT_SEC = 32
# Pattern to parse "Please retry in 31.98s" from Gemini error
RETRY_AFTER_PATTERN = re.compile(r"[Rr]etry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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
        last_error = None
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                out = run_extraction(patient_id=patient_id, note_text=str(note_text))
                results.append(out)
                logger.info("Extraction OK for patient_id=%s", patient_id)
                break
            except (ValueError, json.JSONDecodeError) as e:
                logger.exception("Extraction failed for patient_id=%s: %s", patient_id, e)
                results.append(
                    {
                        "patient_id": patient_id,
                        "error": str(e),
                        "diabetes": {},
                        "blood_pressure": {},
                        "abnormal_markers": [],
                    }
                )
                break
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
                if is_429 and attempt < MAX_429_RETRIES:
                    match = RETRY_AFTER_PATTERN.search(err_str)
                    wait_sec = float(match.group(1)) if match else DEFAULT_429_WAIT_SEC
                    wait_sec = min(wait_sec, 120)
                    logger.warning(
                        "Gemini 429 (quota). Waiting %.0fs then retry %d/%d for patient_id=%s",
                        wait_sec, attempt + 1, MAX_429_RETRIES, patient_id,
                    )
                    time.sleep(wait_sec)
                    last_error = e
                    continue
                last_error = e
                logger.exception("Extraction failed for patient_id=%s: %s", patient_id, e)
                results.append(
                    {
                        "patient_id": patient_id,
                        "error": str(e),
                        "diabetes": {},
                        "blood_pressure": {},
                        "abnormal_markers": [],
                    }
                )
                break
        else:
            if last_error is not None:
                logger.exception(
                    "Extraction failed for patient_id=%s after %d quota retries: %s",
                    patient_id, MAX_429_RETRIES, last_error,
                )
                results.append(
                    {
                        "patient_id": patient_id,
                        "error": str(last_error),
                        "diabetes": {},
                        "blood_pressure": {},
                        "abnormal_markers": [],
                    }
                )

    # Optionally write results to data/ (e.g. extraction_results.json)
    out_path = _PROJECT_ROOT / "data" / "extraction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d results to %s", len(results), out_path)


if __name__ == "__main__":
    main()
