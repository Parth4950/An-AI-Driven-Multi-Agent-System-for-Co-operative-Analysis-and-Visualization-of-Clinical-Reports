"""
Production-grade filter for MIMIC-IV discharge notes.
Streams discharge.csv.gz in chunks and writes rows matching clinical keywords
to filtered_discharge_notes.csv.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Keywords to match (case-insensitive) in the text column
KEYWORDS = ("diabetes", "hypertension", "a1c")
CHUNK_SIZE = 10_000
COMPRESSION = "gzip"

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "discharge.csv.gz"
OUTPUT_PATH = PROJECT_ROOT / "data" / "filtered_discharge_notes.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _ensure_text_column(df: pd.DataFrame) -> None:
    """Raise a clear error if the required 'text' column is missing."""
    if "text" not in df.columns:
        raise ValueError(
            f"Column 'text' does not exist. Available columns: {list(df.columns)}"
        )


def _text_matches(row_text: str) -> bool:
    """Return True if row text contains any keyword (case-insensitive)."""
    if pd.isna(row_text):
        return False
    lower = str(row_text).lower()
    return any(kw in lower for kw in KEYWORDS)


def run_filter() -> None:
    """
    Stream discharge.csv.gz in chunks, filter by keywords, append matches to CSV.
    Creates output file if it does not exist; preserves all original columns.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_scanned = 0
    total_matched = 0
    chunk_number = 0
    write_header = True

    logger.info("Starting filtered read from %s", INPUT_PATH)

    for chunk in pd.read_csv(
        INPUT_PATH,
        compression=COMPRESSION,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_number += 1
        _ensure_text_column(chunk)

        scanned = len(chunk)
        total_scanned += scanned

        mask = chunk["text"].apply(_text_matches)
        matched_chunk = chunk.loc[mask]
        matched_count = len(matched_chunk)
        total_matched += matched_count

        logger.info(
            "Chunk %d | Rows scanned: %d | Rows matched: %d | Total scanned: %d | Total matched: %d",
            chunk_number,
            scanned,
            matched_count,
            total_scanned,
            total_matched,
        )

        if matched_count > 0:
            matched_chunk.to_csv(
                OUTPUT_PATH,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            write_header = False

    logger.info(
        "Finished. Total rows scanned: %d, total rows matched: %d, output: %s",
        total_scanned,
        total_matched,
        OUTPUT_PATH,
    )


def main() -> None:
    """Entry point with main guard."""
    run_filter()


if __name__ == "__main__":
    main()
