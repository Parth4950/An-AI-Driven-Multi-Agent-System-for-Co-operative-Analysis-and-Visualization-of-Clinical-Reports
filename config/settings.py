"""
Application settings loaded from environment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Must be a model name CrewAI recognizes as Gemini (e.g. gemini-2.0-flash, gemini-1.5-flash)
# so it uses native Gemini provider and GEMINI_API_KEY from env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = 0.0

DATA_DIR = PROJECT_ROOT / "data"
FILTERED_NOTES_PATH = DATA_DIR / "filtered_discharge_notes.csv"


def validate_settings() -> None:
    """Raise if required settings are missing."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or is placeholder. Set it in .env"
        )
