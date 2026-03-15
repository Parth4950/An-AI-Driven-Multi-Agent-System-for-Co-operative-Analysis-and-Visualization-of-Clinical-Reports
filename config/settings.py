"""
Application settings loaded from environment.
CrewAI uses GEMINI_API_KEY and GEMINI_MODEL for its native Gemini provider.
OPENAI_API_KEY is never required.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# CrewAI native Gemini: gemini-2.5-flash is current for new users (gemini-2.0-flash deprecated)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = 0

# When True: no Gemini API calls; return deterministic mock JSON (full pipeline testing without billing)
DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in ("true", "1", "yes")

# Rate limit: min seconds between Gemini requests (global)
RATE_LIMIT_SEC = 2
# Max retries on 429 when retry_after is present (quota > 0). No retry when free-tier quota is 0.
MAX_429_RETRIES = 2

DATA_DIR = PROJECT_ROOT / "data"
FILTERED_NOTES_PATH = DATA_DIR / "filtered_discharge_notes.csv"

# Optional: full path to Tesseract executable for image OCR (e.g. C:\Program Files\Tesseract-OCR\tesseract.exe)
# Set in .env if Tesseract is not on system PATH
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip() or None

# CrewAI SQLite task output storage: use project data dir so it's writable (avoids readonly DB errors)
if "CREWAI_DB_PATH" not in os.environ:
    os.environ["CREWAI_DB_PATH"] = str(DATA_DIR)


def validate_settings() -> None:
    """Raise if required settings are missing. Skip GEMINI_API_KEY when DRY_RUN is enabled."""
    if DRY_RUN:
        return
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or is placeholder. Set it in .env (or use DRY_RUN=true for mock mode)"
        )
