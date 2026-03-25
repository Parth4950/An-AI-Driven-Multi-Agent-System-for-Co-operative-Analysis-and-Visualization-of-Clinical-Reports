from fastapi import FastAPI

# Ensure project root is importable (so `src` and `db` can be imported).
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Avoid CrewAI telemetry signal-handler issues in non-main threads.
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from backend.routes import analyze, patients

app = FastAPI(title="Clinical AI API")

app.include_router(analyze.router)
app.include_router(patients.router)


@app.get("/")
def root():
    return {"message": "API running"}

