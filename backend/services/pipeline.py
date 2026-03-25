"""
Pipeline service for the FastAPI backend.

This calls the existing Agent 1→2→3→4 orchestrator without modifying any agent logic.
"""

from __future__ import annotations

from typing import Any, Dict

from src.orchestrator import run_pipeline


def run_pipeline_api(patient_id: str, text: str) -> Dict[str, Any]:
    """
    Run the full pipeline for a single report.

    Returns the same JSON-shaped dict as `src.orchestrator.run_pipeline`.
    """
    # input_type is always "text" here because FastAPI currently receives raw text.
    return run_pipeline(input_text=text, patient_id=patient_id, input_type="text")

