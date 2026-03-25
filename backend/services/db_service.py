"""
Database service for the FastAPI backend.

Uses the existing PostgreSQL helpers in `db/queries.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from db.queries import fetch_all_patients, fetch_patient_history


def get_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    """Return full stored patient history (reports + JSON outputs)."""
    return fetch_patient_history(patient_id)


def get_all_patients() -> List[str]:
    """Return all patient IDs currently stored."""
    return fetch_all_patients()

