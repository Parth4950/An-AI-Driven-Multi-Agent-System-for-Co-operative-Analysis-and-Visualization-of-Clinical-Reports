"""
Direct backend access for the Streamlit dashboard (no HTTP / FastAPI).

Maps UI actions to orchestrator and PostgreSQL helpers with Streamlit caching
so agents and DB reads are not re-initialized on every widget rerun.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from config.settings import validate_settings


def _db_error_message(exc: Exception) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    return (
        f"Database unavailable: {msg}. "
        "Ensure PostgreSQL is running, the schema is initialized (`python -m db.init_db`), "
        "and `CLINICAL_DB_PASSWORD` is set in `.env`."
    )


def _analysis_error_message(exc: Exception) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    upper = msg.upper()
    if "GEMINI" in upper or "API_KEY" in upper or "GOOGLE" in upper:
        return f"LLM configuration error: {msg}. Set `GEMINI_API_KEY` in `.env`."
    if "connect" in msg.lower() or "postgresql" in msg.lower() or "CLINICAL_DB" in upper:
        return _db_error_message(exc)
    return f"Clinical analysis failed: {msg}"


@st.cache_resource(show_spinner=False)
def bootstrap_clinical_app() -> bool:
    """Validate settings once per Streamlit server process."""
    validate_settings()
    return True


def run_clinical_analysis(
    patient_id: str,
    clinical_text: str,
    *,
    input_type: str = "text",
) -> Dict[str, Any]:
    """
    Run Agent 5 orchestrator: Extraction → Risk → Summary → Visualization.
    Not cached — each note is unique.
    """
    bootstrap_clinical_app()
    from src.orchestrator import run_pipeline

    try:
        return run_pipeline(
            input_text=clinical_text,
            patient_id=patient_id,
            input_type=input_type,
        )
    except Exception as exc:
        raise RuntimeError(_analysis_error_message(exc)) from exc


@st.cache_data(ttl=60, show_spinner=False)
def load_all_patient_ids() -> List[str]:
    from db.queries import fetch_all_patients

    try:
        return list(fetch_all_patients())
    except Exception as exc:
        raise RuntimeError(_db_error_message(exc)) from exc


@st.cache_data(ttl=60, show_spinner=False)
def load_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    from db.queries import fetch_patient_history

    try:
        return fetch_patient_history(patient_id)
    except Exception as exc:
        raise RuntimeError(_db_error_message(exc)) from exc


@st.cache_data(ttl=60, show_spinner=False)
def load_patient_trends(patient_id: str) -> Dict[str, Any]:
    from src.trends import build_patient_trends

    try:
        return build_patient_trends(patient_id)
    except Exception as exc:
        raise RuntimeError(_db_error_message(exc)) from exc


def invalidate_patient_data_cache(patient_id: str | None = None) -> None:
    """Refresh patient lists and history after a new report is stored."""
    load_all_patient_ids.clear()
    if patient_id:
        load_patient_history.clear(patient_id)
        load_patient_trends.clear(patient_id)
    else:
        load_patient_history.clear()
        load_patient_trends.clear()
