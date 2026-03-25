from __future__ import annotations

from fastapi import APIRouter

from backend.services.db_service import get_all_patients, get_patient_history
from src.trends import build_patient_trends

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/")
def list_patients():
    return {"patient_ids": get_all_patients()}


@router.get("/{patient_id}")
def get_patient(patient_id: str):
    return get_patient_history(patient_id)


@router.get("/{patient_id}/trends")
def get_patient_trends(patient_id: str):
    # Deterministic trend computation, based only on stored DB data.
    return build_patient_trends(patient_id)

