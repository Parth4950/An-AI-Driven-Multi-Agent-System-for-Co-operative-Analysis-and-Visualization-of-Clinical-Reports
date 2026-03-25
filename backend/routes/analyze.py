from __future__ import annotations

from fastapi import APIRouter

from backend.models.schemas import ReportRequest
from backend.services.pipeline import run_pipeline_api

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("/")
def analyze(data: ReportRequest):
    result = run_pipeline_api(patient_id=data.patient_id, text=data.text)
    return result

