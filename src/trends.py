"""
Deterministic longitudinal trend analysis for a single patient.

Uses ONLY stored DB data (reports + results) — no agent re‑runs, no LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.queries import fetch_patient_history


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _extract_first_or_none(arr: Any) -> Optional[str]:
    if isinstance(arr, list) and arr:
        v = arr[0]
        return str(v).strip() if v is not None else None
    return None


def _trend_label(values: List[Optional[float]]) -> str:
    """Label trend as improving / worsening / stable based on first vs last non‑null."""
    numeric = [v for v in values if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return "stable"
    first = float(numeric[0])
    last = float(numeric[-1])
    if last < first:
        return "improving"
    if last > first:
        return "worsening"
    return "stable"


def _diabetes_risk_from_a1c(a1c_value: Optional[float]) -> Optional[float]:
    """Mirror visualization risk scoring for diabetes using HbA1c."""
    if a1c_value is None:
        return None
    if a1c_value >= 9:
        return 0.9
    if a1c_value >= 8:
        return 0.75
    if a1c_value >= 7:
        return 0.6
    return 0.4


def _hypertension_risk_from_bp(bp_str: str) -> Optional[float]:
    """Mirror visualization risk scoring for hypertension using systolic BP."""
    bp_str = (bp_str or "").strip()
    if not bp_str or "/" not in bp_str:
        return None
    systolic_part = bp_str.split("/", 1)[0].strip()
    try:
        systolic = int(systolic_part)
    except ValueError:
        return None
    if systolic >= 180:
        return 0.9
    if systolic >= 160:
        return 0.75
    if systolic >= 140:
        return 0.6
    return 0.4


def build_patient_trends(patient_id: str) -> Dict[str, Any]:
    """
    Build longitudinal trend JSON for a patient from stored DB rows.

    Output:
    {
      "trend_summary": {
        "diabetes": "improving|worsening|stable",
        "hypertension": "improving|worsening|stable"
      },
      "trends": {
        "dates": [...],
        "a1c": [...],
        "glucose": [...],
        "bp": [...],
        "diabetes_risk": [...],
        "hypertension_risk": [...]
      }
    }
    """
    history = fetch_patient_history(patient_id)
    dates: List[str] = []
    a1c_trend: List[Optional[float]] = []
    glucose_trend: List[Optional[float]] = []
    bp_trend: List[Optional[str]] = []
    diab_risk_trend: List[Optional[float]] = []
    htn_risk_trend: List[Optional[float]] = []

    for row in history:
        created = row.get("report_created_at") or row.get("created_at")
        if created is not None:
            # created may already be str; keep it simple/consistent
            dates.append(str(created))
        else:
            dates.append("")

        extraction = row.get("extraction_json") or {}
        diabetes = (extraction.get("diabetes") or {}) if isinstance(extraction, dict) else {}
        bp = (extraction.get("blood_pressure") or {}) if isinstance(extraction, dict) else {}

        # HbA1c and glucose (numeric strings from extraction JSON)
        a1c_val = _extract_first_or_none(diabetes.get("a1c_values"))
        glucose_val = _extract_first_or_none(diabetes.get("glucose_values"))
        a1c_trend.append(_parse_float(a1c_val))
        glucose_trend.append(_parse_float(glucose_val))

        # Blood pressure: keep the first reading as a string (e.g. "130/85")
        bp_reading = _extract_first_or_none(bp.get("bp_readings"))
        bp_str = bp_reading or ""
        bp_trend.append(bp_str)

        # Dynamic risk scores computed from stored clinical values (no LLM, no re-runs).
        diab_score = _diabetes_risk_from_a1c(a1c_trend[-1])
        htn_score = _hypertension_risk_from_bp(bp_str)
        diab_risk_trend.append(diab_score)
        htn_risk_trend.append(htn_score)

    diabetes_label = _trend_label(diab_risk_trend)
    hypertension_label = _trend_label(htn_risk_trend)

    return {
        "trend_summary": {
            "diabetes": diabetes_label,
            "hypertension": hypertension_label,
        },
        "trends": {
            "dates": dates,
            "a1c": a1c_trend,
            "glucose": glucose_trend,
            "bp": bp_trend,
            "diabetes_risk": diab_risk_trend,
            "hypertension_risk": htn_risk_trend,
        },
    }

