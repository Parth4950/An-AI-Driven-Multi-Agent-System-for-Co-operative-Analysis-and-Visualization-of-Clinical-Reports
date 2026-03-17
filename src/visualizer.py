"""
Agent 4: Visualization transformer.
Converts extraction + risk_analysis + summary into visualization-ready JSON.
Deterministic: no new analysis, no LLM. For frontend gauges, bar charts, evidence tables.
"""

from typing import Any, Optional


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _level_to_score(level: str) -> float:
    """Fallback mapping of risk level to 0-1 score for charts."""
    return {"High": 0.85, "Moderate": 0.55, "Low": 0.25}.get(level, 0.0)


def _diabetes_risk_from_a1c(extraction: dict[str, Any]) -> Optional[float]:
    """
    Compute diabetes risk score from HbA1c using deterministic thresholds:
    - a1c >= 9   -> 0.9
    - a1c >= 8   -> 0.75
    - a1c >= 7   -> 0.6
    - else       -> 0.4
    Returns None if no numeric HbA1c is available.
    """
    diabetes = (extraction or {}).get("diabetes") or {}
    vals = diabetes.get("a1c_values") or []
    if not isinstance(vals, list) or not vals:
        return None
    first = _parse_float(vals[0])
    if first is None:
        return None
    a1c = first
    if a1c >= 9:
        return 0.9
    if a1c >= 8:
        return 0.75
    if a1c >= 7:
        return 0.6
    return 0.4


def _hypertension_risk_from_bp(extraction: dict[str, Any]) -> Optional[float]:
    """
    Compute hypertension risk score from systolic blood pressure using deterministic thresholds:
    - systolic >= 180 -> 0.9
    - systolic >= 160 -> 0.75
    - systolic >= 140 -> 0.6
    - else            -> 0.4
    Returns None if no parsable BP reading is available.
    """
    bp_block = (extraction or {}).get("blood_pressure") or {}
    readings = bp_block.get("bp_readings") or []
    if not isinstance(readings, list) or not readings:
        return None
    raw = str(readings[0] or "").strip()
    if not raw or "/" not in raw:
        return None
    systolic_str = raw.split("/", 1)[0].strip()
    try:
        systolic = int(systolic_str)
    except ValueError:
        return None
    if systolic >= 180:
        return 0.9
    if systolic >= 160:
        return 0.75
    if systolic >= 140:
        return 0.6
    return 0.4


def _confidence_to_level(confidence: str, insight_count: int) -> str:
    """Derive risk level from Agent 2 confidence and number of insights. Deterministic."""
    c = (confidence or "").strip().lower()
    if insight_count == 0:
        return "Low"
    if c == "high":
        return "High" if insight_count >= 1 else "Moderate"
    if c == "medium":
        return "Moderate" if insight_count >= 1 else "Low"
    return "Low"


def _overall_severity(hypertension_level: str, diabetes_level: str) -> tuple[str, str]:
    """Overall severity level and display color. Deterministic."""
    order = {"High": 2, "Moderate": 1, "Low": 0}
    h = order.get(hypertension_level, 0)
    d = order.get(diabetes_level, 0)
    level = "High" if max(h, d) == 2 else ("Moderate" if max(h, d) == 1 else "Low")
    color = "red" if level == "High" else ("orange" if level == "Moderate" else "green")
    return level, color


def _build_evidence_chart(extraction: dict[str, Any], risk_analysis: dict[str, Any]) -> list[dict[str, str]]:
    """Build label/value pairs from extraction and risk_analysis.supporting_evidence. No new data."""
    out: list[dict[str, str]] = []
    ext = extraction or {}
    risk = risk_analysis or {}
    # From extraction
    bp = (ext.get("blood_pressure") or {}).get("bp_readings") or []
    if isinstance(bp, list):
        for v in bp[:5]:  # cap for chart
            if isinstance(v, str) and v.strip():
                out.append({"label": "Blood Pressure", "value": v.strip()})
    diabetes = ext.get("diabetes") or {}
    for v in (diabetes.get("glucose_values") or [])[:5]:
        if isinstance(v, str) and v.strip():
            out.append({"label": "Glucose", "value": f"{v.strip()} mg/dL"})
    for v in (diabetes.get("a1c_values") or [])[:5]:
        if isinstance(v, str) and v.strip():
            out.append({"label": "HbA1c", "value": f"{v.strip()}%"})
    # From risk_analysis supporting_evidence
    evidence = risk.get("supporting_evidence") or {}
    for lab in (evidence.get("labs") or [])[:10]:
        if isinstance(lab, str) and lab.strip():
            out.append({"label": "Lab", "value": lab.strip()})
    for vit in (evidence.get("vitals") or [])[:10]:
        if isinstance(vit, str) and vit.strip():
            out.append({"label": "Vital", "value": vit.strip()})
    return out


def build_visualization(
    extraction: dict[str, Any],
    risk_analysis: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Transform Agent 1 + 2 + 3 outputs into visualization-ready JSON.
    No new medical information. Deterministic and derived from inputs only.
    summary is optional (unused in current schema but accepted for API consistency).
    """
    risk = risk_analysis or {}
    conf = (risk.get("confidence_level") or "").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"

    diabetes_insights = risk.get("diabetes_risk_insights") or []
    hypertension_insights = risk.get("hypertension_risk_insights") or []
    if not isinstance(diabetes_insights, list):
        diabetes_insights = []
    if not isinstance(hypertension_insights, list):
        hypertension_insights = []

    hypertension_level = _confidence_to_level(conf, len(hypertension_insights))
    diabetes_level = _confidence_to_level(conf, len(diabetes_insights))

    level_overall, color = _overall_severity(hypertension_level, diabetes_level)

    # Dynamic risk scores based on actual clinical values (with level-based fallback).
    diabetes_score = _diabetes_risk_from_a1c(extraction)
    if diabetes_score is None:
        diabetes_score = _level_to_score(diabetes_level)
    hypertension_score = _hypertension_risk_from_bp(extraction)
    if hypertension_score is None:
        hypertension_score = _level_to_score(hypertension_level)

    evidence_chart = _build_evidence_chart(extraction, risk)

    return {
        "visualizations": {
            "risk_levels": {
                "hypertension": hypertension_level,
                "diabetes": diabetes_level,
            },
            "risk_scores": {
                "hypertension_score": round(float(hypertension_score), 2),
                "diabetes_score": round(float(diabetes_score), 2),
            },
            "severity_indicator": {
                "level": level_overall,
                "color": color,
            },
            "evidence_chart": evidence_chart,
        }
    }
