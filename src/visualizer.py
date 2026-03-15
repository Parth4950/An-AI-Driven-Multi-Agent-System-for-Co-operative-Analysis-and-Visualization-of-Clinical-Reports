"""
Agent 4: Visualization transformer.
Converts extraction + risk_analysis + summary into visualization-ready JSON.
Deterministic: no new analysis, no LLM. For frontend gauges, bar charts, evidence tables.
"""

from typing import Any


def _level_to_score(level: str) -> float:
    """Map risk level to 0-1 score for charts."""
    return {"High": 0.85, "Moderate": 0.55, "Low": 0.25}.get(level, 0.0)


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

    evidence_chart = _build_evidence_chart(extraction, risk)

    return {
        "visualizations": {
            "risk_levels": {
                "hypertension": hypertension_level,
                "diabetes": diabetes_level,
            },
            "risk_scores": {
                "hypertension_score": round(_level_to_score(hypertension_level), 2),
                "diabetes_score": round(_level_to_score(diabetes_level), 2),
            },
            "severity_indicator": {
                "level": level_overall,
                "color": color,
            },
            "evidence_chart": evidence_chart,
        }
    }
