"""
Agent 2: Clinical Risk & Insight Analyzer.
Consumes validated extraction JSON from Agent 1; returns summary, risk insights, evidence, confidence.
No new extraction; no treatment advice.
"""

import json
import logging
import re
from typing import Any

from crewai import Crew

from config.settings import DRY_RUN, validate_settings
from src.agents import get_risk_analyzer_agent
from src.tasks import risk_analysis_task

logger = logging.getLogger(__name__)

# Extract single JSON object from raw LLM output
_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})")

INSIGHT_KEYS = ("summary", "diabetes_risk_insights", "hypertension_risk_insights", "supporting_evidence", "confidence_level")
CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})


def _extract_json_from_output(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    for m in _JSON_BLOCK_PATTERN.finditer(raw):
        g = m.group(1) or m.group(2)
        if g and g.strip().startswith("{"):
            return g.strip()
    return raw


def _default_insight() -> dict[str, Any]:
    return {
        "summary": "Insufficient data.",
        "diabetes_risk_insights": [],
        "hypertension_risk_insights": [],
        "supporting_evidence": {"labs": [], "vitals": [], "medications": []},
        "confidence_level": "low",
    }


def _normalize_insight(obj: dict[str, Any]) -> dict[str, Any]:
    """Enforce exact output schema; no extra keys."""
    out: dict[str, Any] = {
        "summary": str(obj.get("summary", "")).strip() or "Insufficient data.",
        "diabetes_risk_insights": list(obj.get("diabetes_risk_insights") or []) if isinstance(obj.get("diabetes_risk_insights"), list) else [],
        "hypertension_risk_insights": list(obj.get("hypertension_risk_insights") or []) if isinstance(obj.get("hypertension_risk_insights"), list) else [],
        "supporting_evidence": {"labs": [], "vitals": [], "medications": []},
        "confidence_level": "low",
    }
    ev = obj.get("supporting_evidence")
    if isinstance(ev, dict):
        for k in ("labs", "vitals", "medications"):
            v = ev.get(k)
            out["supporting_evidence"][k] = [str(x).strip() for x in (v if isinstance(v, list) else []) if str(x).strip()]
    cl = str(obj.get("confidence_level", "")).strip().lower()
    if cl in CONFIDENCE_VALUES:
        out["confidence_level"] = cl
    for k in ("diabetes_risk_insights", "hypertension_risk_insights"):
        out[k] = [str(x).strip() for x in out[k] if str(x).strip()]
    return out


def run_risk_analysis(extraction_json: dict[str, Any]) -> dict[str, Any]:
    """
    Run Agent 2 on one extraction JSON. Returns insight object with keys:
    summary, diabetes_risk_insights, hypertension_risk_insights, supporting_evidence, confidence_level.
    On parse/API failure returns default low-confidence insight (no raise).
    """
    validate_settings()
    if DRY_RUN:
        logger.info("DRY_RUN: risk analysis -> default insight (no API call)")
        return _default_insight()

    task = risk_analysis_task(extraction_json)
    agent = task.agent
    crew = Crew(agents=[agent], tasks=[task], memory=False, cache=False)

    try:
        result = crew.kickoff()
        raw = getattr(result, "raw", None) or str(result)
        json_str = _extract_json_from_output(raw)
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return _normalize_insight(obj)
    except (json.JSONDecodeError, TypeError, Exception) as e:
        logger.warning("Risk analysis parse/run failed: %s. Returning default insight.", e)
    return _default_insight()
