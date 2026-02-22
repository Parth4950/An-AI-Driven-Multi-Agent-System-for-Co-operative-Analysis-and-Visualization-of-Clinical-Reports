"""
Agent 3: Clinical Summarizer.
Consumes extraction (Agent 1) and risk_analysis (Agent 2) JSON; returns doctor_summary, patient_summary, key_flags, data_gaps.
No new facts; no treatment advice; no contradicting earlier agents.
"""

import json
import logging
import re
from typing import Any

from crewai import Crew

from config.settings import DRY_RUN, validate_settings
from src.agents import get_summarizer_agent
from src.tasks import summarizer_task

logger = logging.getLogger(__name__)

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})")


def _extract_json_from_output(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    for m in _JSON_BLOCK_PATTERN.finditer(raw):
        g = m.group(1) or m.group(2)
        if g and g.strip().startswith("{"):
            return g.strip()
    return raw


def _default_summary() -> dict[str, Any]:
    return {
        "doctor_summary": "",
        "patient_summary": "",
        "key_flags": [],
        "data_gaps": [],
    }


def _normalize_summary(obj: dict[str, Any]) -> dict[str, Any]:
    """Enforce exact output schema; no extra keys."""
    out: dict[str, Any] = {
        "doctor_summary": str(obj.get("doctor_summary", "")).strip(),
        "patient_summary": str(obj.get("patient_summary", "")).strip(),
        "key_flags": [],
        "data_gaps": [],
    }
    for k in ("key_flags", "data_gaps"):
        v = obj.get(k)
        out[k] = [str(x).strip() for x in (v if isinstance(v, list) else []) if str(x).strip()]
    return out


def run_summarization(extraction_json: dict[str, Any], risk_analysis_json: dict[str, Any]) -> dict[str, Any]:
    """
    Run Agent 3 on extraction + risk_analysis. Returns object with keys:
    doctor_summary, patient_summary, key_flags, data_gaps.
    On parse/API failure returns default empty summary (no raise).
    """
    validate_settings()
    if DRY_RUN:
        logger.info("DRY_RUN: summarization -> default empty summary (no API call)")
        return _default_summary()

    task = summarizer_task(extraction_json, risk_analysis_json)
    agent = task.agent
    crew = Crew(agents=[agent], tasks=[task], memory=False, cache=False)

    try:
        result = crew.kickoff()
        raw = getattr(result, "raw", None) or str(result)
        json_str = _extract_json_from_output(raw)
        obj = json.loads(json_str)
        if isinstance(obj, dict):
            return _normalize_summary(obj)
    except (json.JSONDecodeError, TypeError, Exception) as e:
        logger.warning("Summarization parse/run failed: %s. Returning default summary.", e)
    return _default_summary()
