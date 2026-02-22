"""
CrewAI tasks for clinical extraction and JSON repair.
"""

import json
from crewai import Task

from src.agents import get_extractor_agent, get_risk_analyzer_agent

JSON_GUARDRAIL_TOP = "Output a single JSON object. No markdown, no backticks, no explanations. Required keys present; empty string or [] when not found."

EXPECTED_JSON_INSTRUCTION = (
    "Extract only explicitly stated diabetes and hypertension facts. "
    "Output one JSON object with keys: patient_id, diabetes, blood_pressure, abnormal_markers. "
    "Use empty string or empty array when not found."
)


def extraction_task(patient_id: str, note_text: str) -> Task:
    """Create a Task that extracts diabetes and BP data from one discharge note."""
    agent = get_extractor_agent()
    return Task(
        description=(
            JSON_GUARDRAIL_TOP + "\n\n"
            "From the discharge note below, extract only explicitly stated diabetes and blood pressure data.\n\n"
            "Discharge note:\n---\n"
            f"{note_text}\n---\n\n"
            + EXPECTED_JSON_INSTRUCTION
        ),
        expected_output="Single JSON object. No markdown or backticks.",
        agent=agent,
    )


def repair_json_task(broken_json: str) -> Task:
    """Create a Task that fixes invalid JSON without changing values. One-shot repair."""
    agent = get_extractor_agent()
    return Task(
        description=(
            JSON_GUARDRAIL_TOP + "\n\n"
            "Fix the following JSON to be strictly valid. Do not change values.\n\n"
            "Broken JSON:\n---\n"
            f"{broken_json}\n---\n\n"
            "Return only the corrected JSON object, no other text."
        ),
        expected_output="Valid JSON object only.",
        agent=agent,
    )


def risk_analysis_task(extraction_json: dict) -> Task:
    """Create a Task for Agent 2: analyze extraction JSON and return risk/insight JSON."""
    agent = get_risk_analyzer_agent()
    json_str = json.dumps(extraction_json, indent=2) if isinstance(extraction_json, dict) else str(extraction_json)
    return Task(
        description=(
            "You are Agent 2: Clinical Risk & Insight Analyzer.\n\n"
            "Input: the following JSON from Agent 1 (Extractor). Do NOT extract new facts or modify values.\n"
            "Analyze and summarize metabolic and BP status; identify risk signals and contributing factors "
            "supported only by the data below. Output a single JSON object with keys: summary, "
            "diabetes_risk_insights, hypertension_risk_insights, supporting_evidence (labs, vitals, medications), "
            "confidence_level (high|medium|low). No markdown, no extra keys.\n\n"
            "Extraction JSON:\n---\n"
            f"{json_str}\n---"
        ),
        expected_output="Single JSON object with summary, risk insights, supporting_evidence, confidence_level.",
        agent=agent,
    )
