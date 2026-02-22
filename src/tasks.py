"""
CrewAI tasks for clinical extraction and JSON repair.
"""

from crewai import Task

from src.agents import get_extractor_agent

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
