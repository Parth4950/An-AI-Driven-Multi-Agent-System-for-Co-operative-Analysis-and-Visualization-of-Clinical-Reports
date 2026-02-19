"""
CrewAI tasks for clinical extraction.
"""

from crewai import Task

from config.schema import EXTRACTION_SCHEMA
from src.agents import get_extractor_agent

# Expected output: strict raw JSON only (injected into task description)
EXPECTED_JSON_INSTRUCTION = (
    "Your response must be STRICT RAW JSON only. No explanations. No markdown. No text outside JSON. "
    f"Use exactly this schema: {EXTRACTION_SCHEMA}"
)


def extraction_task(patient_id: str, note_text: str) -> Task:
    """Create a Task that extracts diabetes and BP data from one discharge note."""
    agent = get_extractor_agent()
    return Task(
        description=(
            "Extract diabetes and blood pressure data from this discharge note. "
            "Preserve all medical terminology exactly as written. "
            "Return only valid JSON with the exact schema: patient_id (use provided id), diabetes (type, status, a1c_values, glucose_values, medications), "
            "blood_pressure (hypertension_status, bp_readings, medications), abnormal_markers. "
            "Use empty string or empty array when a value is not found.\n\n"
            f"patient_id to use: {patient_id}\n\n"
            "Discharge note:\n"
            "---\n"
            f"{note_text}\n"
            "---\n\n"
            + EXPECTED_JSON_INSTRUCTION
        ),
        expected_output="Valid JSON object only, no markdown or extra text.",
        agent=agent,
    )
