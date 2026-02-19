"""
Run extraction crew on a single note; validate and return JSON.
"""

import json
import logging
import re
from typing import Any

from crewai import Crew

from src.tasks import extraction_task

logger = logging.getLogger(__name__)

# Optional: strip common markdown wrappers if model still adds them
JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})"
)


def _extract_raw_json(text: str) -> str:
    """Try to get raw JSON from agent output (no markdown)."""
    text = (text or "").strip()
    # If whole thing is JSON object
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try code block
    for match in JSON_BLOCK_PATTERN.finditer(text):
        group = match.group(1) or match.group(2)
        if group and group.strip().startswith("{"):
            return group.strip()
    return text


def _validate_schema(obj: Any) -> None:
    """Validate minimal structure; raise ValueError if invalid."""
    if not isinstance(obj, dict):
        raise ValueError("Root must be a JSON object")
    if "diabetes" not in obj or not isinstance(obj["diabetes"], dict):
        raise ValueError("Missing or invalid 'diabetes' object")
    if "blood_pressure" not in obj or not isinstance(obj["blood_pressure"], dict):
        raise ValueError("Missing or invalid 'blood_pressure' object")
    for key in ("patient_id", "diabetes", "blood_pressure", "abnormal_markers"):
        if key not in obj:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(obj["abnormal_markers"], list):
        raise ValueError("'abnormal_markers' must be an array")


def run_extraction(patient_id: str, note_text: str) -> dict[str, Any]:
    """
    Run the extractor agent on one note; return validated JSON dict.
    Raises on JSON parse or schema validation errors.
    """
    task = extraction_task(patient_id=patient_id, note_text=note_text)
    agent = task.agent
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()

    raw_output = getattr(result, "raw", None) or str(result)
    json_str = _extract_raw_json(raw_output)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s\nRaw (last 500 chars): %s", e, json_str[-500:])
        raise ValueError(f"Agent output is not valid JSON: {e}") from e

    _validate_schema(obj)
    return obj
