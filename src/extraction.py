"""
Run extraction crew on a single note; validate and return JSON.
Guarantees machine-parseable output: inject patient_id after response, JSON repair with optional LLM retry, schema filters, empty defaults, dedupe. Invalid after repair → skip patient (error payload), do not crash.

NO-INFERENCE / POST-PROCESSING:
- If diabetes type and status are both empty → clear a1c_values, glucose_values, diabetes medications.
- If hypertension_status is empty → clear bp_readings, blood_pressure medications.
- Medications: lowercase, deduplicated. Arrays sorted for determinism.

BEFORE vs AFTER example (failing note: diabetes not explicitly stated, raw/labeled values):

BEFORE (model output):
  "diabetes": {
    "type": "",
    "status": "",
    "a1c_values": ["%HbA1c-8.1*", "7.2"],
    "glucose_values": ["Glucose-150*", "142"],
    "medications": ["Metformin 500mg", "INSULIN GLARGINE"]
  },
  "blood_pressure": {
    "hypertension_status": "chronic",
    "bp_readings": ["200/103", "elevated 180/95"],
    "medications": ["Lisinopril 10mg"]
  }

AFTER (post-processor):
  "diabetes": {
    "type": "",
    "status": "",
    "a1c_values": [],
    "glucose_values": [],
    "medications": []
  },
  "blood_pressure": {
    "hypertension_status": "chronic",
    "bp_readings": ["200/103"],
    "medications": ["lisinopril 10mg"]
  }

(Diabetes labs/meds cleared because type and status empty; BP readings filtered to SYS/DIA only; meds lowercased; arrays sorted.)
"""

import json
import logging
import random
import re
import time
from typing import Any

from crewai import Crew

from config.schema import (
    REQUIRED_BP_KEYS,
    REQUIRED_DIABETES_KEYS,
    REQUIRED_TOP_KEYS,
)
from config.settings import DRY_RUN, MAX_429_RETRIES, RATE_LIMIT_SEC
from src.tasks import extraction_task, repair_json_task

logger = logging.getLogger(__name__)

RETRY_AFTER_PATTERN = re.compile(r"[Rr]etry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})")
TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
COMMENT_PATTERN = re.compile(r'//[^\n]*|#[^\n]*')
# bp_readings: extract pure SYS/DIA (strip "BP=", units, text)
BP_READING_PATTERN = re.compile(r"^\d{2,3}/\d{2,3}$")
BP_READING_EXTRACT_PATTERN = re.compile(r"\b(\d{2,3}/\d{2,3})\b")
# Extract numeric for a1c (e.g. "8.1" from "8.1%", "HbA1c-8.1*")
A1C_NUMERIC_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
# Extract numeric for glucose (e.g. "150" from "150*", "Glucose-150", "150 mg/dL")
GLUCOSE_NUMERIC_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

ABNORMAL_MARKER_MAX_LEN = 60
# Exclude symptoms, vitals, narrative (not lab values)
ABNORMAL_MARKER_BLOCKLIST = frozenset(
    s.lower() for s in (
        "tachy", "tachycardic", "brady", "pain", "sob", "dyspnea", "doe", "fatigue",
        "wheezing", "cough", "nausea", "dizziness", "unremarkable", "non-contributory",
        "denies", "impression", "diagnosis", "history", "normal", "elevated", "decreased",
    )
)

_last_request_time: float = 0.0

# Hard-scope gate: if note contains none of these, skip LLM and return empty schema
SCOPE_DIABETES_PHRASES = ("diabetes", "diabetes mellitus")
SCOPE_HTN_PHRASES = ("hypertension", "hypertensive")


def _note_has_scope(note_text: str) -> bool:
    """True if note explicitly mentions diabetes or hypertension (so LLM extraction is in scope)."""
    t = (note_text or "").lower()
    if any(p in t for p in SCOPE_DIABETES_PHRASES):
        return True
    if any(p in t for p in SCOPE_HTN_PHRASES):
        return True
    return False


def _rate_limit() -> None:
    global _last_request_time
    now = time.monotonic()
    if now - _last_request_time < RATE_LIMIT_SEC:
        time.sleep(RATE_LIMIT_SEC - (now - _last_request_time))
    _last_request_time = time.monotonic()


def _mock_extraction(patient_id: str) -> dict[str, Any]:
    return {
        "patient_id": patient_id,
        "diabetes": {
            "type": "",
            "status": "",
            "a1c_values": [],
            "glucose_values": [],
            "medications": [],
        },
        "blood_pressure": {
            "hypertension_status": "",
            "bp_readings": [],
            "medications": [],
        },
        "abnormal_markers": [],
    }


def _extract_raw_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    for match in JSON_BLOCK_PATTERN.finditer(text):
        group = match.group(1) or match.group(2)
        if group and group.strip().startswith("{"):
            return group.strip()
    return text


def _repair_json_string(s: str) -> str:
    s = s.strip()
    s = TRAILING_COMMA_PATTERN.sub(r"\1", s)
    s = COMMENT_PATTERN.sub("", s)
    return s


def _run_repair_once(broken_json: str) -> str:
    """Call LLM once to fix invalid JSON. Returns raw output string."""
    task = repair_json_task(broken_json)
    agent = task.agent
    crew = Crew(agents=[agent], tasks=[task], memory=False, cache=False)
    result = crew.kickoff()
    return getattr(result, "raw", None) or str(result)


def _ensure_dict(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _ensure_str(val: Any) -> str:
    return val if isinstance(val, str) else ""


def _ensure_str_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def _force_empty_defaults(obj: dict[str, Any]) -> None:
    """Ensure all required keys exist; missing → "" or []."""
    for key in REQUIRED_TOP_KEYS:
        if key not in obj:
            if key == "abnormal_markers":
                obj[key] = []
            elif key == "patient_id":
                obj[key] = ""
            else:
                obj[key] = {}
    obj["diabetes"] = _ensure_dict(obj.get("diabetes"))
    obj["blood_pressure"] = _ensure_dict(obj.get("blood_pressure"))
    for k in REQUIRED_DIABETES_KEYS:
        if k not in obj["diabetes"]:
            obj["diabetes"][k] = [] if k in ("a1c_values", "glucose_values", "medications") else ""
    for k in REQUIRED_BP_KEYS:
        if k not in obj["blood_pressure"]:
            obj["blood_pressure"][k] = [] if k in ("bp_readings", "medications") else ""
    for k in ("type", "status", "a1c_values", "glucose_values", "medications"):
        if k in obj["diabetes"]:
            if k in ("type", "status"):
                obj["diabetes"][k] = _ensure_str(obj["diabetes"][k])
            else:
                obj["diabetes"][k] = _ensure_str_list(obj["diabetes"][k])
    for k in ("hypertension_status", "bp_readings", "medications"):
        if k in obj["blood_pressure"]:
            if k == "hypertension_status":
                obj["blood_pressure"][k] = _ensure_str(obj["blood_pressure"][k])
            else:
                obj["blood_pressure"][k] = _ensure_str_list(obj["blood_pressure"][k])
    if "abnormal_markers" in obj:
        obj["abnormal_markers"] = _ensure_str_list(obj["abnormal_markers"])


def _inject_patient_id(obj: dict[str, Any], patient_id: str) -> None:
    """Overwrite patient_id with the known value. Never use model-provided patient_id."""
    obj["patient_id"] = patient_id


def _extract_numeric_strings(arr: list[str], pattern: re.Pattern[str], use_last_match: bool = True) -> list[str]:
    """From each item extract numeric match (last match per item to skip label digits e.g. HbA1c); dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for s in arr:
        s = str(s).strip()
        matches = pattern.findall(s)
        if not matches:
            continue
        val = matches[-1] if use_last_match else matches[0]
        if val not in seen:
            seen.add(val)
            out.append(val)
    return out


def _normalize_a1c_values(arr: list[str]) -> list[str]:
    """Numeric strings only (e.g. '8.1'). Strip %, *, labels, units."""
    return _extract_numeric_strings(arr, A1C_NUMERIC_PATTERN)


def _normalize_glucose_values(arr: list[str]) -> list[str]:
    """Numeric strings only (e.g. '150'). Strip *, labels, units."""
    return _extract_numeric_strings(arr, GLUCOSE_NUMERIC_PATTERN)


def _dedupe_list(arr: list[str]) -> list[str]:
    """Order-preserving dedupe."""
    return list(dict.fromkeys(x.strip() for x in arr if str(x).strip()))


def _filter_abnormal_markers_lab_only(arr: list[str]) -> list[str]:
    """LAB VALUES ONLY. Drop symptoms, vitals, narrative, long text, anything with spaces."""
    out: list[str] = []
    for s in arr:
        s = str(s).strip()
        if len(s) > ABNORMAL_MARKER_MAX_LEN or " " in s:
            continue
        low = s.lower()
        if any(block in low for block in ABNORMAL_MARKER_BLOCKLIST):
            continue
        out.append(s)
    return out


def _filter_abnormal_markers_strict(arr: list[str]) -> list[str]:
    """Remove entries longer than 60 chars or containing spaces (keep lab-style tokens only)."""
    return _filter_abnormal_markers_lab_only(arr)


def _filter_bp_readings_strict(arr: list[str]) -> list[str]:
    """Extract pure \\d{2,3}/\\d{2,3} from each item (strip BP=, spaces, units); dedupe."""
    out: list[str] = []
    seen: set[str] = set()
    for s in arr:
        s = str(s).strip()
        if BP_READING_PATTERN.match(s):
            if s not in seen:
                seen.add(s)
                out.append(s)
        else:
            for m in BP_READING_EXTRACT_PATTERN.findall(s):
                if m not in seen:
                    seen.add(m)
                    out.append(m)
    return out


def _dedupe_medications(arr: list[str]) -> list[str]:
    """Deduplicate medications."""
    return _dedupe_list(arr)


def _enforce_no_inference(obj: dict[str, Any]) -> None:
    """
    Enforce NO INFERENCE: clear diabetes labs/meds if diabetes not explicitly stated;
    clear BP readings/meds if hypertension not explicitly stated.
    """
    d = obj.get("diabetes") or {}
    type_ = (d.get("type") or "").strip()
    status_ = (d.get("status") or "").strip()
    diabetes_stated = bool(type_ or status_)
    if not diabetes_stated:
        obj["diabetes"]["a1c_values"] = []
        obj["diabetes"]["glucose_values"] = []
        obj["diabetes"]["medications"] = []

    bp = obj.get("blood_pressure") or {}
    htn = (bp.get("hypertension_status") or "").strip()
    if not htn:
        obj["blood_pressure"]["bp_readings"] = []
        obj["blood_pressure"]["medications"] = []


def _normalize_medications_generic_lower(arr: list[str]) -> list[str]:
    """Lowercase, dedupe. Generic names only (lowercase)."""
    out = []
    seen: set[str] = set()
    for s in arr:
        s = str(s).strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _sort_for_determinism(obj: dict[str, Any]) -> None:
    """Sort all arrays so identical content produces identical JSON."""
    for key in ("a1c_values", "glucose_values", "medications"):
        if isinstance(obj.get("diabetes"), dict) and key in obj["diabetes"]:
            obj["diabetes"][key] = sorted(obj["diabetes"][key])
    for key in ("bp_readings", "medications"):
        if isinstance(obj.get("blood_pressure"), dict) and key in obj["blood_pressure"]:
            obj["blood_pressure"][key] = sorted(obj["blood_pressure"][key])
    if "abnormal_markers" in obj and isinstance(obj["abnormal_markers"], list):
        obj["abnormal_markers"] = sorted(obj["abnormal_markers"])


def _schema_validate_final(obj: dict[str, Any]) -> dict[str, Any]:
    """Enforce schema BEFORE return: only allowed keys, correct types. Missing → "" or []. No extra keys."""
    out: dict[str, Any] = {
        "patient_id": _ensure_str(obj.get("patient_id")),
        "diabetes": {},
        "blood_pressure": {},
        "abnormal_markers": [],
    }
    d = _ensure_dict(obj.get("diabetes"))
    out["diabetes"] = {
        "type": _ensure_str(d.get("type")),
        "status": _ensure_str(d.get("status")),
        "a1c_values": _ensure_str_list(d.get("a1c_values")) if isinstance(d.get("a1c_values"), list) else [],
        "glucose_values": _ensure_str_list(d.get("glucose_values")) if isinstance(d.get("glucose_values"), list) else [],
        "medications": _ensure_str_list(d.get("medications")) if isinstance(d.get("medications"), list) else [],
    }
    bp = _ensure_dict(obj.get("blood_pressure"))
    out["blood_pressure"] = {
        "hypertension_status": _ensure_str(bp.get("hypertension_status")),
        "bp_readings": _ensure_str_list(bp.get("bp_readings")) if isinstance(bp.get("bp_readings"), list) else [],
        "medications": _ensure_str_list(bp.get("medications")) if isinstance(bp.get("medications"), list) else [],
    }
    am = obj.get("abnormal_markers")
    out["abnormal_markers"] = _ensure_str_list(am) if isinstance(am, list) else []
    return out


def _empty_schema(patient_id: str) -> dict[str, Any]:
    """Full schema with empty values. Use for errors so output is always valid schema."""
    return {
        "patient_id": patient_id,
        "diabetes": {
            "type": "",
            "status": "",
            "a1c_values": [],
            "glucose_values": [],
            "medications": [],
        },
        "blood_pressure": {
            "hypertension_status": "",
            "bp_readings": [],
            "medications": [],
        },
        "abnormal_markers": [],
    }


def _error_payload(patient_id: str, message: str) -> dict[str, Any]:
    """Return EMPTY VALID JSON (full schema), not partial output."""
    out = _empty_schema(patient_id)
    out["error"] = message
    return out


def _is_429(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()


def _has_retry_after(err: Exception) -> bool:
    return bool(RETRY_AFTER_PATTERN.search(str(err)))


def _parse_and_normalize(raw_json_str: str, patient_id: str) -> dict[str, Any] | None:
    """
    Parse JSON (with optional string repair), force defaults, inject patient_id, apply schema filters, dedupe medications.
    Returns None if parsing fails (caller should not crash).
    """
    s = _extract_raw_json(raw_json_str)
    obj: dict[str, Any] | None = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        s = _repair_json_string(s)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    _force_empty_defaults(obj)
    _inject_patient_id(obj, patient_id)
    # Numeric strings only for a1c and glucose (e.g. "8.1", "150")
    obj["diabetes"]["a1c_values"] = _normalize_a1c_values(obj["diabetes"]["a1c_values"])
    obj["diabetes"]["glucose_values"] = _normalize_glucose_values(obj["diabetes"]["glucose_values"])
    # BP: SYS/DIA only; dedupe
    obj["blood_pressure"]["bp_readings"] = _dedupe_list(
        _filter_bp_readings_strict(obj["blood_pressure"]["bp_readings"])
    )
    # abnormal_markers: LAB VALUES ONLY; dedupe
    obj["abnormal_markers"] = _dedupe_list(
        _filter_abnormal_markers_lab_only(obj["abnormal_markers"])
    )
    obj["diabetes"]["medications"] = _dedupe_medications(obj["diabetes"]["medications"])
    obj["blood_pressure"]["medications"] = _dedupe_medications(obj["blood_pressure"]["medications"])
    # No-inference: clear diabetes labs/meds if diabetes not stated; clear BP if HTN not stated
    _enforce_no_inference(obj)
    # Medications: generic names only, lowercase
    obj["diabetes"]["medications"] = _normalize_medications_generic_lower(obj["diabetes"]["medications"])
    obj["blood_pressure"]["medications"] = _normalize_medications_generic_lower(obj["blood_pressure"]["medications"])
    # Determinism: sort all arrays so identical note → identical JSON
    _sort_for_determinism(obj)
    # Final: enforce schema only (no extra keys), validate types
    return _schema_validate_final(obj)


def run_extraction(patient_id: str, note_text: str) -> dict[str, Any]:
    """
    Run extractor on one note. Inject patient_id after response. If JSON invalid, try LLM repair once; if still invalid, log and return error payload (do not crash).
    """
    if DRY_RUN:
        logger.info("DRY_RUN: patient_id=%s -> mock extraction (no API call)", patient_id)
        return _mock_extraction(patient_id)

    # Hard-scope gate: skip LLM entirely if note does not mention diabetes or hypertension
    if not _note_has_scope(note_text):
        logger.info("patient_id=%s: no diabetes/hypertension in note -> empty schema (no LLM call)", patient_id)
        return _empty_schema(patient_id)

    _rate_limit()

    task = extraction_task(patient_id=patient_id, note_text=note_text)
    agent = task.agent
    crew = Crew(agents=[agent], tasks=[task], memory=False, cache=False)

    last_429_error: Exception | None = None
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            result = crew.kickoff()
            break
        except Exception as e:
            if not _is_429(e):
                raise
            if not _has_retry_after(e):
                logger.error("Gemini free-tier quota exhausted. patient_id=%s", patient_id)
                return _error_payload(patient_id, "Gemini free-tier quota exhausted. Enable billing or use DRY_RUN.")
            if attempt >= MAX_429_RETRIES:
                logger.error("Gemini 429 after %d retries. patient_id=%s", MAX_429_RETRIES, patient_id)
                return _error_payload(patient_id, f"Resource exhausted (429) after {MAX_429_RETRIES} retries.")
            match = RETRY_AFTER_PATTERN.search(str(e))
            base_wait = float(match.group(1)) if match else 30.0
            base_wait = min(base_wait, 120.0)
            wait_sec = min(base_wait * (2**attempt) + random.uniform(0, 2.0), 120.0)
            logger.warning("Gemini 429, retry %d/%d in %.0fs. patient_id=%s", attempt + 1, MAX_429_RETRIES, wait_sec, patient_id)
            time.sleep(wait_sec)
            last_429_error = e
            continue
    else:
        if last_429_error is not None:
            return _error_payload(patient_id, f"Resource exhausted (429) after {MAX_429_RETRIES} retries.")
        raise RuntimeError("Unexpected loop exit in run_extraction")

    raw_output = getattr(result, "raw", None) or str(result)
    json_str = _extract_raw_json(raw_output)

    # First attempt: parse (with string repair) and normalize
    obj = _parse_and_normalize(json_str, patient_id)

    # Second attempt: LLM repair once if parse failed
    if obj is None:
        logger.warning("JSON parse failed for patient_id=%s; attempting LLM repair once.", patient_id)
        try:
            _rate_limit()
            repair_output = _run_repair_once(json_str)
            repair_str = _extract_raw_json(repair_output)
            obj = _parse_and_normalize(repair_str, patient_id)
        except Exception as e:
            logger.error("LLM repair failed for patient_id=%s: %s", patient_id, e)
            obj = None

    if obj is None:
        logger.error("Invalid JSON for patient_id=%s after repair; skipping patient (no crash).", patient_id)
        return _error_payload(patient_id, "Invalid JSON after repair; extraction skipped.")

    return obj
