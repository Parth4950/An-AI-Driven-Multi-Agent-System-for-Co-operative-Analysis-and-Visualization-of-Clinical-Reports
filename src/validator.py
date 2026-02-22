"""
Post-processing validation + normalization for clinical extraction output.
Runs AFTER the agent returns JSON. Ensures schema compliance: enums, abnormal_markers
filter, dedupe, no-inference rules. Does not change CrewAI or Gemini configuration.

Example BEFORE (agent output) vs AFTER (validator output) for one note:

BEFORE:
  "diabetes": {
    "type": "Type 2 diabetes mellitus",
    "status": "present",
    "a1c_values": ["7.2"],
    "glucose_values": ["150", "142"],
    "medications": ["metformin", "metformin"]
  },
  "blood_pressure": {
    "hypertension_status": "chronic",
    "bp_readings": ["140/90"],
    "medications": ["lisinopril"]
  },
  "abnormal_markers": ["Glucose-150*", "8.1", "K-4.2*", "normal", "Glucose-150*", ... 25 items]

AFTER:
  "diabetes": {
    "type": "Type 2 diabetes mellitus",
    "status": "",
    "a1c_values": ["7.2"],
    "glucose_values": ["142", "150"],
    "medications": ["metformin"]
  },
  "blood_pressure": {
    "hypertension_status": "",
    "bp_readings": [],
    "medications": []
  },
  "abnormal_markers": ["Glucose-150*", "K-4.2*"]

(Status "present" -> ""; hypertension_status "chronic" -> ""; no-inference clears BP;
abnormal_markers: regex keeps only label-number form, "8.1" and "normal" dropped,
duplicate removed, capped at 20; diabetes.medications deduped.)
"""

import logging
import re
from typing import Any

from config.schema import REQUIRED_BP_KEYS, REQUIRED_DIABETES_KEYS, REQUIRED_TOP_KEYS

logger = logging.getLogger(__name__)

# Allowed enum values (invalid → ""; no auto-correct)
DIABETES_TYPE_ALLOWED = frozenset({"Type 1", "Type 2", "Gestational", "Unspecified", ""})
DIABETES_STATUS_ALLOWED = frozenset({"active", "resolved", "history", ""})
HYPERTENSION_STATUS_ALLOWED = frozenset({
    "Hypertension",
    "Hypertensive urgency",
    "Hypertensive emergency",
    "",
})

# abnormal_markers: metabolic/endocrine only; format LAB-VALUE* or LAB-VALUE
ABNORMAL_MARKER_PATTERN = re.compile(r"^[A-Za-z%]+-[0-9.]+[*]?$")
ABNORMAL_MARKERS_MAX = 10
# FORBIDDEN: CBC, liver, electrolytes, kidney labs, ascitic, vitals, symptoms, diagnoses
ABNORMAL_MARKER_LAB_BLOCKLIST = frozenset(
    s.lower() for s in (
        "wbc", "rbc", "hgb", "hct", "mcv", "mch", "mchc", "plt", "rdw",
        "ast", "alt", "alkphos", "alk", "bilirubin", "sgot", "ldh", "ck",
        "ascitic", "fluid",
        "na", "k", "cl", "mg", "ca", "sodium", "potassium", "chloride", "magnesium", "calcium",
        "creatinine", "creat", "bun", "urea",
    )
)
ABNORMAL_MARKER_FORBIDDEN = frozenset(
    s.lower()
    for s in (
        "greater than", "less than", "pending", "negative", "positive",
        "unremarkable", "normal", "elevated", "decreased", "impression",
        "diagnosis", "history", "symptom", "vitals", "imaging", "narrative",
        "(", ")", "mg/dl", "mmol", "mmhg", "mg ", " units",
    )
)


def _ensure_dict(obj: Any) -> dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _ensure_str_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def _dedupe_list(arr: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in arr:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _enforce_diabetes_type(obj: dict[str, Any], verbose: bool) -> None:
    """Invalid diabetes.type → ""; log warning. No auto-correct."""
    d = obj.get("diabetes") or {}
    if not isinstance(d, dict):
        return
    raw = (d.get("type") or "").strip()
    if raw not in DIABETES_TYPE_ALLOWED:
        if verbose and raw:
            logger.warning(
                "validation: diabetes.type invalid %r -> \"\" (allowed: %s)",
                raw,
                sorted(DIABETES_TYPE_ALLOWED - {""}),
            )
        obj.setdefault("diabetes", {})["type"] = ""


def _enforce_diabetes_status(obj: dict[str, Any], verbose: bool) -> None:
    """Invalid diabetes.status → ""; log warning. No auto-correct."""
    d = obj.get("diabetes") or {}
    if not isinstance(d, dict):
        return
    raw = (d.get("status") or "").strip()
    if raw not in DIABETES_STATUS_ALLOWED:
        if verbose and raw:
            logger.warning(
                "validation: diabetes.status invalid %r -> \"\" (allowed: %s)",
                raw,
                sorted(DIABETES_STATUS_ALLOWED - {""}),
            )
        obj.setdefault("diabetes", {})["status"] = ""


def _enforce_hypertension_status(obj: dict[str, Any], verbose: bool) -> None:
    """Invalid hypertension_status → ""; log warning. No auto-correct."""
    bp = obj.get("blood_pressure") or {}
    if not isinstance(bp, dict):
        return
    raw = (bp.get("hypertension_status") or "").strip()
    if raw not in HYPERTENSION_STATUS_ALLOWED:
        if verbose and raw:
            logger.warning(
                "validation: blood_pressure.hypertension_status invalid %r -> \"\" (allowed: %s)",
                raw,
                sorted(HYPERTENSION_STATUS_ALLOWED - {""}),
            )
        obj.setdefault("blood_pressure", {})["hypertension_status"] = ""


def _filter_abnormal_markers_strict(arr: list[str], verbose: bool) -> list[str]:
    """Only tokens matching format Glucose-150* or %HbA1c-8.1*. Drop invalid; log warning. Cap 20."""
    out: list[str] = []
    seen: set[str] = set()
    for s in arr:
        t = str(s).strip()
        if not t or t in seen:
            continue
        if not ABNORMAL_MARKER_PATTERN.match(t):
            if verbose:
                logger.warning("validation: abnormal_markers invalid format (dropped): %r", t)
            continue
        lab_part = t.split("-")[0].lower().lstrip("%") if "-" in t else ""
        if lab_part in ABNORMAL_MARKER_LAB_BLOCKLIST:
            if verbose:
                logger.warning("validation: abnormal_markers blocklisted LAB (dropped): %r", t)
            continue
        lower = t.lower()
        if any(f in lower for f in ABNORMAL_MARKER_FORBIDDEN):
            if verbose:
                logger.warning("validation: abnormal_markers forbidden (dropped): %r", t)
            continue
        seen.add(t)
        out.append(t)
    if len(out) > ABNORMAL_MARKERS_MAX:
        if verbose:
            logger.warning(
                "validation: abnormal_markers capped %d -> %d",
                len(out),
                ABNORMAL_MARKERS_MAX,
            )
        out = out[:ABNORMAL_MARKERS_MAX]
    return out


def _enforce_no_inference(obj: dict[str, Any], verbose: bool) -> None:
    """If diabetes not explicitly stated (type and status empty), clear diabetes lists. Same for BP."""
    d = obj.get("diabetes") or {}
    if isinstance(d, dict):
        type_ = (d.get("type") or "").strip()
        status_ = (d.get("status") or "").strip()
        if not type_ and not status_:
            if any([d.get("a1c_values"), d.get("glucose_values"), d.get("medications")]):
                if verbose:
                    logger.info(
                        "validation: diabetes not stated -> clear a1c_values, glucose_values, medications"
                    )
            obj.setdefault("diabetes", {})["a1c_values"] = []
            obj.setdefault("diabetes", {})["glucose_values"] = []
            obj.setdefault("diabetes", {})["medications"] = []

    bp = obj.get("blood_pressure") or {}
    if isinstance(bp, dict):
        htn = (bp.get("hypertension_status") or "").strip()
        if not htn:
            if bp.get("bp_readings") or bp.get("medications"):
                if verbose:
                    logger.info(
                        "validation: hypertension not stated -> clear bp_readings, medications"
                    )
            obj.setdefault("blood_pressure", {})["bp_readings"] = []
            obj.setdefault("blood_pressure", {})["medications"] = []


def _dedupe_all_lists(obj: dict[str, Any], verbose: bool) -> None:
    """Deduplicate all list fields; log if any duplicate was removed."""
    for key in ("a1c_values", "glucose_values", "medications"):
        d = obj.get("diabetes")
        if isinstance(d, dict) and key in d and isinstance(d[key], list):
            arr = _ensure_str_list(d[key])
            deduped = _dedupe_list(arr)
            if verbose and len(deduped) < len(arr):
                logger.info("validation: diabetes.%s deduped %d -> %d", key, len(arr), len(deduped))
            obj["diabetes"][key] = sorted(deduped)

    for key in ("bp_readings", "medications"):
        bp = obj.get("blood_pressure")
        if isinstance(bp, dict) and key in bp and isinstance(bp[key], list):
            arr = _ensure_str_list(bp[key])
            deduped = _dedupe_list(arr)
            if verbose and len(deduped) < len(arr):
                logger.info(
                    "validation: blood_pressure.%s deduped %d -> %d",
                    key,
                    len(arr),
                    len(deduped),
                )
            obj["blood_pressure"][key] = sorted(deduped)

    # abnormal_markers already filtered/deduped/capped in _apply_abnormal_markers_filter; leave as-is


def _apply_abnormal_markers_filter(obj: dict[str, Any], verbose: bool) -> None:
    """Replace abnormal_markers with regex-filtered, deduped, capped list."""
    arr = obj.get("abnormal_markers")
    if not isinstance(arr, list):
        obj["abnormal_markers"] = []
        return
    obj["abnormal_markers"] = _filter_abnormal_markers_strict(
        _ensure_str_list(arr), verbose
    )
    obj["abnormal_markers"] = sorted(obj["abnormal_markers"])


def validate_extraction_output(obj: dict[str, Any], *, verbose: bool = False) -> dict[str, Any]:
    """
    Load agent JSON (dict), enforce enums, filter abnormal_markers, dedupe all lists,
    enforce no-inference (clear diabetes/BP sections when diagnosis not stated).
    Returns validated JSON only. Logs corrections when verbose=True.

    Schema shape assumed: patient_id, diabetes { type, status, a1c_values, glucose_values, medications },
    blood_pressure { hypertension_status, bp_readings, medications }, abnormal_markers.
    """
    if not isinstance(obj, dict):
        return obj

    # Ensure nested structure so we don't KeyError
    obj.setdefault("diabetes", _ensure_dict(obj.get("diabetes")))
    obj.setdefault("blood_pressure", _ensure_dict(obj.get("blood_pressure")))
    obj.setdefault("abnormal_markers", [])

    # 1) Enums in code: type, status (with active fallback), hypertension_status
    _enforce_diabetes_type(obj, verbose)
    _enforce_diabetes_status(obj, verbose)
    _enforce_hypertension_status(obj, verbose)

    # 2) No-inference: clear diabetes/BP lists when diagnosis not stated (must run after enum enforcement)
    _enforce_no_inference(obj, verbose)

    # 3) abnormal_markers: regex filter, dedupe, cap 20
    _apply_abnormal_markers_filter(obj, verbose)

    # 4) Dedupe all list fields (and sort for determinism)
    _dedupe_all_lists(obj, verbose)

    # 5) Ensure full schema (all keys present) so written JSON is always valid
    _ensure_full_schema(obj)

    return obj


def _ensure_full_schema(obj: dict[str, Any]) -> None:
    """Ensure every required key exists; missing → "" or []."""
    for key in REQUIRED_TOP_KEYS:
        if key not in obj:
            obj[key] = [] if key == "abnormal_markers" else "" if key == "patient_id" else {}
    obj["diabetes"] = _ensure_dict(obj.get("diabetes"))
    obj["blood_pressure"] = _ensure_dict(obj.get("blood_pressure"))
    for k in REQUIRED_DIABETES_KEYS:
        if k not in obj["diabetes"]:
            obj["diabetes"][k] = [] if k in ("a1c_values", "glucose_values", "medications") else ""
    for k in REQUIRED_BP_KEYS:
        if k not in obj["blood_pressure"]:
            obj["blood_pressure"][k] = [] if k in ("bp_readings", "medications") else ""
    if "abnormal_markers" not in obj or not isinstance(obj["abnormal_markers"], list):
        obj["abnormal_markers"] = []


def empty_schema(patient_id: str) -> dict[str, Any]:
    """Return full schema with empty values. Use for hard-scope skip and error payloads."""
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
