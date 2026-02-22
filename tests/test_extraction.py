"""
Unit tests for extraction: JSON validity, empty-safe behavior, scope enforcement.
Run from project root: python -m unittest tests.test_extraction -v
"""

import json
import sys
import unittest
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import extraction

run_extraction = extraction.run_extraction
_empty_schema = extraction._empty_schema
_schema_validate_final = extraction._schema_validate_final
_parse_and_normalize = extraction._parse_and_normalize
_normalize_a1c_values = extraction._normalize_a1c_values
_normalize_glucose_values = extraction._normalize_glucose_values
_filter_abnormal_markers_lab_only = extraction._filter_abnormal_markers_lab_only
_filter_bp_readings_strict = extraction._filter_bp_readings_strict
_dedupe_list = extraction._dedupe_list
_error_payload = extraction._error_payload


class TestJsonValidity(unittest.TestCase):
    """JSON validity: output is always json.loads()-able and has all schema keys."""

    def test_run_extraction_dry_run_returns_valid_json(self):
        """run_extraction(DRY_RUN) returns dict that is json-serializable and has all schema keys."""
        orig = getattr(extraction, "DRY_RUN", False)
        extraction.DRY_RUN = True
        try:
            out = run_extraction("test_id", "some note")
        finally:
            extraction.DRY_RUN = orig
        self.assertIn("patient_id", out)
        self.assertIn("diabetes", out)
        self.assertIn("blood_pressure", out)
        self.assertIn("abnormal_markers", out)
        self.assertEqual(out["diabetes"]["type"], "")
        self.assertIsInstance(out["diabetes"]["a1c_values"], list)
        self.assertIsInstance(out["blood_pressure"]["bp_readings"], list)
        back = json.loads(json.dumps(out))
        self.assertEqual(back["patient_id"], out["patient_id"])

    def test_schema_validate_final_output_is_valid_json(self):
        """_schema_validate_final returns only schema keys; round-trips via json."""
        obj = {
            "patient_id": "p1",
            "diabetes": {"type": "Type 2", "status": "active", "a1c_values": ["8.1"], "glucose_values": ["150"], "medications": []},
            "blood_pressure": {"hypertension_status": "Hypertension", "bp_readings": ["120/80"], "medications": []},
            "abnormal_markers": ["Creat-1.6"],
        }
        result = _schema_validate_final(obj)
        parsed = json.loads(json.dumps(result))
        self.assertEqual(set(parsed.keys()), {"patient_id", "diabetes", "blood_pressure", "abnormal_markers"})
        self.assertEqual(set(parsed["diabetes"].keys()), {"type", "status", "a1c_values", "glucose_values", "medications"})
        self.assertEqual(set(parsed["blood_pressure"].keys()), {"hypertension_status", "bp_readings", "medications"})


class TestEmptySafe(unittest.TestCase):
    """Empty-safe behavior: missing keys get defaults; failures return full schema."""

    def test_empty_schema_has_all_keys(self):
        """_empty_schema(patient_id) has every required key; safe for json.dumps."""
        out = _empty_schema("empty_patient")
        self.assertEqual(out["patient_id"], "empty_patient")
        self.assertEqual(out["diabetes"]["type"], "")
        self.assertEqual(out["diabetes"]["a1c_values"], [])
        self.assertEqual(out["blood_pressure"]["bp_readings"], [])
        self.assertEqual(out["abnormal_markers"], [])
        json.loads(json.dumps(out))

    def test_error_payload_is_valid_json_with_all_keys(self):
        """Failures return EMPTY VALID JSON (full schema), not partial."""
        out = _error_payload("err_id", "something failed")
        self.assertEqual(out["patient_id"], "err_id")
        self.assertIn("type", out["diabetes"])
        self.assertIn("bp_readings", out["blood_pressure"])
        json.loads(json.dumps(out))

    def test_parse_and_normalize_minimal_json_returns_valid_schema(self):
        """Minimal/empty-like JSON still produces full schema with defaults."""
        raw = '{"patient_id":"","diabetes":{},"blood_pressure":{},"abnormal_markers":[]}'
        result = _parse_and_normalize(raw, "injected_id")
        self.assertIsNotNone(result)
        self.assertEqual(result["patient_id"], "injected_id")
        self.assertEqual(result["diabetes"]["type"], "")
        self.assertEqual(result["diabetes"]["a1c_values"], [])
        json.loads(json.dumps(result))


class TestScopeEnforcement(unittest.TestCase):
    """Scope enforcement: numeric only for a1c/glucose; bp SYS/DIA; abnormal_markers lab only."""

    def test_normalize_a1c_strips_units_and_labels(self):
        """a1c_values → numeric strings only (e.g. '8.1')."""
        out = _normalize_a1c_values(["8.1%", "HbA1c-7.2*", "9.0"])
        self.assertEqual(set(out), {"8.1", "7.2", "9.0"})
        self.assertEqual(len(out), 3)

    def test_normalize_glucose_strips_units_and_labels(self):
        """glucose_values → numeric strings only (e.g. '150')."""
        out = _normalize_glucose_values(["150*", "Glucose-138", "142 mg/dL"])
        self.assertIn("150", out)
        self.assertIn("138", out)
        self.assertIn("142", out)

    def test_filter_bp_readings_keeps_only_sys_dia(self):
        """bp_readings: only XX/XX or XXX/XX."""
        out = _filter_bp_readings_strict(["200/103", "120/80", "95 mmHg", "nope", "12/3"])
        self.assertIn("200/103", out)
        self.assertIn("120/80", out)
        self.assertNotIn("95 mmHg", out)
        self.assertNotIn("nope", out)

    def test_filter_abnormal_markers_drops_symptoms_and_narrative(self):
        """abnormal_markers: LAB VALUES ONLY; symptoms/vitals/narrative dropped."""
        arr = ["Glucose-150*", "tachycardic", "Creat-1.6*", "pain", "HbA1c-8.1*", "unremarkable"]
        out = _filter_abnormal_markers_lab_only(arr)
        self.assertNotIn("tachycardic", out)
        self.assertNotIn("pain", out)
        self.assertNotIn("unremarkable", out)
        self.assertTrue(len(out) >= 1)

    def test_dedupe_list_removes_duplicates(self):
        """Arrays are deduplicated."""
        out = _dedupe_list(["a", "b", "a", "c", "b"])
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out), {"a", "b", "c"})

    def test_parse_and_normalize_injects_patient_id(self):
        """patient_id is always overwritten with provided value."""
        raw = '{"patient_id":"wrong_id","diabetes":{"type":"","status":"","a1c_values":[],"glucose_values":[],"medications":[]},"blood_pressure":{"hypertension_status":"","bp_readings":[],"medications":[]},"abnormal_markers":[]}'
        result = _parse_and_normalize(raw, "correct_id")
        self.assertIsNotNone(result)
        self.assertEqual(result["patient_id"], "correct_id")


if __name__ == "__main__":
    unittest.main()
