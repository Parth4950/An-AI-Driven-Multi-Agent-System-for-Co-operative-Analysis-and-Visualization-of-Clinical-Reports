"""
Expected JSON schema for clinical extraction output.
Used in agent instructions and for validation.
MUST match exactly: no extra keys, no missing keys.
"""

EXTRACTION_SCHEMA = """
{
  "patient_id": "",
  "diabetes": {
    "type": "",
    "status": "",
    "a1c_values": [],
    "glucose_values": [],
    "medications": []
  },
  "blood_pressure": {
    "hypertension_status": "",
    "bp_readings": [],
    "medications": []
  },
  "abnormal_markers": []
}
"""

# Exact keys required for strict validation (no extras allowed)
REQUIRED_TOP_KEYS = ("patient_id", "diabetes", "blood_pressure", "abnormal_markers")
REQUIRED_DIABETES_KEYS = ("type", "status", "a1c_values", "glucose_values", "medications")
REQUIRED_BP_KEYS = ("hypertension_status", "bp_readings", "medications")

# Field semantics for the agent (exact enum values)
DIABETES_TYPE_VALUES = ("Type 1", "Type 2", "Gestational", "Unspecified", "")
DIABETES_STATUS_VALUES = ("active", "history", "resolved", "")
MEDICATIONS_ITEM_FORMAT = "Name and exact dosage as in the note (e.g., 'metformin 500 mg PO BID', 'insulin glargine 12 units qHS')."
BP_READING_FORMAT = "Exact value as in note (e.g., '150/95 mmHg')."
A1C_FORMAT = "Exact value and unit as in note (e.g., '7.2%', '8.1%')."
GLUCOSE_FORMAT = "Exact value and unit as in note (e.g., '142 mg/dL')."
