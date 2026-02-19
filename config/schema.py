"""
Expected JSON schema for clinical extraction output.
Used in agent instructions and for validation.
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

# Field semantics for the agent (preserve medical terminology)
DIABETES_TYPE_VALUES = (
    "Type 1 diabetes mellitus",
    "Type 2 diabetes mellitus",
    "Gestational diabetes",
    "Unspecified diabetes",
    "",
)
DIABETES_STATUS_VALUES = ("active", "historical", "family history", "")
MEDICATIONS_ITEM_FORMAT = "Name and exact dosage as in the note (e.g., 'metformin 500 mg PO BID', 'insulin glargine 12 units qHS')."
BP_READING_FORMAT = "Exact value as in note (e.g., '150/95 mmHg')."
A1C_FORMAT = "Exact value and unit as in note (e.g., '7.2%', '8.1%')."
GLUCOSE_FORMAT = "Exact value and unit as in note (e.g., '142 mg/dL')."
