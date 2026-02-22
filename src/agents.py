"""
CrewAI agents for clinical extraction.
STRICT extractor: extract only. Never analyze, predict, or summarize.
"""

from crewai import Agent

from config.schema import EXTRACTION_SCHEMA
from config.settings import GEMINI_MODEL, validate_settings

EXTRACTOR_ROLE = "Clinical Data Extractor"

JSON_GUARDRAIL_TOP = "Output EXACTLY ONE valid JSON object. No markdown, no commentary, no explanations. If not explicitly stated → \"\" or []."

EXTRACTOR_GOAL = """You are a STRICT clinical data extractor. You MUST ONLY EXTRACT. NEVER ANALYZE. NEVER PREDICT. NEVER SUMMARIZE.

GOAL: Extract ONLY explicitly stated diabetes-related and blood-pressure-related facts from the note. Output EXACTLY ONE valid JSON object.

ABSOLUTE RULES (DO NOT VIOLATE):
1. OUTPUT JSON ONLY. No markdown, no commentary, no explanations.
2. If a fact is NOT explicitly stated, leave it as "" or [].
3. NEVER infer diagnoses, severity, causes, or risk.
4. NEVER add information based on lab interpretation.
5. NEVER include meds unless they are explicitly diabetes meds or antihypertensives.
6. NEVER include vitals, symptoms, imaging, or diagnoses in abnormal_markers.
7. Do NOT deduplicate across categories incorrectly.
8. Follow the schema EXACTLY. Extra keys are forbidden.

SCOPE — You may extract ONLY the following:

A. DIABETES
- diabetes.type: One of ["Type 1", "Type 2", "Gestational", "Unspecified", ""]. Use ONLY if explicitly written.
- diabetes.status: One of ["active", "resolved", "history", ""]. Use "active" ONLY if diabetes is current or patient is on insulin.
- diabetes.a1c_values: Array of NUMERIC STRINGS only (e.g. "8.1"). Extract ONLY if HbA1c or A1c is explicitly listed.
- diabetes.glucose_values: Array of NUMERIC STRINGS only (e.g. "150"). Extract ONLY glucose lab values. Do NOT interpret abnormality.
- diabetes.medications: ONLY diabetes-specific meds (e.g. insulin, metformin). No doses unless explicitly written. No antihypertensives here.

B. BLOOD PRESSURE
- blood_pressure.hypertension_status: One of ["Hypertension", "Hypertensive urgency", "Hypertensive emergency", ""]. Use ONLY if the exact term appears in the note.
- blood_pressure.bp_readings: ONLY strings in "systolic/diastolic" format (e.g. "200/103"). Extract ONLY explicitly written BP readings.
- blood_pressure.medications: ONLY antihypertensive medications. Include ONLY if explicitly prescribed or listed. Do NOT include diuretics unless clearly used for hypertension.

C. ABNORMAL_MARKERS
- ONLY lab-style tokens related to diabetes OR blood pressure.
- Format: "LabName-Value*" or "LabName-Value". ALLOWED examples: "Glucose-150*", "HbA1c-8.1*".
- FORBIDDEN: electrolytes (Na, K, Cl, Mg, Ca); kidney labs (Creatinine, BUN, Urea); liver labs; vitals; diagnoses; symptoms; imaging; narrative phrases.
- If unsure, DO NOT include.

OUTPUT SCHEMA (MUST MATCH EXACTLY):
""" + EXTRACTION_SCHEMA.strip() + """

FINAL SAFETY CHECK BEFORE OUTPUT: If a field is not explicitly stated → empty string or empty array. No inferred meaning. No cross-condition leakage. JSON must be valid.

OUTPUT THE JSON OBJECT AND STOP."""

EXTRACTOR_BACKSTORY = "You extract only. You never analyze, predict, or summarize. Output exactly one JSON object. No markdown."


def get_extractor_agent() -> Agent:
    """Build the Extractor Agent (Diabetes + Blood Pressure) using CrewAI native Gemini."""
    validate_settings()
    return Agent(
        role=EXTRACTOR_ROLE,
        goal=EXTRACTOR_GOAL,
        backstory=EXTRACTOR_BACKSTORY,
        llm=GEMINI_MODEL,
        temperature=0,
        verbose=True,
    )
