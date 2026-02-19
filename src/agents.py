"""
CrewAI agents for clinical extraction.
Uses CrewAI native Gemini (model string + GEMINI_API_KEY from env).
"""

from crewai import Agent

from config.schema import EXTRACTION_SCHEMA
from config.settings import GEMINI_MODEL, GEMINI_TEMPERATURE, validate_settings

EXTRACTOR_ROLE = "Clinical Note Extractor"
EXTRACTOR_GOAL = """Extract structured diabetes and blood pressure data from discharge notes.
Output ONLY valid JSON matching the exact schema provided. No markdown, no code fences, no explanation, no text outside the JSON.
Preserve all medical terminology exactly as written in the note (diagnoses, medications, lab names, units)."""

EXTRACTOR_BACKSTORY = """You are a clinical data extractor for attending physicians.
You extract only: (1) Diabetes Mellitus — type (one of: Type 1 diabetes mellitus, Type 2 diabetes mellitus, Gestational diabetes, Unspecified diabetes, or empty);
status (active, historical, family history, or empty); A1C values in exact format from the note; glucose levels; insulin medications with exact dosages; oral hypoglycemics with exact dosages.
(2) Blood pressure — exact BP readings (e.g. 150/95 mmHg); hypertension diagnosis; hypertensive urgency/emergency if mentioned; antihypertensive medications with exact dosages.
(3) Any abnormal markers mentioned. Use empty strings or empty arrays when not found. Preserve exact wording and units from the note."""


def get_extractor_agent() -> Agent:
    """Build the single Extractor Agent (Diabetes + Blood Pressure) using CrewAI native Gemini."""
    validate_settings()
    return Agent(
        role=EXTRACTOR_ROLE,
        goal=EXTRACTOR_GOAL,
        backstory=EXTRACTOR_BACKSTORY,
        llm=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        verbose=True,
    )
