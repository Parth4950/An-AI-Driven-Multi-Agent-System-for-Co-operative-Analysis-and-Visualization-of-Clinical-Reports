"""
Agent 5: Orchestrator for the clinical AI multi-agent system.
Runs the full pipeline in order: Extraction → Risk Analysis → Summary → Visualization.
Only coordinates existing agents; does not modify their logic.
"""

from typing import Any

from src.extraction import run_extraction
from src.risk_analysis import run_risk_analysis
from src.summarizer import run_summarization
from src.visualizer import build_visualization


def _extract_clinical_features(input_text: str, patient_id: str = "pipeline") -> dict[str, Any]:
    """Wrapper for run_extraction; single-arg API for orchestrator."""
    return run_extraction(patient_id=patient_id, note_text=input_text)


def _analyze_risk(extraction_output: dict[str, Any]) -> dict[str, Any]:
    """Wrapper for run_risk_analysis."""
    return run_risk_analysis(extraction_output)


def _generate_summary(
    extraction_output: dict[str, Any],
    risk_output: dict[str, Any],
) -> dict[str, Any]:
    """Wrapper for run_summarization."""
    return run_summarization(extraction_output, risk_output)


def run_pipeline(
    input_text: str,
    patient_id: str = "pipeline",
) -> dict[str, Any]:
    """
    Run the full clinical pipeline on one note: Agent 1 → 2 → 3 → 4.
    Returns a single JSON-shaped dict with extraction, risk_analysis, summary, visualizations.
    """
    print("Agent 1: Clinical Extraction")
    extraction_output = _extract_clinical_features(input_text, patient_id)

    print("Agent 2: Risk Analysis")
    risk_output = _analyze_risk(extraction_output)

    print("Agent 3: Clinical Summarizer")
    summary_output = _generate_summary(extraction_output, risk_output)

    print("Agent 4: Visualization Agent")
    visualization_output = build_visualization(
        extraction_output,
        risk_output,
        summary_output,
    )

    return {
        "extraction": extraction_output,
        "risk_analysis": risk_output,
        "summary": summary_output,
        "visualizations": visualization_output,
    }
