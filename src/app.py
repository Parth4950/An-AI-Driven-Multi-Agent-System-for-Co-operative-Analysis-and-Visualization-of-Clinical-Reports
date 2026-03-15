"""
Clinical Risk Analysis Dashboard — AI-powered diabetes and hypertension risk from clinical notes.
Presents pipeline outputs as readable text, tables, charts, and visual indicators (no raw JSON).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.orchestrator import run_pipeline

st.set_page_config(page_title="Clinical Risk Analysis Dashboard", layout="wide")

# ——— Section 1: Title ———
st.title("Clinical Risk Analysis Dashboard")
st.markdown("*AI‑powered analysis of diabetes and hypertension risk from clinical notes.*")
st.divider()

# ——— Section 2: Input ———
st.subheader("Input Clinical Note")
clinical_note = st.text_area(
    "Paste clinical or discharge note below",
    height=280,
    placeholder="Paste discharge summary or clinical note text here...",
    label_visibility="collapsed",
)
run_clicked = st.button("Analyze", type="primary", use_container_width=False)

if run_clicked:
    note = (clinical_note or "").strip()
    if not note:
        st.warning("Please enter a clinical note before running analysis.")
    else:
        with st.spinner("Running pipeline (Extraction → Risk Analysis → Summary → Visualization)…"):
            try:
                result = run_pipeline(note)
                st.session_state["last_result"] = result
                st.success("Analysis complete.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.exception(e)
                if "last_result" in st.session_state:
                    del st.session_state["last_result"]

# ——— Sections 3–9: Only when we have a result (no JSON) ———
if "last_result" not in st.session_state:
    st.info("Paste a clinical note and click **Analyze** to see results.")
    st.stop()

result = st.session_state["last_result"]
summary = result.get("summary") or {}
viz = (result.get("visualizations") or {}).get("visualizations") or {}

# ——— Section 3: Doctor Summary ———
st.subheader("Doctor Summary")
doctor_text = (summary.get("doctor_summary") or "").strip()
if doctor_text:
    st.markdown(doctor_text)
else:
    st.caption("No doctor summary available.")

st.divider()

# ——— Section 4: Patient Summary ———
st.subheader("Patient Summary")
patient_text = (summary.get("patient_summary") or "").strip()
if patient_text:
    st.markdown(patient_text)
else:
    st.caption("No patient summary available.")

st.divider()

# ——— Section 5: Risk Severity Indicator ———
st.subheader("Risk Severity")
severity = viz.get("severity_indicator") or {}
level = (severity.get("level") or "Low").strip()
color = (severity.get("color") or "green").strip().lower()
emoji = {"red": "🔴", "orange": "🟠", "green": "🟢"}.get(color, "🟢")
label = f"{emoji} {level.upper()} RISK"
if color == "red":
    st.error(label)
elif color == "orange":
    st.warning(label)
else:
    st.success(label)

st.divider()

# ——— Section 6: Risk Score Bar Chart ———
st.subheader("Clinical Risk Scores")
scores = viz.get("risk_scores") or {}
d_score = float(scores.get("diabetes_score") or 0)
h_score = float(scores.get("hypertension_score") or 0)
chart_df = pd.DataFrame(
    {"Risk Score": [d_score, h_score]},
    index=["Diabetes", "Hypertension"],
)
st.bar_chart(chart_df)

st.divider()

# ——— Section 7: Clinical Evidence Table ———
st.subheader("Clinical Evidence")
evidence = viz.get("evidence_chart") or []
if isinstance(evidence, list) and evidence:
    rows = []
    for item in evidence:
        if isinstance(item, dict):
            rows.append({
                "Measurement": str(item.get("label") or "").strip() or "—",
                "Value": str(item.get("value") or "").strip() or "—",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No evidence data available.")
else:
    st.caption("No evidence data available.")

st.divider()

# ——— Section 8: Key Clinical Flags ———
st.subheader("Key Clinical Flags")
flags = summary.get("key_flags") or []
if isinstance(flags, list) and flags:
    for f in flags:
        text = str(f).strip()
        if text:
            st.markdown(f"• {text}")
else:
    st.caption("No key flags identified.")

st.divider()

# ——— Section 9: Data Gaps ———
gaps = summary.get("data_gaps") or []
if isinstance(gaps, list) and gaps:
    gap_texts = [str(g).strip() for g in gaps if str(g).strip()]
    if gap_texts:
        st.subheader("Data Gaps")
        with st.container():
            st.warning("The following clinically relevant information was not found in the note:")
            for g in gap_texts:
                st.markdown(f"• {g}")
