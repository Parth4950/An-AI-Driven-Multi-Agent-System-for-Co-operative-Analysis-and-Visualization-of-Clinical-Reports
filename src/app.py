"""
Clinical Risk Analysis Dashboard — AI-powered diabetes and hypertension risk from clinical notes.
Presents pipeline outputs as readable text, tables, charts, and visual indicators (no raw JSON).
Includes report export: PDF and JSON.
"""

import io
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env so TESSERACT_CMD and other vars are available (e.g. for document_parser)
import os
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")
# Disable CrewAI telemetry to avoid "signal only works in main thread" when run under Streamlit
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import pandas as pd
import streamlit as st
import altair as alt
import requests
from fpdf import FPDF

from src.document_parser import extract_text_from_file, is_supported_file

# FastAPI backend base URL
API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


def _backend_error_message(exc: Exception) -> str:
    """Create a user-friendly backend connectivity message."""
    return (
        f"Could not connect to FastAPI backend at {API_BASE_URL}. "
        "Start the API server with: `uvicorn backend.main:app --reload`."
    )


def _api_post_analyze(patient_id: str, text: str) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/analyze/",
            json={"patient_id": patient_id, "text": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(_backend_error_message(exc)) from exc


def _api_get_patients() -> list[str]:
    try:
        resp = requests.get(f"{API_BASE_URL}/patients/", timeout=30)
        resp.raise_for_status()
        return resp.json().get("patient_ids") or []
    except requests.RequestException as exc:
        raise RuntimeError(_backend_error_message(exc)) from exc


def _api_get_patient_history(patient_id: str) -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/patients/{patient_id}", timeout=30)
        resp.raise_for_status()
        return resp.json() or []
    except requests.RequestException as exc:
        raise RuntimeError(_backend_error_message(exc)) from exc


def _api_get_patient_trends(patient_id: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/patients/{patient_id}/trends", timeout=30)
        resp.raise_for_status()
        return resp.json() or {}
    except requests.RequestException as exc:
        raise RuntimeError(_backend_error_message(exc)) from exc


def _build_report_data(result):
    """Build structured report dict from pipeline outputs (for JSON export and PDF)."""
    extraction = result.get("extraction") or {}
    summary = result.get("summary") or {}
    risk = result.get("risk_analysis") or {}
    viz = (result.get("visualizations") or {}).get("visualizations") or {}
    patient_id = (extraction.get("patient_id") or "").strip() or "Not specified"
    doctor_summary = (summary.get("doctor_summary") or "").strip()
    patient_summary = (summary.get("patient_summary") or "").strip()
    severity = (viz.get("severity_indicator") or {}).get("level") or "Low"
    scores = viz.get("risk_scores") or {}
    hypertension_score = float(scores.get("hypertension_score") or 0)
    diabetes_score = float(scores.get("diabetes_score") or 0)
    evidence = viz.get("evidence_chart") or []
    evidence_rows = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                evidence_rows.append({
                    "measurement": str(item.get("label") or "").strip() or "—",
                    "value": str(item.get("value") or "").strip() or "—",
                })
    key_flags = summary.get("key_flags") or []
    key_flags = [str(f).strip() for f in key_flags if str(f).strip()]
    data_gaps = summary.get("data_gaps") or []
    data_gaps = [str(g).strip() for g in data_gaps if str(g).strip()]
    return {
        "report_title": "Clinical Risk Analysis Report",
        "generated_by": "AI Clinical Analyzer",
        "patient_id": patient_id,
        "doctor_summary": doctor_summary,
        "patient_summary": patient_summary,
        "risk_severity": severity,
        "hypertension_score": hypertension_score,
        "diabetes_score": diabetes_score,
        "clinical_evidence": evidence_rows,
        "key_flags": key_flags,
        "data_gaps": data_gaps,
    }


def _pdf_safe_text(s):
    """Ensure text fits PDF: replace problematic chars and limit line length for fpdf."""
    if not s or not isinstance(s, str):
        return ""
    s = s.replace("\x00", "").strip()
    return s[:8000] if len(s) > 8000 else s


def _build_pdf_bytes(report):
    """Generate PDF report bytes from structured report data."""
    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    w = pdf.epw  # effective width (page width minus margins)
    if w <= 0:
        w = 170  # fallback A4 approximate
    pdf.set_font("Helvetica", "B", size=16)
    pdf.multi_cell(w, 10, _pdf_safe_text(report["report_title"]), align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(w, 6, _pdf_safe_text(f"Generated by {report['generated_by']}"), align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(w, 6, _pdf_safe_text(f"Patient ID: {report['patient_id']}"), ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(w, 8, "Doctor Summary", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(w, 6, _pdf_safe_text(report["doctor_summary"] or "No doctor summary available."))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(w, 8, "Patient Explanation", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(w, 6, _pdf_safe_text(report["patient_summary"] or "No patient summary available."))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(w, 8, "Risk Severity", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(w, 6, f"Level: {report['risk_severity']}", ln=True)
    pdf.cell(w, 6, f"Hypertension Risk Score: {report['hypertension_score']:.2f}", ln=True)
    pdf.cell(w, 6, f"Diabetes Risk Score: {report['diabetes_score']:.2f}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(w, 8, "Clinical Evidence", ln=True)
    pdf.set_font("Helvetica", size=10)
    if report["clinical_evidence"]:
        with pdf.table() as table:
            row = table.row()
            row.cell("Measurement")
            row.cell("Value")
            for r in report["clinical_evidence"]:
                row = table.row()
                row.cell(_pdf_safe_text(r.get("measurement", ""))[:80])
                row.cell(_pdf_safe_text(r.get("value", ""))[:80])
    else:
        pdf.multi_cell(w, 6, "No evidence data available.")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(w, 8, "Key Clinical Flags", ln=True)
    pdf.set_font("Helvetica", size=10)
    if report["key_flags"]:
        for f in report["key_flags"]:
            pdf.multi_cell(w, 6, "  - " + _pdf_safe_text(f))
    else:
        pdf.multi_cell(w, 6, "None identified.")
    pdf.ln(2)

    if report["data_gaps"]:
        pdf.set_font("Helvetica", "B", size=11)
        pdf.cell(w, 8, "Data Gaps", ln=True)
        pdf.set_font("Helvetica", size=10)
        for g in report["data_gaps"]:
            pdf.multi_cell(w, 6, "  - " + _pdf_safe_text(g))

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()

st.set_page_config(page_title="Clinical Risk Analysis Dashboard", layout="wide")

# Force clear any cached data/state to avoid stale UI issues.
try:
    st.cache_data.clear()
except Exception:
    pass

# --- Safe UI CSS ---
st.markdown(
    """
<style>

/* SAFE BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* CARD STYLE */
.patient-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    margin: 12px;
    display: inline-block;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* HOVER EFFECT (SAFE) */
.patient-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 20px 50px rgba(99,102,241,0.4);
}

/* FLOAT ANIMATION (SAFE) */
.patient-card {
    animation: floatCard 5s ease-in-out infinite;
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

/* RISK BADGE */
.risk-high {
    background: rgba(239,68,68,0.2);
    color: #ef4444;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: bold;
}

/* TITLE */
.main-title {
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    color: transparent;
}

</style>
""",
    unsafe_allow_html=True,
)

# Helper: Risk badge HTML
def _risk_badge_html(level: str) -> str:
    lvl = (level or "").strip().lower()
    if lvl == "high":
        return "<div class='risk-high'>🔴 High</div>"
    if lvl in ("moderate", "med", "medium"):
        return "<div class='risk-high' style='background: rgba(249,115,22,0.2); color:#f97316;'>🟠 Moderate</div>"
    if lvl == "low":
        return "<div class='risk-high' style='background: rgba(34,197,94,0.2); color:#22c55e;'>🟢 Low</div>"
    return "<div class='risk-high' style='background: rgba(255,255,255,0.10); color: rgba(255,255,255,0.85);'>—</div>"

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.caption("Clinical console")
page = st.sidebar.radio("Go to", ["📤 Upload Report", "📊 Patient Dashboard"], index=0)

# ——— Premium Header ———
st.markdown(
    '<div class="header-wrap">'
    '<div class="main-title">Clinical Risk Analysis</div>'
    '<div style="margin-top:8px; color: rgba(255,255,255,0.7); font-size:14px;">'
    'AI-powered diabetes & hypertension monitoring with longitudinal trends'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

if page == "📤 Upload Report":
    # ——— Section 2: Upload Clinical Report ———
    st.subheader("Upload Clinical Report")
    st.caption("Provide a patient identifier, then paste clinical text **or** upload a file (PDF, DOCX, PNG, JPG).")
    patient_id_input = st.text_input(
        "Patient ID",
        value="",
        placeholder="e.g. MRN1234 or internal patient code",
    )
    clinical_note = st.text_area(
        "Paste clinical or discharge note",
        height=200,
        placeholder="Paste discharge summary or clinical note text here...",
        label_visibility="collapsed",
    )
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    run_clicked = st.button("Analyze", type="primary", use_container_width=False)

    if run_clicked:
        clinical_text = None
        from_file = False
        extraction_failed = False
        if uploaded_file is not None:
            if not is_supported_file(uploaded_file.name):
                st.warning("Unsupported file type. Please upload PDF, DOCX, PNG, or JPG.")
            else:
                file_bytes = uploaded_file.read()
                clinical_text, err = extract_text_from_file(file_bytes, uploaded_file.name)
                if err:
                    st.warning(err)
                    extraction_failed = True
                else:
                    from_file = True
        if clinical_text is None and not (uploaded_file is not None and not is_supported_file(uploaded_file.name)):
            pasted = (clinical_note or "").strip()
            if pasted:
                clinical_text = pasted
        if clinical_text:
            if from_file:
                st.info("Clinical text extracted successfully. Running analysis...")
            progress = st.progress(0, text="Analyzing clinical data…")
            with st.spinner("Analyzing clinical data (Extraction → Risk Analysis → Summary → Visualization)…"):
                try:
                    progress.progress(18)
                    patient_id_effective = (patient_id_input or "").strip() or "unknown"
                    input_type = "text"
                    if from_file and uploaded_file is not None:
                        input_type = (uploaded_file.name.rsplit(".", 1)[-1] or "file").lower()
                    # Pipeline call goes through FastAPI (no AI logic inside Streamlit).
                    result = _api_post_analyze(patient_id_effective, clinical_text)
                    st.session_state["last_result"] = result
                    st.session_state["last_patient_id"] = patient_id_effective
                    # Build longitudinal trends from stored DB data (no re-runs).
                    st.session_state["patient_trends"] = _api_get_patient_trends(patient_id_effective)
                    progress.progress(100)
                    st.success("Analysis complete.")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    if "last_result" in st.session_state:
                        del st.session_state["last_result"]
                    progress.progress(0)
        elif not extraction_failed and not (uploaded_file and not is_supported_file(uploaded_file.name)):
            st.warning("Please enter a clinical note or upload a file before running analysis.")

    # ——— Sections 3–11: Only when we have a result (no JSON) ———
    if "last_result" not in st.session_state:
        st.info("Paste a clinical note or upload a file (PDF, DOCX, PNG, JPG), then click **Analyze** to see results.")
        st.stop()

    result = st.session_state["last_result"]
    patient_id_for_trends = st.session_state.get("last_patient_id", "")
    patient_trends = st.session_state.get("patient_trends")
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

    st.divider()

    # ——— Section 10: Patient Trends (Longitudinal History) ———
    st.subheader("Patient Trends")
    if patient_trends and isinstance(patient_trends, dict):
        t = patient_trends.get("trends") or {}
        dates = t.get("dates") or []
        a1c = t.get("a1c") or []
        glucose = t.get("glucose") or []
        bp_vals = t.get("bp") or []
        diab_risk = t.get("diabetes_risk") or []
        htn_risk = t.get("hypertension_risk") or []

    if dates:
        # Format dates (remove microseconds, keep date + time if present)
        pretty_dates = [str(d).split(".")[0] for d in dates]

        # Helper to add arrows vs previous value
        def with_trend_arrows(values):
            out = []
            prev = None
            for v in values:
                if v is None:
                    out.append("")
                    prev = v
                    continue
                arrow = ""
                if isinstance(prev, (int, float)) and isinstance(v, (int, float)):
                    if v < prev:
                        arrow = " ↓"
                    elif v > prev:
                        arrow = " ↑"
                out.append(f"{v}{arrow}")
                prev = v
            return out

        # Trend table with arrows
        trend_df = pd.DataFrame(
            {
                "Date": pretty_dates,
                "HbA1c": with_trend_arrows(a1c),
                "Glucose": with_trend_arrows(glucose),
                "BP": bp_vals,
                "Diabetes Risk": with_trend_arrows(diab_risk),
                "Hypertension Risk": with_trend_arrows(htn_risk),
            }
        )
        st.markdown("**Report History**")
        st.dataframe(trend_df, use_container_width=True, hide_index=True)

        # Line charts for risk over time
        chart_df = pd.DataFrame(
            {
                "Date": pretty_dates,
                "Diabetes Risk": diab_risk,
                "Hypertension Risk": htn_risk,
            }
        ).set_index("Date")
        st.markdown("**Risk Over Time**")
        st.line_chart(chart_df)

        # Clinical Insights section
        summary_trend = patient_trends.get("trend_summary") or {}
        diab_overall = summary_trend.get("diabetes_overall") or "Not enough data"
        diab_recent = summary_trend.get("diabetes_recent") or "Not enough data"
        htn_overall = summary_trend.get("hypertension_overall") or "Not enough data"
        htn_recent = summary_trend.get("hypertension_recent") or "Not enough data"

        st.markdown("**Clinical Insights**")
        st.markdown("**Diabetes:**")
        st.markdown(f"- Overall Trend: {diab_overall}")
        st.markdown(f"- Recent Trend: {diab_recent}")
        st.markdown("**Hypertension:**")
        st.markdown(f"- Overall Trend: {htn_overall}")
        st.markdown(f"- Recent Trend: {htn_recent}")
    else:
        st.caption("No stored history found for this patient yet. Trends will appear after multiple reports are stored.")

    st.divider()

    # ——— Section 11: Download Report ———
    st.subheader("Download Report")
    report_data = _build_report_data(result)
    col1, col2 = st.columns(2)
    with col1:
        pdf_bytes = _build_pdf_bytes(report_data)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="clinical_risk_analysis_report.pdf",
            mime="application/pdf",
        )
    with col2:
        json_str = json.dumps(report_data, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name="clinical_risk_analysis_report.json",
            mime="application/json",
        )

else:
    # ——— Patient Dashboard / Admin view ———
    st.subheader("Patient Dashboard")

    try:
        patients = _api_get_patients()
    except RuntimeError as e:
        st.error(str(e))
        st.info("Run this in another terminal:\n\n`uvicorn backend.main:app --reload`")
        st.stop()
    if not patients:
        st.info("No patients found in the database yet. Upload a report to create patient records.")
        st.stop()

    st.markdown("**All Patients Overview**")

    # Build latest snapshot per patient (for the card grid)
    overview_rows = []
    for pid in patients:
        history_rows = _api_get_patient_history(pid)
        if not history_rows:
            overview_rows.append(
                {
                    "Patient ID": pid,
                    "Last HbA1c": None,
                    "Last BP": "",
                    "Risk Level": "No reports",
                }
            )
            continue
        last = history_rows[-1]
        extraction = last.get("extraction_json") or {}
        viz_root = last.get("visualization_json") or {}
        viz = (viz_root.get("visualizations") if isinstance(viz_root, dict) else None) or {}
        diabetes = (extraction.get("diabetes") or {}) if isinstance(extraction, dict) else {}
        bp_block = (extraction.get("blood_pressure") or {}) if isinstance(extraction, dict) else {}
        a1c_vals = diabetes.get("a1c_values") or []
        bp_vals = bp_block.get("bp_readings") or []
        risk_levels = viz.get("risk_levels") or {}
        overall_level = (viz.get("severity_indicator") or {}).get("level") or "Low"
        overview_rows.append(
            {
                "Patient ID": pid,
                "Last HbA1c": a1c_vals[0] if isinstance(a1c_vals, list) and a1c_vals else None,
                "Last BP": bp_vals[0] if isinstance(bp_vals, list) and bp_vals else "",
                "Risk Level": overall_level,
            }
        )

    # Animated card grid (visual only; selection still uses selectbox)
    if not patients:
        st.warning("No patients found")
    else:
        cards_inner_html = ""
        for i, row in enumerate(overview_rows):
            pid = str(row.get("Patient ID") or "")
            a1c = row.get("Last HbA1c")
            bp = row.get("Last BP") or "—"
            level = str(row.get("Risk Level") or "No reports")
            a1c_txt = f"{a1c}" if a1c is not None else "—"

            if level == "No reports":
                badge_html = "<div class='risk-high' style='background: rgba(255,255,255,0.12); color: rgba(255,255,255,0.9);'>No reports</div>"
            else:
                badge_html = _risk_badge_html(level)

            delay_style = f"animation-delay:{i * 0.05:.2f}s;"
            # Build a SINGLE-LINE, indentation-free HTML block to prevent Markdown from treating it as code.
            card_html = (
                f"<div class='patient-card' style='{delay_style}'>"
                f"<h3 style='margin:0 0 10px 0; font-size:14px; font-weight:900;'>{pid}</h3>"
                f"<p style='margin:0 0 8px 0; font-size:12px;'>HbA1c: <b>{a1c_txt}</b></p>"
                f"<p style='margin:0 0 8px 0; font-size:12px;'>BP: <b>{bp}</b></p>"
                f"{badge_html}"
                f"</div>"
            )
            cards_inner_html += card_html

        grid_html = f"<div style='display:flex; flex-wrap:wrap; gap:12px;'>{cards_inner_html}</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Select Patient to View History**")
    selected_patient = st.selectbox("Patient ID", patients)

    if not selected_patient:
        st.info("Select a patient to view detailed history.")
        st.stop()

    history_rows = _api_get_patient_history(selected_patient)
    if not history_rows:
        st.warning("No reports available for this patient.")
        st.stop()

    # Patient summary card (latest status)
    latest = history_rows[-1]
    extraction = latest.get("extraction_json") or {}
    viz_root = latest.get("visualization_json") or {}
    viz = (viz_root.get("visualizations") if isinstance(viz_root, dict) else None) or {}
    diabetes = (extraction.get("diabetes") or {}) if isinstance(extraction, dict) else {}
    bp_block = (extraction.get("blood_pressure") or {}) if isinstance(extraction, dict) else {}
    a1c_vals = diabetes.get("a1c_values") or []
    bp_vals = bp_block.get("bp_readings") or []
    scores = viz.get("risk_scores") or {}
    levels = viz.get("risk_levels") or {}

    latest_a1c = a1c_vals[0] if isinstance(a1c_vals, list) and a1c_vals else None
    latest_bp = bp_vals[0] if isinstance(bp_vals, list) and bp_vals else ""
    diab_level = (levels.get("diabetes") or "Low").strip()
    htn_level = (levels.get("hypertension") or "Low").strip()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="glass" style="padding:16px; min-height:140px;">'
            f'<div style="font-weight:900; font-size:16px; margin-bottom:10px;">Patient Info</div>'
            f'<div style="color: rgba(255,255,255,0.75); font-size:13px; margin-bottom:10px;">'
            f'Patient ID: <b>{selected_patient}</b>'
            f'</div>'
            f'<div style="color: rgba(255,255,255,0.85); font-weight:700; font-size:13px;">'
            f'HbA1c: <b>{latest_a1c if latest_a1c is not None else "—"}</b>'
            f'</div>'
            f'<div style="color: rgba(255,255,255,0.85); font-weight:700; font-size:13px; margin-top:6px;">'
            f'BP: <b>{latest_bp or "—"}</b>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="glass" style="padding:16px; min-height:140px;">'
            f'<div style="font-weight:900; font-size:16px; margin-bottom:10px;">Latest Risk</div>'
            f'<div style="margin-bottom:10px;">Diabetes: {_risk_badge_html(diab_level)}</div>'
            f'<div>Hypertension: {_risk_badge_html(htn_level)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Detailed patient history table + upgraded graphs using deterministic trends
    trends_data = _api_get_patient_trends(selected_patient)
    t = trends_data.get("trends") or {}
    dates = t.get("dates") or []
    a1c = t.get("a1c") or []
    glucose = t.get("glucose") or []
    bp_vals = t.get("bp") or []
    diab_risk = t.get("diabetes_risk") or []
    htn_risk = t.get("hypertension_risk") or []

    if dates:
        pretty_dates = [str(d).split(".")[0] for d in dates]

        # Smart Insights card (overall + recent)
        s = trends_data.get("trend_summary") or {}
        diab_overall = s.get("diabetes_overall") or "Not enough data"
        diab_recent = s.get("diabetes_recent") or "Not enough data"
        htn_overall = s.get("hypertension_overall") or "Not enough data"
        htn_recent = s.get("hypertension_recent") or "Not enough data"

        def _insight_kind(recent_label: str) -> str:
            lbl = (recent_label or "").lower()
            if "worsening" in lbl:
                return "red"
            if "improving" in lbl:
                return "green"
            return "neutral"

        kind = _insight_kind(diab_recent) if "Not enough data" not in str(diab_recent) else _insight_kind(htn_recent)
        icon = "⚠️" if kind == "red" else ("✅" if kind == "green" else "ℹ️")
        st.markdown(
            f'<div class="insight {kind}">'
            f'<div style="font-weight:900; font-size:14px; margin-bottom:8px;">{icon} Clinical Insights</div>'
            f'<div style="color: rgba(255,255,255,0.82); font-size:13px; line-height:1.55;">'
            f'<b>Diabetes:</b> Overall {diab_overall} &nbsp;|&nbsp; Recent {diab_recent}<br/>'
            f'<b>Hypertension:</b> Overall {htn_overall} &nbsp;|&nbsp; Recent {htn_recent}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Patient History Table (kept, but visually toned down)
        st.markdown("**Patient History (All Reports)**")
        history_df = pd.DataFrame(
            {
                "Date": pretty_dates,
                "HbA1c": a1c,
                "Glucose": glucose,
                "BP": bp_vals,
                "Diabetes Risk": diab_risk,
                "Hypertension Risk": htn_risk,
            }
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        # Charts upgraded with smooth curves + glow-like layered lines
        # HbA1c chart
        df_a1c = pd.DataFrame({"Date": pretty_dates, "HbA1c": a1c})
        base_a1c = alt.Chart(df_a1c).encode(x=alt.X("Date", sort=None), y="HbA1c:Q")
        glow_a1c = base_a1c.mark_line(interpolate="basis", strokeWidth=6, color="rgba(56,189,248,0.18)")
        line_a1c = base_a1c.mark_line(interpolate="basis", strokeWidth=3, color="#38bdf8")
        st.markdown("**HbA1c Over Time**")
        st.altair_chart(glow_a1c + line_a1c, use_container_width=True)

        # BP chart (systolic)
        def _parse_sys(bp_str: str):
            bp_s = (bp_str or "").strip()
            if not bp_s or "/" not in bp_s:
                return None
            try:
                return int(bp_s.split("/", 1)[0].strip())
            except ValueError:
                return None

        bp_systolic = [_parse_sys(x) for x in bp_vals]
        df_bp = pd.DataFrame({"Date": pretty_dates, "SystolicBP": bp_systolic})
        base_bp = alt.Chart(df_bp).encode(x=alt.X("Date", sort=None), y="SystolicBP:Q")
        glow_bp = base_bp.mark_line(interpolate="basis", strokeWidth=6, color="rgba(99,102,241,0.18)")
        line_bp = base_bp.mark_line(interpolate="basis", strokeWidth=3, color="#6366f1")
        st.markdown("**Blood Pressure (Systolic) Over Time**")
        st.altair_chart(glow_bp + line_bp, use_container_width=True)

        # Risk chart
        df_risk = pd.DataFrame({"Date": pretty_dates, "Diabetes Risk": diab_risk, "Hypertension Risk": htn_risk})
        base_r = alt.Chart(df_risk).encode(x=alt.X("Date", sort=None))
        glow_d = base_r.mark_line(interpolate="basis", strokeWidth=6, color="rgba(56,189,248,0.18)").encode(y="Diabetes Risk:Q")
        line_d = base_r.mark_line(interpolate="basis", strokeWidth=3, color="#38bdf8").encode(y="Diabetes Risk:Q")
        glow_h = base_r.mark_line(interpolate="basis", strokeWidth=6, color="rgba(168,85,247,0.18)").encode(y="Hypertension Risk:Q")
        line_h = base_r.mark_line(interpolate="basis", strokeWidth=3, color="#a855f7").encode(y="Hypertension Risk:Q")
        st.markdown("**Risk Scores Over Time**")
        st.altair_chart(glow_d + line_d + glow_h + line_h, use_container_width=True)
    else:
        st.warning("No trend data available for this patient yet.")
