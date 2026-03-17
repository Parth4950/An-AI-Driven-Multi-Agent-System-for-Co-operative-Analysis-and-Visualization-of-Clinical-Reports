"""
Database insert helpers for the clinical AI pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

from psycopg2.extras import Json, RealDictCursor

from db.db_config import get_connection


def ensure_patient(patient_id: str) -> None:
    """Create patient row if it does not exist."""
    if not patient_id:
        return
    try:
        conn = get_connection()
    except Exception:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO patients (patient_id)
                    VALUES (%s)
                    ON CONFLICT (patient_id) DO NOTHING;
                    """,
                    (patient_id,),
                )
        print("[DB] Ensured patient exists.")
    except Exception as exc:
        print(f"[DB] Failed to ensure patient: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_report(patient_id: str, raw_text: str, input_type: str) -> int | None:
    """
    Insert a single report row and return its report_id.

    input_type can be e.g. 'text', 'pdf', 'docx', 'image'.
    """
    try:
        conn = get_connection()
    except Exception:
        # Connection failure already logged; fail gracefully.
        return None

    report_id: int | None = None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reports (patient_id, raw_text, input_type)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """,
                    (patient_id, raw_text, input_type),
                )
                row = cur.fetchone()
                if row:
                    report_id = row[0]
        print("[DB] Stored report in `reports` table.")
        return report_id
    except Exception as exc:
        print(f"[DB] Failed to insert report: {exc}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_results(
    report_id: int,
    extraction_output: Dict[str, Any],
    risk_output: Dict[str, Any],
    summary_output: Dict[str, Any],
    visualization_output: Dict[str, Any],
) -> None:
    """
    Insert pipeline outputs into the results table, linked to a report.
    """
    try:
        conn = get_connection()
    except Exception:
        # Connection failure already logged; fail gracefully.
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO results (
                        report_id,
                        extraction_json,
                        risk_json,
                        summary_json,
                        visualization_json
                    )
                    VALUES (%s, %s, %s, %s, %s);
                    """,
                    (
                        report_id,
                        Json(extraction_output),
                        Json(risk_output),
                        Json(summary_output),
                        Json(visualization_output),
                    ),
                )
        print("[DB] Stored pipeline outputs in `results` table.")
    except Exception as exc:
        print(f"[DB] Failed to insert results: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all reports and associated results for a patient, ordered by report.created_at ASC.
    Uses only stored DB data (no re‑runs).
    """
    if not patient_id:
        return []
    try:
        conn = get_connection()
    except Exception:
        return []

    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        r.id AS report_id,
                        r.patient_id,
                        r.created_at AS report_created_at,
                        res.extraction_json,
                        res.risk_json,
                        res.visualization_json,
                        res.created_at AS result_created_at
                    FROM reports r
                    JOIN results res ON res.report_id = r.id
                    WHERE r.patient_id = %s
                    ORDER BY r.created_at ASC;
                    """,
                    (patient_id,),
                )
                rows = cur.fetchall() or []
                return [dict(row) for row in rows]
    except Exception as exc:
        print(f"[DB] Failed to fetch patient history: {exc}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_all_patients() -> List[str]:
    """Return a sorted list of all patient_ids in the patients table."""
    try:
        conn = get_connection()
    except Exception:
        return []

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT patient_id
                    FROM patients
                    ORDER BY patient_id;
                    """
                )
                rows = cur.fetchall() or []
                return [r[0] for r in rows if r and r[0]]
    except Exception as exc:
        print(f"[DB] Failed to fetch patients: {exc}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_patient_history(patient_id: str) -> List[Dict[str, Any]]:
    """
    Convenience wrapper that returns a lightweight per‑report view:
    [
      {
        "date": ...,
        "a1c": ...,
        "glucose": ...,
        "bp": ...,
        "diabetes_risk": ...,
        "hypertension_risk": ...,
      },
      ...
    ]
    """
    rows = fetch_patient_history(patient_id)
    history: List[Dict[str, Any]] = []
    for row in rows:
        created = row.get("report_created_at") or row.get("created_at")
        extraction = row.get("extraction_json") or {}
        viz_root = row.get("visualization_json") or {}
        viz = (viz_root.get("visualizations") if isinstance(viz_root, dict) else None) or {}

        diabetes = (extraction.get("diabetes") or {}) if isinstance(extraction, dict) else {}
        bp_block = (extraction.get("blood_pressure") or {}) if isinstance(extraction, dict) else {}
        a1c_vals = diabetes.get("a1c_values") or []
        gluc_vals = diabetes.get("glucose_values") or []
        bp_vals = bp_block.get("bp_readings") or []

        scores = viz.get("risk_scores") or {}
        history.append(
            {
                "date": created,
                "a1c": (a1c_vals[0] if isinstance(a1c_vals, list) and a1c_vals else None),
                "glucose": (gluc_vals[0] if isinstance(gluc_vals, list) and gluc_vals else None),
                "bp": (bp_vals[0] if isinstance(bp_vals, list) and bp_vals else None),
                "diabetes_risk": scores.get("diabetes_score"),
                "hypertension_risk": scores.get("hypertension_score"),
            }
        )
    return history


