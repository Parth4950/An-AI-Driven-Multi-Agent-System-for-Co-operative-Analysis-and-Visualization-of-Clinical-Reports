"""
Initialize PostgreSQL schema for the clinical AI system.

Ensures a relational structure:
- patients (patient_id PK)
- reports (per‑report text, FK to patients)
- results (per‑report JSON outputs, FK to reports)
"""

from __future__ import annotations

from db.db_config import get_connection


def init_db() -> None:
    """Create required tables and relationships if they do not exist."""
    try:
        conn = get_connection()
    except Exception:
        # Connection errors are already printed in db_config.get_connection.
        return

    try:
        with conn:
            with conn.cursor() as cur:
                # Patients table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patients (
                        patient_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                # Reports table (per patient, multiple reports)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reports (
                        id SERIAL PRIMARY KEY,
                        patient_id TEXT,
                        raw_text TEXT,
                        input_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                # Results table (one row per report run)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS results (
                        id SERIAL PRIMARY KEY,
                        report_id INTEGER,
                        extraction_json JSONB,
                        risk_json JSONB,
                        summary_json JSONB,
                        visualization_json JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

                # Ensure results.report_id column exists and is linked to reports.id
                cur.execute(
                    """
                    ALTER TABLE results
                    ADD COLUMN IF NOT EXISTS report_id INTEGER;
                    """
                )
                # Add FK constraints best‑effort; ignore if they already exist
                try:
                    cur.execute(
                        """
                        ALTER TABLE reports
                        ADD CONSTRAINT fk_reports_patient
                        FOREIGN KEY (patient_id) REFERENCES patients(patient_id);
                        """
                    )
                except Exception:
                    # Ignore if constraint already exists.
                    pass
                try:
                    cur.execute(
                        """
                        ALTER TABLE results
                        ADD CONSTRAINT fk_results_report
                        FOREIGN KEY (report_id) REFERENCES reports(id);
                        """
                    )
                except Exception:
                    pass
        print("[DB] Tables ensured (patients, reports, results).")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    init_db()

