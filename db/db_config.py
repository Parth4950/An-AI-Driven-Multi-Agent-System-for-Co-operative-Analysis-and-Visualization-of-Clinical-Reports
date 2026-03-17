"""
Database configuration for the clinical AI system.

Provides a simple helper to obtain a psycopg2 connection.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
from psycopg2.extensions import connection as PGConnection


def get_connection(password: Optional[str] = None) -> PGConnection:
    """
    Create and return a new PostgreSQL connection.

    Parameters
    ----------
    password:
        Optional explicit password override. If not provided, the password
        is read from the CLINICAL_DB_PASSWORD environment variable.
    """
    db_password = password if password is not None else os.getenv("CLINICAL_DB_PASSWORD", "")

    try:
        conn = psycopg2.connect(
            dbname="clinical_ai_db",
            user="postgres",
            password=db_password,
            host=os.getenv("CLINICAL_DB_HOST", "localhost"),
            port=os.getenv("CLINICAL_DB_PORT", "5432"),
        )
        return conn
    except Exception as exc:  # pragma: no cover - defensive logging
        # Keep this simple and side‑effect free; callers handle failures.
        print(f"[DB] Failed to connect to PostgreSQL: {exc}")
        raise

