"""
Runtime checks for Streamlit / CrewAI deployment targets.

CrewAI officially supports Python >=3.10 and <3.14. Its ChromaDB dependency
(1.1.x) uses a Pydantic v1 shim that fails on Python 3.14 with:
  unable to infer type for attribute "chroma_server_nofile"
"""

from __future__ import annotations

import sys


def check_runtime_compatibility() -> str | None:
    """
    Return a user-facing error message when the environment is unsupported.
    Return None when checks pass.
    """
    if sys.version_info >= (3, 14):
        return (
            "This app requires Python 3.10–3.13 (CrewAI/ChromaDB are not compatible with Python 3.14). "
            "In Streamlit Community Cloud: open **Manage app → Settings → Python version** and select **3.12**, "
            "then reboot the app."
        )

    try:
        import chromadb  # noqa: F401
    except Exception as exc:
        msg = str(exc).strip() or exc.__class__.__name__
        if "chroma_server_nofile" in msg:
            return (
                "ChromaDB failed to initialize (Pydantic compatibility issue). "
                "Set **Python 3.12** in Streamlit app settings and reboot. "
                f"Details: {msg}"
            )
        return f"ChromaDB import failed: {msg}"

    return None
