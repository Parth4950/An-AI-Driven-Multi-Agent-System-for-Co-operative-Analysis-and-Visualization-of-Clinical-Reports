"""
Load Streamlit Cloud / local secrets into os.environ before settings are read.

On Community Cloud, `.env` is not deployed. Secrets are configured in the app
dashboard and exposed via `st.secrets`.
"""

from __future__ import annotations

import os
from typing import Any


def _set_env(key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        os.environ.setdefault(key, text)


def apply_streamlit_secrets(secrets: Any) -> None:
    """Copy flat and nested TOML secrets into the process environment."""
    try:
        for key, value in secrets.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    env_key = f"{key}_{sub_key}".upper()
                    _set_env(env_key, sub_value)
                    if sub_key.upper() == "API_KEY" and key.upper() in ("GEMINI", "GOOGLE"):
                        _set_env("GEMINI_API_KEY", sub_value)
                        _set_env("GOOGLE_API_KEY", sub_value)
            else:
                _set_env(str(key).upper(), value)
                _set_env(str(key), value)
    except Exception:
        return

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = gemini_key
