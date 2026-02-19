"""
Gemini LLM instance for CrewAI agents.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    validate_settings,
)


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Create and return ChatGoogleGenerativeAI for CrewAI. Loads API key from .env."""
    validate_settings()
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        google_api_key=GEMINI_API_KEY,
    )
