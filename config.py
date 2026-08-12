"""Model factory and tunables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-flash-latest"

# Tool 1: search
MAX_RESULTS = 5
SNIPPET_CHARS = 300

# Tool 2: scrape
MAX_CONTENT_CHARS = 8_000
REQUEST_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Pipeline
MAX_URLS = 4
SOURCE_CHARS = 4_000


def get_model(temperature: float = 0):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, google_api_key=api_key, temperature=temperature
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER!r} (expected 'openai' or 'gemini')")
