"""Environment/config loading. Import this before anything that touches
fastf1, LangSmith, or the Anthropic client, so env vars are set first."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get("LANGSMITH_PROJECT", "box-box-bot")

FASTF1_CACHE_DIR = PROJECT_ROOT / os.environ.get("FASTF1_CACHE_DIR", "data/cache")

RACE_RECAPS_DIR = PROJECT_ROOT / "data" / "race_recaps"
RAG_PERSIST_DIR = PROJECT_ROOT / "data" / "vectorstore"

# LangSmith reads these exact env var names, so mirror our .env names onto them.
os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGSMITH_TRACING)
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
