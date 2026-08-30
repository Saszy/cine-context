from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

DB_MOVIES = ROOT / "db" / "movies.db"
DB_RATINGS = ROOT / "db" / "ratings.db"
DATA_DIR = ROOT / "data"
ENRICHED_JSON = DATA_DIR / "enriched_movies.json"
ENRICHED_DB = DATA_DIR / "enriched.db"

SAMPLE_SIZE = 80
SAMPLE_SEED = 42
MIN_RATINGS = 5
MIN_OVERVIEW_LEN = 40
MIN_FINANCIALS = 1_000

def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def openai_api_key() -> str:
    load_dotenv(ROOT / ".env", override=True)
    return os.getenv("OPENAI_API_KEY", "").strip()


# Back-compat for imports; prefer openai_api_key() so a late-loaded .env is seen.
OPENAI_MODEL = openai_model()
OPENAI_API_KEY = openai_api_key()

# Rule bands — code assigns tiers; LLM may only suggest optional overrides.
BUDGET_LOW_MAX = 15_000_000
BUDGET_HIGH_MIN = 80_000_000
REVENUE_LOW_MAX = 40_000_000
REVENUE_HIGH_MIN = 200_000_000

PES_ROI_WEIGHT = 0.55
PES_RATING_WEIGHT = 0.45
PROMPT_VERSION = "data-prep-v1"
