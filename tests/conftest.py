from __future__ import annotations

import pytest

from movie_system.enrich import load_enriched
from movie_system.models import EnrichedMovie


@pytest.fixture(autouse=True)
def _observe_to_tmp(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OBSERVE_PATH", str(tmp_path / "query_traces.jsonl"))


@pytest.fixture(scope="session")
def catalog() -> list[EnrichedMovie]:
    return load_enriched()


def make_movie(**overrides: object) -> EnrichedMovie:
    data = {
        "movie_id": 1,
        "imdb_id": "tt0000001",
        "title": "Test Film",
        "overview": "A short overview used in tests.",
        "production_companies": ["Test Co"],
        "release_date": "2000-01-01",
        "budget": 10_000_000,
        "revenue": 40_000_000,
        "runtime": 100.0,
        "language": None,
        "genres": ["Action"],
        "status": "Released",
        "n_ratings": 10,
        "avg_rating": 3.5,
        "overview_sentiment": "positive",
        "sentiment_rationale": "test",
        "budget_tier": "low",
        "budget_rationale": "test",
        "revenue_tier": "medium",
        "revenue_rationale": "test",
        "production_effectiveness_score": 70,
        "production_effectiveness_label": "effective",
        "production_effectiveness_rationale": "test",
        "themes": ["action"],
        "intended_audience": "testers",
        "source": "heuristic",
    }
    data.update(overrides)
    return EnrichedMovie.model_validate(data)
