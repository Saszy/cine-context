from __future__ import annotations

from movie_system.enrich import assign_tier, compute_facts, heuristic_enrich, pes_label
from movie_system.models import MovieRecord


def _movie(budget: int, revenue: int, rating: float = 2.5) -> MovieRecord:
    return MovieRecord(
        movie_id=1,
        imdb_id="tt1",
        title="X",
        overview="A quiet story about friendship and hope in a small town.",
        production_companies=[],
        budget=budget,
        revenue=revenue,
        genres=["Drama"],
        n_ratings=8,
        avg_rating=rating,
    )


def test_budget_tier_bands() -> None:
    assert assign_tier(1, 15_000_000, 80_000_000) == "low"
    assert assign_tier(14_999_999, 15_000_000, 80_000_000) == "low"
    assert assign_tier(15_000_000, 15_000_000, 80_000_000) == "medium"
    assert assign_tier(80_000_000, 15_000_000, 80_000_000) == "high"


def test_pes_break_even_is_fifty() -> None:
    facts = compute_facts(_movie(budget=10_000_000, revenue=10_000_000, rating=2.5))
    assert facts.roi == 1.0
    assert facts.production_effectiveness_score == 50
    assert facts.production_effectiveness_label == "as_expected"


def test_pes_labels() -> None:
    assert pes_label(80) == "highly_effective"
    assert pes_label(60) == "effective"
    assert pes_label(45) == "as_expected"
    assert pes_label(44) == "underperformed"


def test_heuristic_sentiment_is_neutral_not_keywords() -> None:
    movie = _movie(5_000_000, 8_000_000)
    movie.overview = "A dark murder revenge war crime horror tragedy."
    extra = heuristic_enrich(movie)
    assert extra.overview_sentiment == "neutral"
    assert extra.source == "heuristic"
    assert "LLM-only" in extra.sentiment_rationale
