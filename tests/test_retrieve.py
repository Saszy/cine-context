from __future__ import annotations

from movie_system.models import QueryIntent
from movie_system.retrieve import retrieve, user_ratings
from tests.conftest import make_movie


def test_unknown_titles_do_not_fall_back_to_top_pes() -> None:
    catalog = [
        make_movie(movie_id=1, title="Ice Age: The Meltdown", production_effectiveness_score=91),
        make_movie(movie_id=2, title="Pulp Fiction", production_effectiveness_score=90),
    ]
    intent = QueryIntent(task="compare", movie_titles=["The Dark Knight", "Inception"])
    assert retrieve(intent, catalog) == []


def test_known_titles_only() -> None:
    catalog = [
        make_movie(movie_id=1, title="Pulp Fiction"),
        make_movie(movie_id=2, title="Lost in Translation"),
        make_movie(movie_id=3, title="Saw"),
    ]
    intent = QueryIntent(task="compare", movie_titles=["Pulp Fiction", "Lost in Translation"])
    hits = retrieve(intent, catalog)
    assert {m.title for m in hits} == {"Pulp Fiction", "Lost in Translation"}


def test_filters_genre_and_revenue() -> None:
    catalog = [
        make_movie(movie_id=1, title="A", genres=["Action"], revenue_tier="high", overview_sentiment="positive"),
        make_movie(movie_id=2, title="B", genres=["Action"], revenue_tier="low", overview_sentiment="positive"),
        make_movie(movie_id=3, title="C", genres=["Drama"], revenue_tier="high", overview_sentiment="positive"),
    ]
    intent = QueryIntent(
        task="recommend",
        genres=["Action"],
        revenue_tier="high",
        sentiment="positive",
    )
    hits = retrieve(intent, catalog)
    assert [m.movie_id for m in hits] == [1]


def test_impossible_effectiveness_is_empty() -> None:
    catalog = [make_movie(production_effectiveness_score=70)]
    intent = QueryIntent(task="search", min_effectiveness=99)
    assert retrieve(intent, catalog) == []


def test_unknown_user_has_no_ratings(catalog) -> None:
    assert user_ratings(999999, catalog) == []
