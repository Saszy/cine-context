from __future__ import annotations

import pytest

from movie_system.evaluate import GOLDEN_INTENTS
from movie_system.system import parse_intent


@pytest.mark.parametrize("case", GOLDEN_INTENTS)
def test_golden_heuristic_intents(case: dict) -> None:
    intent = parse_intent(case["query"], use_llm=False)
    assert intent.task == case["task"]
    if "user_id" in case:
        assert intent.user_id == case["user_id"]
    if "genres" in case:
        assert {g.lower() for g in case["genres"]}.issubset({g.lower() for g in intent.genres})
    if "revenue_tier" in case:
        assert intent.revenue_tier == case["revenue_tier"]
    if "budget_tier" in case:
        assert intent.budget_tier == case["budget_tier"]
    if "sentiment" in case:
        assert intent.sentiment == case["sentiment"]


def test_compare_extracts_quoted_titles() -> None:
    intent = parse_intent(
        "Compare budget of 'Pulp Fiction' and 'Lost in Translation'",
        use_llm=False,
    )
    assert intent.task == "compare"
    assert "Pulp Fiction" in intent.movie_titles
    assert "Lost in Translation" in intent.movie_titles


def test_high_budget_hyphen() -> None:
    intent = parse_intent(
        "Predict the rating user 15 would give a high-budget action movie",
        use_llm=False,
    )
    assert intent.budget_tier == "high"
    assert intent.user_id == 15


def test_missing_user_number() -> None:
    intent = parse_intent(
        "Summarize preferences for user based on their ratings and movie overviews",
        use_llm=False,
    )
    assert intent.task == "user_summary"
    assert intent.user_id is None


def test_non_numeric_user_is_ignored() -> None:
    intent = parse_intent(
        "Predict the rating user abc would give 'Pulp Fiction'",
        use_llm=False,
    )
    assert intent.user_id is None
    assert intent.task == "rating_predict"


def test_ambiguous_query_is_search() -> None:
    intent = parse_intent("something fun", use_llm=False)
    assert intent.task == "search"
    assert not intent.movie_titles
    assert intent.user_id is None
