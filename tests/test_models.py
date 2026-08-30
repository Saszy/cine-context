from __future__ import annotations

import pytest
from pydantic import ValidationError

from movie_system.models import QueryIntent, SystemResponse


def test_intent_coerces_string_null_tiers() -> None:
    intent = QueryIntent.model_validate(
        {
            "task": "user_summary",
            "budget_tier": "null",
            "revenue_tier": "none",
            "sentiment": "",
            "user_id": "564",
        }
    )
    assert intent.budget_tier is None
    assert intent.revenue_tier is None
    assert intent.sentiment is None
    assert intent.user_id == 564


def test_intent_rejects_invalid_tier() -> None:
    with pytest.raises(ValidationError):
        QueryIntent.model_validate({"task": "search", "budget_tier": "huge"})


def test_normalize_llm_json_null_strings() -> None:
    from movie_system.llm import normalize_llm_json

    cleaned = normalize_llm_json(
        {"budget_tier": "null", "titles": ["Pulp Fiction", "none"], "n": 1}
    )
    assert cleaned == {"budget_tier": None, "titles": ["Pulp Fiction", None], "n": 1}


def test_predicted_rating_null_string() -> None:
    resp = SystemResponse.model_validate(
        {
            "task": "recommend",
            "answer": "ok",
            "predicted_rating": "null",
        }
    )
    assert resp.predicted_rating is None
