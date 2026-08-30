from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


Sentiment = Literal["positive", "negative", "neutral", "mixed"]
Tier = Literal["low", "medium", "high"]
TaskType = Literal["recommend", "compare", "user_summary", "rating_predict", "search"]


class MovieRecord(BaseModel):
    movie_id: int
    imdb_id: str
    title: str
    overview: str
    production_companies: list[str]
    release_date: Optional[str] = None
    budget: int
    revenue: int
    runtime: Optional[float] = None
    language: Optional[str] = None
    genres: list[str]
    status: Optional[str] = None
    n_ratings: int
    avg_rating: float


class FinancialFacts(BaseModel):
    budget_tier: Tier
    revenue_tier: Tier
    roi: float
    production_effectiveness_score: int = Field(ge=0, le=100)
    production_effectiveness_label: str


class EnrichmentJudgment(BaseModel):
    """LLM-owned fields. Tiers and PES are computed in code and merged after."""

    movie_id: int
    overview_sentiment: Sentiment
    sentiment_rationale: str
    budget_rationale: str
    revenue_rationale: str
    production_effectiveness_rationale: str
    themes: list[str] = Field(min_length=1, max_length=6)
    intended_audience: str
    budget_tier_override: Optional[Tier] = None
    budget_override_rationale: Optional[str] = None
    revenue_tier_override: Optional[Tier] = None
    revenue_override_rationale: Optional[str] = None

    @field_validator("themes")
    @classmethod
    def normalize_themes(cls, value: list[str]) -> list[str]:
        cleaned = [t.strip() for t in value if t and t.strip()]
        return cleaned[:6] or ["unspecified"]


class Enrichment(EnrichmentJudgment):
    budget_tier: Tier
    revenue_tier: Tier
    production_effectiveness_score: int = Field(ge=0, le=100)
    production_effectiveness_label: str
    source: Literal["llm", "heuristic"] = "llm"
    model: Optional[str] = None
    prompt_version: str = "data-prep-v1"


class EnrichedMovie(MovieRecord, Enrichment):
    pass


class EnrichmentBatch(BaseModel):
    movies: list[EnrichmentJudgment]


def _blank_to_none(value: Any) -> Any:
    """Models often emit the string 'null' instead of JSON null."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "undefined"}:
        return None
    return value


class QueryIntent(BaseModel):
    task: TaskType
    genres: list[str] = []
    sentiment: Optional[Sentiment] = None
    budget_tier: Optional[Tier] = None
    revenue_tier: Optional[Tier] = None
    min_effectiveness: Optional[int] = None
    movie_titles: list[str] = []
    user_id: Optional[int] = None
    rating_examples_needed: bool = False
    notes: str = ""

    @field_validator(
        "sentiment",
        "budget_tier",
        "revenue_tier",
        "min_effectiveness",
        "user_id",
        mode="before",
    )
    @classmethod
    def optional_null_strings(cls, value: Any) -> Any:
        return _blank_to_none(value)


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    reason: str


class SystemResponse(BaseModel):
    task: TaskType
    answer: str
    movies: list[RecommendationItem] = []
    caveats: list[str] = []
    predicted_rating: Optional[float] = None

    @field_validator("predicted_rating", mode="before")
    @classmethod
    def optional_predicted_rating(cls, value: Any) -> Any:
        return _blank_to_none(value)


class UserRatingRow(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    overview: str
    user_rating: float
    avg_rating: Optional[float] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    enrichment: Optional[dict[str, Any]] = None
    title_verified: bool = False
