from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from movie_system.config import (
    BUDGET_HIGH_MIN,
    BUDGET_LOW_MAX,
    DATA_DIR,
    ENRICHED_DB,
    ENRICHED_JSON,
    OPENAI_MODEL,
    PES_RATING_WEIGHT,
    PES_ROI_WEIGHT,
    PROMPT_VERSION,
    REVENUE_HIGH_MIN,
    REVENUE_LOW_MAX,
)
from movie_system.llm import complete_json, llm_available, movie_to_prompt_dict
from movie_system.models import (
    Enrichment,
    EnrichmentBatch,
    EnrichmentJudgment,
    EnrichedMovie,
    FinancialFacts,
    MovieRecord,
    Tier,
)
from movie_system.prompts import ENRICH_SYSTEM


def assign_tier(value: int, low_max: int, high_min: int) -> Tier:
    if value < low_max:
        return "low"
    if value >= high_min:
        return "high"
    return "medium"


def pes_label(score: int) -> str:
    if score >= 80:
        return "highly_effective"
    if score >= 60:
        return "effective"
    if score >= 45:
        return "as_expected"
    return "underperformed"


def compute_facts(movie: MovieRecord) -> FinancialFacts:
    roi = movie.revenue / movie.budget
    roi_score = min(100.0, max(0.0, (math.log10(max(roi, 0.01)) + 1) / 2 * 100))
    rating_score = (movie.avg_rating / 5.0) * 100
    pes = int(round(PES_ROI_WEIGHT * roi_score + PES_RATING_WEIGHT * rating_score))
    pes = max(0, min(100, pes))
    return FinancialFacts(
        budget_tier=assign_tier(movie.budget, BUDGET_LOW_MAX, BUDGET_HIGH_MIN),
        revenue_tier=assign_tier(movie.revenue, REVENUE_LOW_MAX, REVENUE_HIGH_MIN),
        roi=roi,
        production_effectiveness_score=pes,
        production_effectiveness_label=pes_label(pes),
    )


def _usd(n: int) -> str:
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    return f"${n:,}"


def _clean_override(rule: Tier, override: Tier | None, rationale: str | None) -> tuple[Tier | None, str | None]:
    if override is None or override == rule:
        return None, None
    text = (rationale or "").strip() or None
    return override, text


def merge_enrichment(
    movie: MovieRecord,
    facts: FinancialFacts,
    judgment: EnrichmentJudgment,
    *,
    source: str,
    model: str | None,
) -> Enrichment:
    budget_override, budget_why = _clean_override(
        facts.budget_tier,
        judgment.budget_tier_override,
        judgment.budget_override_rationale,
    )
    revenue_override, revenue_why = _clean_override(
        facts.revenue_tier,
        judgment.revenue_tier_override,
        judgment.revenue_override_rationale,
    )
    return Enrichment(
        movie_id=movie.movie_id,
        overview_sentiment=judgment.overview_sentiment,
        sentiment_rationale=judgment.sentiment_rationale,
        budget_tier=facts.budget_tier,
        budget_rationale=judgment.budget_rationale,
        revenue_tier=facts.revenue_tier,
        revenue_rationale=judgment.revenue_rationale,
        production_effectiveness_score=facts.production_effectiveness_score,
        production_effectiveness_label=facts.production_effectiveness_label,
        production_effectiveness_rationale=judgment.production_effectiveness_rationale,
        themes=judgment.themes,
        intended_audience=judgment.intended_audience,
        budget_tier_override=budget_override,
        budget_override_rationale=budget_why,
        revenue_tier_override=revenue_override,
        revenue_override_rationale=revenue_why,
        source=source,  # type: ignore[arg-type]
        model=model,
        prompt_version=PROMPT_VERSION,
    )


def heuristic_judgment(movie: MovieRecord, facts: FinancialFacts) -> EnrichmentJudgment:
    themes = movie.genres[:4] or ["drama"]
    audience = f"viewers who like {' / '.join(themes[:2]).lower()}"
    return EnrichmentJudgment(
        movie_id=movie.movie_id,
        overview_sentiment="neutral",
        sentiment_rationale=(
            "Heuristic fallback: overview sentiment is LLM-only; "
            "no API key, so this is left neutral rather than inferred from plot keywords."
        ),
        budget_rationale=(
            f"Budget {_usd(movie.budget)} maps to {facts.budget_tier} "
            f"on the $15M / $80M rule bands."
        ),
        revenue_rationale=(
            f"Revenue {_usd(movie.revenue)} maps to {facts.revenue_tier} "
            f"on the $40M / $200M rule bands."
        ),
        production_effectiveness_rationale=(
            f"ROI is {facts.roi:.2f}x and mean user rating is {movie.avg_rating:.2f}/5 "
            f"from {movie.n_ratings} ratings, producing a blended score of "
            f"{facts.production_effectiveness_score} ({facts.production_effectiveness_label})."
        ),
        themes=themes,
        intended_audience=audience,
    )


def heuristic_enrich(movie: MovieRecord) -> Enrichment:
    facts = compute_facts(movie)
    return merge_enrichment(
        movie,
        facts,
        heuristic_judgment(movie, facts),
        source="heuristic",
        model=None,
    )


def llm_enrich_batch(movies: list[MovieRecord]) -> list[Enrichment]:
    facts_by_id = {m.movie_id: compute_facts(m) for m in movies}
    payload = []
    for movie in movies:
        facts = facts_by_id[movie.movie_id]
        row = movie_to_prompt_dict(movie.model_dump())
        row.update(
            {
                "budget_tier": facts.budget_tier,
                "revenue_tier": facts.revenue_tier,
                "roi": round(facts.roi, 2),
                "production_effectiveness_score": facts.production_effectiveness_score,
                "production_effectiveness_label": facts.production_effectiveness_label,
            }
        )
        payload.append(row)
    user = (
        "Write sentiment, rationales, and themes. Copy movie_id exactly. "
        "Use the CODE tiers and PES as given.\n\n"
        + json.dumps(payload, indent=2)
    )
    batch = complete_json(ENRICH_SYSTEM, user, EnrichmentBatch, temperature=0.2)
    by_id = {item.movie_id: item for item in batch.movies}
    missing = [m.movie_id for m in movies if m.movie_id not in by_id]
    if missing:
        raise RuntimeError(f"LLM omitted movie_ids: {missing}")
    return [
        merge_enrichment(
            movie,
            facts_by_id[movie.movie_id],
            by_id[movie.movie_id],
            source="llm",
            model=OPENAI_MODEL,
        )
        for movie in movies
    ]


def enrich_movies(movies: list[MovieRecord], *, use_llm: bool) -> list[EnrichedMovie]:
    if use_llm and not llm_available():
        raise RuntimeError("use_llm=True but OPENAI_API_KEY is not set.")

    enrichments: dict[int, Enrichment] = {}
    if use_llm:
        chunk_size = 8
        for i in range(0, len(movies), chunk_size):
            chunk = movies[i : i + chunk_size]
            for item in llm_enrich_batch(chunk):
                enrichments[item.movie_id] = item
    else:
        for movie in movies:
            enrichments[movie.movie_id] = heuristic_enrich(movie)

    merged: list[EnrichedMovie] = []
    for movie in movies:
        extra = enrichments[movie.movie_id]
        merged.append(EnrichedMovie(**{**movie.model_dump(), **extra.model_dump()}))
    return merged


def save_enriched(movies: list[EnrichedMovie]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_JSON.write_text(
        json.dumps([m.model_dump() for m in movies], indent=2),
        encoding="utf-8",
    )
    _write_sqlite(movies)


def load_enriched(path: Path | None = None) -> list[EnrichedMovie]:
    target = path or ENRICHED_JSON
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [EnrichedMovie.model_validate(row) for row in raw]


def _write_sqlite(movies: list[EnrichedMovie]) -> None:
    if ENRICHED_DB.exists():
        ENRICHED_DB.unlink()
    conn = sqlite3.connect(ENRICHED_DB)
    try:
        conn.execute(
            """
            CREATE TABLE enriched (
              movie_id INTEGER PRIMARY KEY,
              title TEXT NOT NULL,
              overview TEXT,
              budget INTEGER,
              revenue INTEGER,
              runtime REAL,
              release_date TEXT,
              genres TEXT,
              n_ratings INTEGER,
              avg_rating REAL,
              overview_sentiment TEXT,
              budget_tier TEXT,
              revenue_tier TEXT,
              production_effectiveness_score INTEGER,
              production_effectiveness_label TEXT,
              themes TEXT,
              intended_audience TEXT,
              source TEXT,
              model TEXT,
              prompt_version TEXT,
              payload TEXT NOT NULL
            )
            """
        )
        rows = [
            (
                m.movie_id,
                m.title,
                m.overview,
                m.budget,
                m.revenue,
                m.runtime,
                m.release_date,
                json.dumps(m.genres),
                m.n_ratings,
                m.avg_rating,
                m.overview_sentiment,
                m.budget_tier,
                m.revenue_tier,
                m.production_effectiveness_score,
                m.production_effectiveness_label,
                json.dumps(m.themes),
                m.intended_audience,
                m.source,
                m.model,
                m.prompt_version,
                m.model_dump_json(),
            )
            for m in movies
        ]
        conn.executemany(
            """
            INSERT INTO enriched VALUES (
              ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
