from __future__ import annotations

import json
import re
from typing import Optional

from movie_system.guardrails import apply_response_guards, block_reason, refusal
from movie_system.llm import LLMError, complete_json, llm_available
from movie_system.observe import observe_request
from movie_system.query_cache import lookup as cache_lookup
from movie_system.query_cache import mark_lookup, store as cache_store
from movie_system.models import (
    EnrichedMovie,
    QueryIntent,
    RecommendationItem,
    SystemResponse,
    UserRatingRow,
)
from movie_system.prompts import GENERATE_SYSTEM, INTENT_SYSTEM
from movie_system.retrieve import context_payload, retrieve, user_ratings


def parse_intent(query: str, use_llm: bool) -> QueryIntent:
    if use_llm and llm_available():
        try:
            return complete_json(  # type: ignore[return-value]
                INTENT_SYSTEM,
                f"User query:\n{query}",
                QueryIntent,
                temperature=0,
            )
        except LLMError:
            parsed = _heuristic_intent(query)
            parsed.notes = (parsed.notes + " llm_intent_fallback").strip()
            return parsed
    return _heuristic_intent(query)


def _heuristic_intent(query: str) -> QueryIntent:
    q = query.lower()
    user_match = re.search(r"user\s+(\d+)", q)
    user_id = int(user_match.group(1)) if user_match else None

    if "summarize" in q and (user_id or "preference" in q):
        task = "user_summary"
    elif "compare" in q:
        task = "compare"
    elif "predict" in q or "would user" in q:
        task = "rating_predict"
    elif "recommend" in q:
        task = "recommend"
    else:
        task = "search"

    genres = []
    for name in (
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "Sci-Fi", "TV Movie", "Thriller", "War",
        "Western",
    ):
        if name.lower() in q:
            genres.append("Science Fiction" if name in {"Sci-Fi", "Science Fiction"} else name)

    sentiment = None
    for label in ("positive", "negative", "neutral", "mixed"):
        if f"{label} sentiment" in q or f"{label} overview" in q:
            sentiment = label
            break

    def _tier(word: str) -> Optional[str]:
        m = re.search(rf"{word}\s+(low|medium|high)", q)
        if m:
            return m.group(1)
        m = re.search(rf"(low|medium|high)\s+{word}", q)
        return m.group(1) if m else None

    budget_tier = _tier("budget")
    revenue_tier = _tier("revenue")
    if budget_tier is None:
        hyphen = re.search(r"(low|medium|high)[- ]budget", q)
        if hyphen:
            budget_tier = hyphen.group(1)
    if revenue_tier is None:
        hyphen = re.search(r"(low|medium|high)[- ]revenue", q)
        if hyphen:
            revenue_tier = hyphen.group(1)

    titles: list[str] = []
    quoted = re.findall(r"[\"']([^\"']+)[\"']", query)
    titles.extend(quoted)

    min_eff = 70 if any(w in q for w in ("effective", "successful production")) else None
    return QueryIntent(
        task=task,  # type: ignore[arg-type]
        genres=genres,
        sentiment=sentiment,  # type: ignore[arg-type]
        budget_tier=budget_tier,  # type: ignore[arg-type]
        revenue_tier=revenue_tier,  # type: ignore[arg-type]
        min_effectiveness=min_eff,
        movie_titles=titles,
        user_id=user_id,
        rating_examples_needed=task == "rating_predict",
        notes="heuristic parser",
    )


def _heuristic_generate(
    query: str,
    intent: QueryIntent,
    movies: list[EnrichedMovie],
    user_rows: list[UserRatingRow],
) -> SystemResponse:
    caveats = [
        "Generated without an LLM (heuristic fallback). Set OPENAI_API_KEY for model-written answers."
    ]
    if intent.task == "user_summary":
        if not user_rows:
            return SystemResponse(
                task="user_summary",
                answer=f"No ratings for user {intent.user_id} join to movies.movieId.",
                caveats=caveats
                + ["movieId in ratings.db often does not match movies.db (MovieLens vs TMDB ids)."],
            )
        liked = [r for r in user_rows if r.user_rating >= 4]
        disliked = [r for r in user_rows if r.user_rating <= 2.5]
        verified_liked = [r for r in liked if r.title_verified]
        liked_genres = sorted({g for r in liked for g in r.genres})[:6]
        examples = verified_liked[:4] if verified_liked else []
        if verified_liked:
            example_bit = f"Catalog-verified high ratings: {', '.join(r.title for r in examples)}."
        else:
            example_bit = (
                "No catalog-verified titles in this snapshot; "
                "joined names may not be the films the user actually rated."
            )
            caveats.append("Unverified titles omitted from the preference examples.")
        answer = (
            f"User {intent.user_id} has {len(user_rows)} matched titles in this snapshot "
            f"(high={len(liked)}, low={len(disliked)}). "
            f"Liked genres include {', '.join(liked_genres) or 'a mixed set'}. "
            f"{example_bit}"
        )
        items = [
            RecommendationItem(
                movie_id=r.movie_id,
                title=r.title,
                reason=f"User rated this {r.user_rating}/5 (catalog-verified).",
            )
            for r in verified_liked[:5]
        ]
        return SystemResponse(task="user_summary", answer=answer, movies=items, caveats=caveats)

    if not movies:
        return SystemResponse(
            task=intent.task,
            answer="No catalog movies matched those filters. Try dropping a tier or genre constraint.",
            caveats=caveats,
        )

    if intent.task == "compare":
        parts = []
        for m in movies[:3]:
            parts.append(
                f"{m.title}: budget ${m.budget:,.0f} ({m.budget_tier}), "
                f"revenue ${m.revenue:,.0f} ({m.revenue_tier}), "
                f"runtime {m.runtime} min, rating {m.avg_rating:.2f}, "
                f"PES {m.production_effectiveness_score}."
            )
        return SystemResponse(
            task="compare",
            answer=" ".join(parts),
            movies=[
                RecommendationItem(movie_id=m.movie_id, title=m.title, reason="Selected for comparison.")
                for m in movies[:3]
            ],
            caveats=caveats,
        )

    if intent.task == "rating_predict":
        examples = user_rows[:8] if user_rows else []
        baseline = (
            sum(r.user_rating for r in examples) / len(examples)
            if examples
            else movies[0].avg_rating
        )
        target = movies[0]
        predicted = round(0.6 * baseline + 0.4 * target.avg_rating, 1)
        predicted = min(5.0, max(0.5, predicted))
        return SystemResponse(
            task="rating_predict",
            answer=(
                f"Predicted rating for {target.title}: {predicted}/5, "
                f"blending the user's recent example mean ({baseline:.2f}) "
                f"with the catalog average ({target.avg_rating:.2f})."
            ),
            movies=[
                RecommendationItem(
                    movie_id=target.movie_id,
                    title=target.title,
                    reason="Prediction target.",
                )
            ],
            predicted_rating=predicted,
            caveats=caveats,
        )

    top = movies[:5]
    lines = [
        f"{m.title} (PES {m.production_effectiveness_score}, {m.overview_sentiment} overview, "
        f"{m.revenue_tier} revenue)"
        for m in top
    ]
    answer = f"Top matches for {query!r}: " + "; ".join(lines) + "."
    items = [
        RecommendationItem(
            movie_id=m.movie_id,
            title=m.title,
            reason=(
                f"{m.intended_audience}. Sentiment {m.overview_sentiment}, "
                f"revenue {m.revenue_tier}, PES {m.production_effectiveness_score}."
            ),
        )
        for m in top
    ]
    return SystemResponse(task=intent.task, answer=answer, movies=items, caveats=caveats)


def run_query(
    query: str,
    catalog: list[EnrichedMovie],
    *,
    use_llm: bool = True,
) -> tuple[QueryIntent, SystemResponse, list[EnrichedMovie]]:
    with observe_request(query, use_llm=use_llm) as trace:
        with trace.span("guardrails"):
            blocked = block_reason(query)
        if blocked:
            mark_lookup(False)
            trace.blocked = blocked
            intent = QueryIntent(task="search", notes=f"blocked:{blocked}")
            response = refusal("search", blocked)
            trace.fill(intent, response, [])
            return intent, response, []

        should_cache = use_llm and llm_available()
        if should_cache:
            with trace.span("cache"):
                cached = cache_lookup(query, catalog)
            if cached is not None:
                intent, response, retrieved = cached
                intent = intent.model_copy(
                    update={"notes": (intent.notes + " cache_hit").strip()}
                )
                trace.cache_hit = True
                trace.fill(intent, response, retrieved)
                return intent, response, retrieved
        else:
            mark_lookup(False)

        with trace.span("intent"):
            intent = parse_intent(query, use_llm=use_llm)
        if intent.task in {"user_summary", "rating_predict"} and intent.user_id is None:
            response = refusal(intent.task, "need_user")
            trace.blocked = "need_user"
            trace.fill(intent, response, [])
            return intent, response, []

        user_rows: list[UserRatingRow] = []
        if intent.user_id is not None:
            user_rows = user_ratings(intent.user_id, catalog)

        extra_caveats: list[str] = []
        retrieved: list[EnrichedMovie] = []
        with trace.span("retrieve"):
            if intent.task == "user_summary":
                extra_caveats.append(
                    "ratings.movieId is not a reliable join to movies.movieId "
                    "(MovieLens-style IDs vs TMDB-style IDs). Treat matched titles as noisy."
                )
            else:
                retrieved = retrieve(intent, catalog)
                if not retrieved and (
                    intent.genres or intent.revenue_tier or intent.sentiment
                ):
                    relaxed = intent.model_copy(
                        update={"sentiment": None, "min_effectiveness": None}
                    )
                    retrieved = retrieve(relaxed, catalog)
                    extra_caveats.append(
                        "No exact filter match; dropped sentiment and min-effectiveness constraints."
                    )

        skip_llm = (
            intent.task in {"recommend", "compare", "rating_predict"} and not retrieved
        )
        with trace.span("generate"):
            if use_llm and llm_available() and not skip_llm:
                user_payload = {
                    "query": query,
                    "intent": intent.model_dump(),
                    "catalog_matches": context_payload(retrieved),
                    "user_rating_examples": [r.model_dump() for r in user_rows[:10]],
                    "retrieval_notes": extra_caveats,
                }
                response = complete_json(  # type: ignore[assignment]
                    GENERATE_SYSTEM,
                    json.dumps(user_payload, indent=2),
                    SystemResponse,
                    temperature=0.3,
                )
            else:
                response = _heuristic_generate(query, intent, retrieved, user_rows)

        with trace.span("ground"):
            allowed = {m.movie_id for m in retrieved}
            if intent.task == "user_summary":
                allowed |= {r.movie_id for r in user_rows if r.title_verified}
            response = apply_response_guards(
                intent,
                response,
                allowed_ids=allowed,
                retrieved_n=len(retrieved),
            )
            if extra_caveats:
                response = response.model_copy(
                    update={"caveats": response.caveats + extra_caveats}
                )
        if should_cache:
            cache_store(query, catalog, intent, response, retrieved)
        trace.fill(intent, response, retrieved)
        return intent, response, retrieved
