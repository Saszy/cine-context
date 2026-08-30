from __future__ import annotations

from movie_system.models import EnrichedMovie, QueryIntent, UserRatingRow
from movie_system.db import connect, parse_named_list
from movie_system.enrich import load_enriched


def load_catalog() -> list[EnrichedMovie]:
    return load_enriched()


def _genre_hit(movie: EnrichedMovie, wanted: list[str]) -> bool:
    if not wanted:
        return True
    have = {g.lower() for g in movie.genres}
    return any(w.lower() in have for w in wanted)


def retrieve(intent: QueryIntent, catalog: list[EnrichedMovie]) -> list[EnrichedMovie]:
    hits = list(catalog)

    if intent.movie_titles:
        titles = [t.lower() for t in intent.movie_titles]
        named = [
            m
            for m in catalog
            if any(t in m.title.lower() or m.title.lower() in t for t in titles)
        ]
        return named[:8]

    if intent.genres:
        hits = [m for m in hits if _genre_hit(m, intent.genres)]
    if intent.sentiment:
        hits = [m for m in hits if m.overview_sentiment == intent.sentiment]
    if intent.budget_tier:
        hits = [m for m in hits if m.budget_tier == intent.budget_tier]
    if intent.revenue_tier:
        hits = [m for m in hits if m.revenue_tier == intent.revenue_tier]
    if intent.min_effectiveness is not None:
        hits = [
            m
            for m in hits
            if m.production_effectiveness_score >= intent.min_effectiveness
        ]

    hits.sort(
        key=lambda m: (m.production_effectiveness_score, m.avg_rating, m.revenue),
        reverse=True,
    )
    return hits[:12]


def context_payload(movies: list[EnrichedMovie]) -> list[dict]:
    rows = []
    for m in movies:
        rows.append(
            {
                "movie_id": m.movie_id,
                "title": m.title,
                "genres": m.genres,
                "overview": m.overview[:400],
                "budget": m.budget,
                "revenue": m.revenue,
                "runtime": m.runtime,
                "avg_rating": round(m.avg_rating, 2),
                "n_ratings": m.n_ratings,
                "overview_sentiment": m.overview_sentiment,
                "budget_tier": m.budget_tier,
                "revenue_tier": m.revenue_tier,
                "production_effectiveness_score": m.production_effectiveness_score,
                "production_effectiveness_label": m.production_effectiveness_label,
                "themes": m.themes,
                "intended_audience": m.intended_audience,
            }
        )
    return rows


def user_ratings(user_id: int, catalog: list[EnrichedMovie], limit: int = 24) -> list[UserRatingRow]:
    """Join this user's ratings to movies. Enrichment is attached when the title is in the sample."""
    by_id = {m.movie_id: m for m in catalog}
    sql = """
      SELECT
        m.movieId AS movie_id,
        m.title,
        m.overview,
        m.genres AS genres_raw,
        m.budget,
        m.revenue,
        r.rating AS user_rating
      FROM ratings_db.ratings r
      JOIN movies m ON m.movieId = r.movieId
      WHERE r.userId = ?
      ORDER BY r.rating DESC, m.title
    """
    conn = connect()
    try:
        rows = conn.execute(sql, (user_id,)).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    parsed: list[UserRatingRow] = []
    for row in rows:
        extra = by_id.get(row["movie_id"])
        parsed.append(
            UserRatingRow(
                movie_id=row["movie_id"],
                title=row["title"],
                genres=parse_named_list(row["genres_raw"]),
                overview=(row["overview"] or "")[:280],
                user_rating=float(row["user_rating"]),
                avg_rating=extra.avg_rating if extra else None,
                budget=row["budget"],
                revenue=row["revenue"],
                enrichment=(
                    {
                        "overview_sentiment": extra.overview_sentiment,
                        "themes": extra.themes,
                        "production_effectiveness_score": extra.production_effectiveness_score,
                    }
                    if extra
                    else None
                ),
                title_verified=extra is not None,
            )
        )

    high = [p for p in parsed if p.user_rating >= 4.0][: limit // 2]
    low = [p for p in parsed if p.user_rating <= 2.5][: limit // 2]
    picked_ids = {p.movie_id for p in high + low}
    if len(high) + len(low) < limit:
        for p in parsed:
            if p.movie_id not in picked_ids:
                high.append(p)
                picked_ids.add(p.movie_id)
            if len(high) + len(low) >= limit:
                break
    return high + low
