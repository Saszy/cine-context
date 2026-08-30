from __future__ import annotations

import random
from collections import defaultdict

from movie_system.config import (
    MIN_FINANCIALS,
    MIN_OVERVIEW_LEN,
    MIN_RATINGS,
    SAMPLE_SEED,
    SAMPLE_SIZE,
)
from movie_system.db import connect, parse_named_list
from movie_system.models import MovieRecord

POOL_SQL = """
SELECT
  m.movieId AS movie_id,
  m.imdbId AS imdb_id,
  m.title,
  m.overview,
  m.productionCompanies AS production_companies_raw,
  m.releaseDate AS release_date,
  m.budget,
  m.revenue,
  m.runtime,
  m.language,
  m.genres AS genres_raw,
  m.status,
  COUNT(r.rating) AS n_ratings,
  AVG(r.rating) AS avg_rating
FROM movies m
JOIN ratings_db.ratings r ON r.movieId = m.movieId
WHERE m.status = 'Released'
  AND m.budget >= :min_fin
  AND m.revenue >= :min_fin
  AND m.overview IS NOT NULL
  AND length(m.overview) >= :min_overview
GROUP BY m.movieId
HAVING n_ratings >= :min_ratings
"""


def _row_to_record(row) -> MovieRecord:
    return MovieRecord(
        movie_id=row["movie_id"],
        imdb_id=row["imdb_id"],
        title=row["title"],
        overview=row["overview"],
        production_companies=parse_named_list(row["production_companies_raw"]),
        release_date=row["release_date"] or None,
        budget=int(row["budget"]),
        revenue=int(row["revenue"]),
        runtime=row["runtime"],
        language=row["language"] or None,
        genres=parse_named_list(row["genres_raw"]),
        status=row["status"],
        n_ratings=int(row["n_ratings"]),
        avg_rating=float(row["avg_rating"]),
    )


def load_pool() -> list[MovieRecord]:
    conn = connect()
    try:
        rows = conn.execute(
            POOL_SQL,
            {
                "min_fin": MIN_FINANCIALS,
                "min_overview": MIN_OVERVIEW_LEN,
                "min_ratings": MIN_RATINGS,
            },
        ).fetchall()
        return [_row_to_record(row) for row in rows]
    finally:
        conn.close()


def stratified_sample(
    pool: list[MovieRecord],
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[MovieRecord]:
    """Round-robin by primary genre so the catalog is not all Action blockbusters."""
    rng = random.Random(seed)
    buckets: dict[str, list[MovieRecord]] = defaultdict(list)
    for movie in pool:
        primary = movie.genres[0] if movie.genres else "Unknown"
        buckets[primary].append(movie)
    for key in buckets:
        rng.shuffle(buckets[key])

    selected: list[MovieRecord] = []
    seen: set[int] = set()
    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(selected) < min(size, len(pool)):
        progressed = False
        for key in keys:
            if not buckets[key]:
                continue
            movie = buckets[key].pop()
            if movie.movie_id in seen:
                continue
            selected.append(movie)
            seen.add(movie.movie_id)
            progressed = True
            if len(selected) >= size:
                break
        if not progressed:
            break
    selected.sort(key=lambda m: m.movie_id)
    return selected


def sample_movies(size: int = SAMPLE_SIZE) -> list[MovieRecord]:
    return stratified_sample(load_pool(), size=size)
