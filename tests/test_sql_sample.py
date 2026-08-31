from __future__ import annotations

from movie_system.config import MIN_FINANCIALS, MIN_OVERVIEW_LEN, MIN_RATINGS
from movie_system.sample import load_pool


def test_sample_pool_is_a_real_sql_join() -> None:
    """Data handling + SQL: ATTACH ratings.db, JOIN, GROUP BY, HAVING."""
    pool = load_pool()
    assert len(pool) >= 80
    assert all(m.n_ratings >= MIN_RATINGS for m in pool)
    assert all(m.budget >= MIN_FINANCIALS and m.revenue >= MIN_FINANCIALS for m in pool)
    assert all(m.overview and len(m.overview) >= MIN_OVERVIEW_LEN for m in pool)
    assert all(m.status == "Released" for m in pool)
    assert len({m.movie_id for m in pool}) == len(pool)
