from __future__ import annotations

import re
from typing import Optional

from movie_system.models import QueryIntent, RecommendationItem, SystemResponse

SECRET_RE = re.compile(
    r"api[\s_-]?key|openai_api_key|\.env\b|auth.?secret|credentials|secret key",
    re.I,
)
DUMP_RE = re.compile(r"\b(print|dump|reveal|leak|expose|show me)\b", re.I)
INJECTION_RE = re.compile(
    r"ignore (all )?(previous |prior |the )?(instructions|context)|"
    r"from your training data|"
    r"you are (now )?(a |an )?(pirate|dan|jailbreak)|"
    r"forget (the )?(catalog|rules|instructions)",
    re.I,
)

REFUSALS = {
    "secrets": (
        "I can't reveal API keys, environment files, or other secrets. "
        "Ask a movie-catalog question (recommend, compare, or summarize a user)."
    ),
    "injection": (
        "I only answer from the prepared movie catalog. "
        "I won't ignore CONTEXT or recommend titles that are not in it. "
        "Try a recommendation, comparison, or user summary."
    ),
    "need_user": (
        "Please include a numeric user id, for example "
        "'Summarize preferences for user 564' or "
        "'Predict the rating user 15 would give Pulp Fiction'."
    ),
}


def block_reason(query: str) -> Optional[str]:
    q = query.strip()
    if SECRET_RE.search(q) and (DUMP_RE.search(q) or ".env" in q.lower()):
        return "secrets"
    if SECRET_RE.search(q) and not any(
        w in q.lower() for w in ("recommend", "compare", "summarize", "predict", "movie")
    ):
        return "secrets"
    if INJECTION_RE.search(q):
        return "injection"
    return None


def refusal(task: str, kind: str) -> SystemResponse:
    return SystemResponse(
        task=task,  # type: ignore[arg-type]
        answer=REFUSALS[kind],
        movies=[],
        caveats=[f"blocked:{kind}"],
        predicted_rating=None,
    )


def clamp_rating(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return min(5.0, max(0.5, round(float(value), 1)))


def ground_movies(
    movies: list[RecommendationItem],
    allowed_ids: set[int],
) -> tuple[list[RecommendationItem], int]:
    kept = [item for item in movies if item.movie_id in allowed_ids]
    return kept, len(movies) - len(kept)


def apply_response_guards(
    intent: QueryIntent,
    response: SystemResponse,
    *,
    allowed_ids: set[int],
    retrieved_n: int,
) -> SystemResponse:
    movies, dropped = ground_movies(response.movies, allowed_ids)
    caveats = list(response.caveats)
    if dropped:
        caveats.append(f"Dropped {dropped} ungrounded title(s) not present in CONTEXT.")

    if intent.task in {"recommend", "compare"} and retrieved_n == 0:
        movies = []
        if "No catalog movies matched" not in response.answer:
            response = response.model_copy(
                update={
                    "answer": (
                        "No catalog movies matched those filters. "
                        "I will not invent titles outside the prepared set."
                    )
                }
            )

    predicted = clamp_rating(response.predicted_rating)
    if intent.task != "rating_predict":
        predicted = None

    return response.model_copy(
        update={"movies": movies, "caveats": caveats, "predicted_rating": predicted}
    )
