from __future__ import annotations

# Short label + full query. Used as Chainlit starters and after each answer.
SAMPLE_PROMPTS: list[tuple[str, str]] = [
    (
        "Action + high revenue",
        "Recommend action movies with high revenue and positive sentiment",
    ),
    (
        "Family + medium budget",
        "Recommend family films with medium budget",
    ),
    (
        "Sci-fi hits",
        "Recommend science fiction movies with high revenue",
    ),
    (
        "Low-budget horror",
        "Recommend horror movies with low budget",
    ),
    (
        "Effective cheap films",
        "Find low budget movies that were still production-effective",
    ),
    (
        "User 564 tastes",
        "Summarize preferences for user 564 based on their ratings and movie overviews",
    ),
    (
        "User 15 tastes",
        "Summarize preferences for user 15 based on their ratings and movie overviews",
    ),
    (
        "User 547 tastes",
        "Summarize preferences for user 547 based on their ratings and movie overviews",
    ),
    (
        "Pulp Fiction vs Lost in Translation",
        "Compare budget, revenue, and runtime of 'Pulp Fiction' and 'Lost in Translation'",
    ),
    (
        "LOTR vs Saw",
        "Compare 'The Lord of the Rings: The Two Towers' and 'Saw'",
    ),
    (
        "Monsters vs Wonka",
        "Compare 'Monsters, Inc.' and 'Willy Wonka & the Chocolate Factory'",
    ),
    (
        "Predict user 15 on Pulp Fiction",
        "Predict the rating user 15 would give 'Pulp Fiction'",
    ),
    (
        "Predict user 564 on action",
        "Predict the rating user 564 would give a high-budget action movie",
    ),
    (
        "High PES + high revenue",
        "Find high revenue movies with a high production effectiveness score",
    ),
]


def prompts_except(current: str, limit: int | None = None) -> list[tuple[str, str]]:
    rest = [(label, q) for label, q in SAMPLE_PROMPTS if q != current]
    if limit is None:
        return rest
    return rest[:limit]


def followup_chips(current: str, limit: int | None = None) -> list[dict[str, str]]:
    return [
        {"label": label, "query": query}
        for label, query in prompts_except(current, limit=limit)
    ]
