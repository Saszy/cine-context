from __future__ import annotations

from dataclasses import dataclass, field

from movie_system.enrich import compute_facts, load_enriched
from movie_system.models import EnrichedMovie
from movie_system.system import parse_intent, run_query

GOLDEN_INTENTS = [
    {
        "query": "Recommend action movies with high revenue and positive sentiment",
        "task": "recommend",
        "genres": ["Action"],
        "revenue_tier": "high",
        "sentiment": "positive",
    },
    {
        "query": "Summarize preferences for user 564 based on their ratings and movie overviews",
        "task": "user_summary",
        "user_id": 564,
    },
    {
        "query": "Compare budget, revenue, and runtime of 'Pulp Fiction' and 'Lost in Translation'",
        "task": "compare",
    },
    {
        "query": "Predict the rating user 15 would give a high-budget action movie",
        "task": "rating_predict",
        "user_id": 15,
        "budget_tier": "high",
    },
    {
        "query": "Find low budget movies that were still production-effective",
        "task": "search",
        "budget_tier": "low",
    },
    {
        "query": "Recommend family films with medium budget",
        "task": "recommend",
        "genres": ["Family"],
        "budget_tier": "medium",
    },
]


@dataclass
class EvalReport:
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def total(self) -> int:
        return len(self.checks)


def _intent_ok(query: str, expected: dict, use_llm: bool) -> tuple[bool, str]:
    intent = parse_intent(query, use_llm=use_llm)
    problems = []
    if intent.task != expected["task"]:
        problems.append(f"task={intent.task} expected={expected['task']}")
    if "user_id" in expected and intent.user_id != expected["user_id"]:
        problems.append(f"user_id={intent.user_id}")
    if "genres" in expected:
        got = {g.lower() for g in intent.genres}
        need = {g.lower() for g in expected["genres"]}
        if not need.issubset(got):
            problems.append(f"genres={intent.genres}")
    if "revenue_tier" in expected and intent.revenue_tier != expected["revenue_tier"]:
        problems.append(f"revenue_tier={intent.revenue_tier}")
    if "budget_tier" in expected and intent.budget_tier != expected["budget_tier"]:
        problems.append(f"budget_tier={intent.budget_tier}")
    if "sentiment" in expected and intent.sentiment != expected["sentiment"]:
        problems.append(f"sentiment={intent.sentiment}")
    return (not problems, "; ".join(problems) or "ok")


def evaluate_catalog(movies: list[EnrichedMovie]) -> EvalReport:
    report = EvalReport()
    report.add("sample_size_50_100", 50 <= len(movies) <= 100, f"n={len(movies)}")
    ids = [m.movie_id for m in movies]
    report.add("unique_ids", len(ids) == len(set(ids)), f"n={len(ids)}")
    report.add(
        "all_have_five_attributes",
        all(
            m.overview_sentiment
            and m.budget_tier
            and m.revenue_tier
            and m.production_effectiveness_score is not None
            and m.themes
            for m in movies
        ),
    )
    report.add(
        "pes_in_range",
        all(0 <= m.production_effectiveness_score <= 100 for m in movies),
    )
    report.add(
        "financials_present",
        all(m.budget > 0 and m.revenue > 0 for m in movies),
    )
    tier_ok = True
    pes_ok = True
    heuristic_neutral = True
    details = []
    for m in movies:
        facts = compute_facts(m)
        if m.budget_tier != facts.budget_tier or m.revenue_tier != facts.revenue_tier:
            tier_ok = False
            details.append(f"tier {m.movie_id}")
        if m.production_effectiveness_score != facts.production_effectiveness_score:
            pes_ok = False
            details.append(f"pes {m.movie_id}")
        if m.production_effectiveness_label != facts.production_effectiveness_label:
            pes_ok = False
            details.append(f"label {m.movie_id}")
        if m.source == "heuristic" and m.overview_sentiment != "neutral":
            heuristic_neutral = False
    report.add("tiers_match_rule_bands", tier_ok, "; ".join(details[:5]))
    report.add("pes_matches_formula", pes_ok, "; ".join(details[:5]))
    report.add("heuristic_sentiment_is_neutral", heuristic_neutral)
    return report


def evaluate_system(movies: list[EnrichedMovie], *, use_llm: bool) -> EvalReport:
    report = EvalReport()
    for case in GOLDEN_INTENTS:
        ok, detail = _intent_ok(case["query"], case, use_llm=False)
        report.add(f"intent::{case['task']}::{case['query'][:32]}", ok, detail)

    demo = GOLDEN_INTENTS[0]["query"]
    intent, response, retrieved = run_query(demo, movies, use_llm=use_llm)
    allowed = {m.movie_id for m in retrieved}
    grounded = all(item.movie_id in allowed for item in response.movies) if response.movies else True
    report.add("recommend_grounded_ids", grounded, f"k={len(response.movies)}")
    report.add("recommend_nonempty_answer", bool(response.answer.strip()))
    report.add("intent_is_recommend", intent.task == "recommend", intent.task)

    _, user_resp, _ = run_query(GOLDEN_INTENTS[1]["query"], movies, use_llm=use_llm)
    report.add("user_summary_nonempty", bool(user_resp.answer.strip()), user_resp.task)
    return report


def run_eval(*, use_llm: bool = False) -> dict:
    movies = load_enriched()
    catalog = evaluate_catalog(movies)
    system = evaluate_system(movies, use_llm=use_llm)
    checks = catalog.checks + system.checks
    return {
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "checks": checks,
    }
