from __future__ import annotations

from movie_system.models import EnrichedMovie, QueryIntent, SystemResponse
from movie_system.query_cache import last_lookup_hit, normalize_query
from movie_system.system import run_query


def _enable_cache(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "llm_query_cache.json"
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setenv("LLM_CACHE_PATH", str(cache_file))


def _fake_llm(monkeypatch) -> dict[str, int]:
    from movie_system import system as system_mod

    calls = {"n": 0}

    def fake_complete(system, user, model_cls, **_kwargs):
        calls["n"] += 1
        if model_cls is QueryIntent:
            return QueryIntent(
                task="recommend",
                genres=["Action"],
                revenue_tier="high",
                sentiment="positive",
            )
        return SystemResponse(task="recommend", answer="from-llm", movies=[])

    monkeypatch.setattr(system_mod, "complete_json", fake_complete)
    monkeypatch.setattr(system_mod, "llm_available", lambda: True)
    return calls


def test_repeat_query_does_not_call_openai(
    catalog: list[EnrichedMovie], tmp_path, monkeypatch
) -> None:
    _enable_cache(tmp_path, monkeypatch)
    calls = _fake_llm(monkeypatch)
    query = "Recommend action movies with high revenue and positive sentiment"

    first_intent, first_response, _ = run_query(query, catalog, use_llm=True)
    assert first_response.answer == "from-llm"
    assert "cache_hit" not in first_intent.notes
    paid = calls["n"]
    assert paid >= 1

    second_intent, second_response, _ = run_query(query, catalog, use_llm=True)
    assert calls["n"] == paid
    assert last_lookup_hit()
    assert "cache_hit" in second_intent.notes
    assert second_response.answer == first_response.answer


def test_case_and_spacing_share_a_cache_key(
    catalog: list[EnrichedMovie], tmp_path, monkeypatch
) -> None:
    _enable_cache(tmp_path, monkeypatch)
    calls = _fake_llm(monkeypatch)
    run_query(
        "Recommend action movies with high revenue and positive sentiment",
        catalog,
        use_llm=True,
    )
    paid = calls["n"]
    run_query(
        "  RECOMMEND action movies with high revenue and positive sentiment  ",
        catalog,
        use_llm=True,
    )
    assert calls["n"] == paid


def test_different_query_is_a_cache_miss(
    catalog: list[EnrichedMovie], tmp_path, monkeypatch
) -> None:
    _enable_cache(tmp_path, monkeypatch)
    calls = _fake_llm(monkeypatch)
    run_query("Recommend action movies with high revenue", catalog, use_llm=True)
    after_first = calls["n"]
    run_query("Recommend horror movies with low budget", catalog, use_llm=True)
    assert calls["n"] > after_first


def test_heuristic_path_does_not_use_cache(
    catalog: list[EnrichedMovie], tmp_path, monkeypatch
) -> None:
    _enable_cache(tmp_path, monkeypatch)
    calls = _fake_llm(monkeypatch)
    run_query("Recommend action movies", catalog, use_llm=False)
    assert calls["n"] == 0
    assert not last_lookup_hit()


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  Foo   BAR\n") == "foo bar"
