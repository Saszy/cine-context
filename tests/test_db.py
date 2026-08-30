from __future__ import annotations

from movie_system.db import parse_named_list


def test_parse_json_genres() -> None:
    raw = '[{"id": 28, "name": "Action"}, {"id": 18, "name": "Drama"}]'
    assert parse_named_list(raw) == ["Action", "Drama"]


def test_parse_pipe_genres() -> None:
    assert parse_named_list("Action|Drama") == ["Action", "Drama"]


def test_parse_empty() -> None:
    assert parse_named_list(None) == []
    assert parse_named_list("") == []
