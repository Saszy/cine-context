from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from movie_system.config import DB_MOVIES, DB_RATINGS


def parse_named_list(raw: Optional[str]) -> list[str]:
    """Genres/companies are JSON in this dump; the brief described pipes."""
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [part.strip() for part in raw.split("|") if part.strip()]
    if isinstance(data, list):
        names: list[str] = []
        for item in data:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
            elif isinstance(item, str) and item.strip():
                names.append(item.strip())
        return names
    return []


def connect(readonly: bool = True) -> sqlite3.Connection:
    uri = f"file:{DB_MOVIES}?mode=ro" if readonly else str(DB_MOVIES)
    conn = sqlite3.connect(uri, uri=readonly, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{DB_RATINGS}' AS ratings_db")
    return conn


def connect_file(path: Any, readonly: bool = True) -> sqlite3.Connection:
    path = str(path)
    uri = f"file:{path}?mode=ro" if readonly else path
    conn = sqlite3.connect(uri, uri=readonly, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
