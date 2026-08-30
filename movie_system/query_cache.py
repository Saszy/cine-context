from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from movie_system.config import DATA_DIR, openai_model
from movie_system.models import EnrichedMovie, QueryIntent, SystemResponse

CACHE_VERSION = 1
_lock = threading.Lock()
_tls = threading.local()


def cache_enabled() -> bool:
    raw = os.getenv("LLM_CACHE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def cache_path() -> Path:
    override = os.getenv("LLM_CACHE_PATH", "").strip()
    if override:
        return Path(override)
    return DATA_DIR / "llm_query_cache.json"


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split()).casefold()


def catalog_fingerprint(catalog: list[EnrichedMovie]) -> str:
    ids = ",".join(str(m.movie_id) for m in sorted(catalog, key=lambda m: m.movie_id))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def cache_key(query: str, catalog: list[EnrichedMovie]) -> str:
    payload = {
        "v": CACHE_VERSION,
        "query": normalize_query(query),
        "model": openai_model(),
        "catalog": catalog_fingerprint(catalog),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def last_lookup_hit() -> bool:
    return bool(getattr(_tls, "hit", False))


def mark_lookup(hit: bool) -> None:
    _tls.hit = hit


def lookup(
    query: str, catalog: list[EnrichedMovie]
) -> Optional[tuple[QueryIntent, SystemResponse, list[EnrichedMovie]]]:
    mark_lookup(False)
    if not cache_enabled():
        return None
    key = cache_key(query, catalog)
    with _lock:
        store = _read()
        entry = store.get("entries", {}).get(key)
    if not entry:
        return None
    by_id = {m.movie_id: m for m in catalog}
    retrieved_ids = [int(i) for i in entry.get("retrieved_ids", [])]
    if any(i not in by_id for i in retrieved_ids):
        return None
    try:
        intent = QueryIntent.model_validate(entry["intent"])
        response = SystemResponse.model_validate(entry["response"])
    except Exception:
        return None
    retrieved = [by_id[i] for i in retrieved_ids]
    mark_lookup(True)
    return intent, response, retrieved


def store(
    query: str,
    catalog: list[EnrichedMovie],
    intent: QueryIntent,
    response: SystemResponse,
    retrieved: list[EnrichedMovie],
) -> None:
    if not cache_enabled():
        return
    key = cache_key(query, catalog)
    entry = {
        "intent": intent.model_dump(),
        "response": response.model_dump(),
        "retrieved_ids": [m.movie_id for m in retrieved],
    }
    path = cache_path()
    with _lock:
        store_data = _read()
        entries: dict[str, Any] = store_data.setdefault("entries", {})
        entries[key] = entry
        _write(path, store_data)


def _read() -> dict[str, Any]:
    path = cache_path()
    if not path.exists():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CACHE_VERSION, "entries": {}}
    if not isinstance(payload, dict):
        return {"version": CACHE_VERSION, "entries": {}}
    payload.setdefault("entries", {})
    return payload


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
