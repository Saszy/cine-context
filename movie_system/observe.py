from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from movie_system.config import DATA_DIR
from movie_system.models import EnrichedMovie, QueryIntent, SystemResponse

logger = logging.getLogger("movie_system")

_current: ContextVar[Optional["RequestTrace"]] = ContextVar(
    "movie_request_trace", default=None
)
_last_lock = threading.Lock()
_last: Optional["RequestTrace"] = None


def observe_enabled() -> bool:
    raw = os.getenv("OBSERVE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def traces_path() -> Path:
    override = os.getenv("OBSERVE_PATH", "").strip()
    if override:
        return Path(override)
    return DATA_DIR / "query_traces.jsonl"


def current_trace() -> Optional["RequestTrace"]:
    return _current.get()


def last_trace() -> Optional["RequestTrace"]:
    with _last_lock:
        return _last


def safe_query(query: str, blocked: Optional[str] = None) -> str:
    if blocked == "secrets":
        return "[redacted]"
    text = " ".join(query.split())
    if len(text) > 200:
        return text[:200] + "…"
    return text


@dataclass
class LlmCall:
    purpose: str
    model: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    repaired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "repaired": self.repaired,
        }


@dataclass
class RequestTrace:
    request_id: str
    ts: str
    query: str
    use_llm: bool
    blocked: Optional[str] = None
    cache_hit: bool = False
    task: Optional[str] = None
    engine: str = "heuristic"
    retrieved_n: int = 0
    movie_ids: list[int] = field(default_factory=list)
    caveats_n: int = 0
    latency_ms: int = 0
    spans: dict[str, int] = field(default_factory=dict)
    llm_calls: list[LlmCall] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens or 0 for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens or 0 for c in self.llm_calls)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.spans[name] = int((time.perf_counter() - started) * 1000)

    def add_llm_call(self, call: LlmCall) -> None:
        self.llm_calls.append(call)

    def fill(
        self,
        intent: QueryIntent,
        response: SystemResponse,
        retrieved: list[EnrichedMovie],
    ) -> None:
        self.task = intent.task
        self.retrieved_n = len(retrieved)
        self.movie_ids = [m.movie_id for m in retrieved]
        self.caveats_n = len(response.caveats)
        if self.blocked:
            self.engine = "refused"
        elif self.cache_hit:
            self.engine = "cached"
        elif self.llm_calls:
            self.engine = "llm"
        else:
            self.engine = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ts": self.ts,
            "query": self.query,
            "use_llm": self.use_llm,
            "blocked": self.blocked,
            "cache_hit": self.cache_hit,
            "task": self.task,
            "engine": self.engine,
            "retrieved_n": self.retrieved_n,
            "movie_ids": self.movie_ids,
            "caveats_n": self.caveats_n,
            "latency_ms": self.latency_ms,
            "spans": self.spans,
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


@contextmanager
def observe_request(query: str, *, use_llm: bool) -> Iterator[RequestTrace]:
    trace = RequestTrace(
        request_id=uuid.uuid4().hex[:12],
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        query=safe_query(query),
        use_llm=use_llm,
    )
    token = _current.set(trace)
    started = time.perf_counter()
    try:
        yield trace
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace.latency_ms = int((time.perf_counter() - started) * 1000)
        if trace.blocked == "secrets":
            trace.query = "[redacted]"
        _current.reset(token)
        _remember(trace)
        emit(trace)


def emit(trace: RequestTrace) -> None:
    if not observe_enabled():
        return
    logger.info(
        "query id=%s task=%s engine=%s retrieved=%s cache=%s %sms tokens=%s/%s",
        trace.request_id,
        trace.task or "-",
        trace.engine,
        trace.retrieved_n,
        trace.cache_hit,
        trace.latency_ms,
        trace.prompt_tokens,
        trace.completion_tokens,
    )
    path = traces_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), separators=(",", ":")) + "\n")
    except OSError:
        logger.warning("Could not write trace to %s", path)


def read_traces(limit: int = 20) -> list[dict[str, Any]]:
    path = traces_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _remember(trace: RequestTrace) -> None:
    global _last
    with _last_lock:
        _last = trace
