from __future__ import annotations

import json
import time
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from movie_system.config import openai_api_key, openai_model
from movie_system.observe import LlmCall, current_trace


class LLMError(RuntimeError):
    pass


_NULLISH = {"null", "none", "undefined"}


def normalize_llm_json(value: Any) -> Any:
    """Turn LLM 'null' strings into JSON null so Pydantic optional fields validate."""
    if isinstance(value, dict):
        return {key: normalize_llm_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_llm_json(item) for item in value]
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def llm_available() -> bool:
    return bool(openai_api_key())


def _client() -> OpenAI:
    key = openai_api_key()
    if not key:
        raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
    return OpenAI(api_key=key)


def complete_json(
    system: str,
    user: str,
    model_cls: type[BaseModel],
    *,
    temperature: float = 0.2,
    retries: int = 1,
) -> BaseModel:
    """Ask the model for JSON and validate it. One repair pass on schema errors."""
    client = _client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error: Optional[str] = None
    for _ in range(retries + 1):
        if last_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed validation:\n"
                        f"{last_error}\n"
                        "Return corrected JSON only, matching the schema."
                    ),
                }
            )
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=openai_model(),
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": content})
        usage = response.usage
        _record_llm_call(
            model_cls.__name__,
            latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            repaired=last_error is not None,
        )
        try:
            payload: dict[str, Any] = normalize_llm_json(json.loads(content))
            return model_cls.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
    raise LLMError(f"Could not get valid JSON from the model: {last_error}")


def _record_llm_call(
    purpose: str,
    latency_ms: int,
    *,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    repaired: bool,
) -> None:
    trace = current_trace()
    if trace is None:
        return
    trace.add_llm_call(
        LlmCall(
            purpose=purpose,
            model=openai_model(),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            repaired=repaired,
        )
    )


def movie_to_prompt_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "movie_id": row["movie_id"],
        "title": row["title"],
        "overview": row["overview"],
        "genres": row["genres"],
        "release_date": row.get("release_date"),
        "budget_usd": row["budget"],
        "revenue_usd": row["revenue"],
        "runtime_minutes": row.get("runtime"),
        "avg_user_rating": round(float(row["avg_rating"]), 2),
        "n_ratings": row["n_ratings"],
        "production_companies": row.get("production_companies", []),
    }
