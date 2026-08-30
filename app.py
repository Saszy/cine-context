from __future__ import annotations

import asyncio

import chainlit as cl

from movie_system.chainlit_compat import patch_local_steps
from movie_system.enrich import load_enriched
from movie_system.llm import llm_available
from movie_system.models import EnrichedMovie, QueryIntent, SystemResponse
from movie_system.observe import last_trace
from movie_system.sample_prompts import SAMPLE_PROMPTS, followup_chips
from movie_system.system import run_query

patch_local_steps()


def format_reply(
    intent: QueryIntent,
    response: SystemResponse,
    retrieved: list[EnrichedMovie],
    used_llm: bool,
) -> str:
    lines = [response.answer, ""]
    if response.predicted_rating is not None:
        lines.append(f"**Predicted rating:** {response.predicted_rating}/5")
        lines.append("")
    if response.movies:
        lines.append("**Titles**")
        for item in response.movies:
            lines.append(f"- **{item.title}** (`{item.movie_id}`): {item.reason}")
        lines.append("")
    if response.caveats:
        lines.append("**Caveats**")
        for caveat in response.caveats:
            lines.append(f"- {caveat}")
        lines.append("")
    if "cache_hit" in intent.notes:
        engine = "OpenAI (cached)"
    elif used_llm:
        engine = "OpenAI"
    else:
        engine = "heuristic"
    bits = [f"task=`{intent.task}`", f"retrieved={len(retrieved)}"]
    trace = last_trace()
    if trace is not None:
        bits.append(f"{trace.latency_ms}ms")
        tokens = trace.prompt_tokens + trace.completion_tokens
        if tokens:
            bits.append(f"{tokens} tok")
    bits.append(engine)
    lines.append(f"_{' · '.join(bits)}_")
    return "\n".join(lines)


async def handle_query(text: str) -> None:
    catalog = cl.user_session.get("catalog")
    if catalog is None:
        catalog = await asyncio.to_thread(load_enriched)
        cl.user_session.set("catalog", catalog)
    use_llm = llm_available()
    try:
        intent, response, retrieved = await asyncio.to_thread(
            run_query, text, catalog, use_llm=use_llm
        )
        body = format_reply(intent, response, retrieved, use_llm)
    except Exception as exc:
        body = f"Could not answer that. {exc}"
    await cl.Message(
        content=body,
        elements=[
            cl.CustomElement(
                name="PromptChips",
                props={"prompts": followup_chips(text)},
                display="inline",
            )
        ],
    ).send()


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(label=label, message=query) for label, query in SAMPLE_PROMPTS
    ]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # Do not use on_chat_start: Chainlit 2.3 + Python 3.9 raises ContextVar LookupError.
    await handle_query(message.content)
