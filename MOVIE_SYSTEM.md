# Movie system decision record

This file locks how the **Movie System Design** task works. [README.md](README.md) is the original spec and is not modified here. Enrichment rules live in [DATA_PREPARATION.md](DATA_PREPARATION.md). Run notes live in [SOLUTION.md](SOLUTION.md).

The system consumes the prepared catalog (`data/enriched_movies.json` / `data/enriched.db`) plus `db/ratings.db` for user-specific tasks. It does not mutate `db/movies.db`.

**Split:** code retrieves and grounds; the LLM writes structured answers from that context only. Same idea as data prep: math and filters in code, language in the model.

```
natural-language query
  -> parse intent JSON (task, filters, user_id, titles)
  -> retrieve (80-row catalog and/or that user's joined ratings)
  -> generate JSON {answer, movies[], caveats, predicted_rating}
  -> drop any movie_id that was not in CONTEXT
```

Prompt version for this stage: `movie-system-v1`. Intent temperature: 0. Generate temperature: 0.3. `source` of the answer is `llm` or `heuristic`.

---

## Goal (from the brief)

An LLM-integrated movie assistant that can:

1. Personalized **recommendations**
2. **User preference summaries** (ratings + overviews)
3. **Comparative analyses** (budget, revenue, runtime, plus enriched fields)
4. **Rating prediction** with 5–10 example ratings as few-shot context
5. Natural-language querying over the prepared attributes (sentiment, tiers, PES)

Required demo queries:

- `Recommend action movies with high revenue and positive sentiment`
- `Summarize preferences for user based on their ratings and movie overviews`

Interface: **Chainlit chatbot** (`chainlit run app.py`). The CLI (`python -m movie_system ask "..."`) stays for scripts and eval. Same retrieve-then-generate path either way.

---

## Architecture

| Option | Verdict |
|--------|---------|
| Stuff all 80 movies into every prompt | Rejected. Wastes tokens; harder to prove grounding; does not scale. |
| Embeddings / vector search | Rejected. 80 rows; filters are structured (genre, tier, sentiment). Extra dependency without a grounding win. |
| Agent with free-form SQL tools | Rejected for this exercise. Harder to eval; easy to over-fetch. |
| Retrieve-then-generate: intent JSON → filter catalog → generate JSON | **Chosen.** |

Retrieval is in-memory over the 80-row catalog (loaded from JSON). User ratings are loaded with SQL against `ratings` ⋈ `movies` when `user_id` is present. Rule tiers from data prep are the filter fields; optional LLM tier overrides are commentary only.

---

## Tasks

| Task | When | Context | Output |
|------|------|---------|--------|
| `recommend` | “Recommend…” | filtered catalog, top 12 by PES then rating | 3–5 titles + reasons citing CONTEXT fields |
| `compare` | “Compare…” | named titles (quoted), else empty | side-by-side budget, revenue, runtime, rating, PES, sentiment |
| `user_summary` | “Summarize preferences for user N” | up to ~10 high and ~10 low joined ratings; enrich when the title is in the 80 | taste summary; do not invent users |
| `rating_predict` | “Predict rating…” / “what would user N rate” | 5–10 of that user’s ratings as examples + target movie from catalog | one `predicted_rating` on the 0.5–5 scale |
| `search` | anything else | same filters as recommend | short catalog answer, still grounded |

**Locked catalog vs ratings**

- Recommendations, compare, and search stay **inside the 80 enriched movies**.
- User summaries may include titles **outside** the 80 (joined to full `movies`) so preference text is not starved. Attach enrichment when `movie_id` is in the sample.
- Always caveat that `ratings.movieId` is a noisy MovieLens vs TMDB join.

---

### Recommendations

| Option | Verdict |
|--------|---------|
| Collaborative filtering / matrix factorization | Rejected. Out of scope; ID join is already broken. |
| LLM recommends from world knowledge | Rejected. Hallucinated titles. |
| Filter catalog on intent, rank by PES / rating, LLM writes reasons | **Chosen.** |

If filters match nothing: say so, then relax **sentiment** and **min_effectiveness** only, and put that in `caveats`. Do not invent movies. Do not silently drop genre or revenue/budget tier.

Heuristic sentiment is often `neutral` until `prepare --llm` is run. The brief’s “positive sentiment” query will likely hit the relax path; that is expected and must be visible in caveats.

---

### User preference summaries

| Option | Verdict |
|--------|---------|
| Summarize only the 80-movie overlap | Rejected as the only path. Many users would have too few enriched titles. |
| Dump all of a user’s ratings into the prompt | Rejected. User 564 has hundreds of joined rows. |
| Extreme ratings snapshot (high and low) + overviews, cap ~24 | **Chosen.** |

The generate prompt must use overviews and ratings, not just genre counts. Mention when the user’s score disagrees with catalog `avg_rating`.

---

### Comparative analysis

Named titles come from quoted strings in the query (`'Pulp Fiction'`). If a title is not in the 80-row catalog, return empty retrieve and say so — **do not** fall back to top-PES movies.

Compare using: budget, revenue, runtime, avg rating, PES, sentiment, tiers. LLM (or heuristic) writes the prose; numbers come from CONTEXT.

---

### Rating prediction (few-shot)

| Option | Verdict |
|--------|---------|
| Train a regressor | Rejected. Wrong grain for 2–3 hours; noisy IDs. |
| LLM predicts with no examples | Rejected. The brief asks for 5–10 example ratings. |
| Pass 5–10 of the user’s ratings + target movie stats; LLM returns one number | **Chosen.** |

Heuristic fallback may blend example-mean and catalog average; label it as heuristic. Clip to 0.5–5.

---

## Prompting

Two-stage chain (not a single kitchen-sink prompt):

1. **Intent** — JSON schema (`QueryIntent`). Temperature 0. Heuristic parser implements the same schema so the pipeline runs without an API key.
2. **Generate** — JSON schema (`SystemResponse`). Temperature 0.3. Payload = query + intent + `catalog_matches` + up to 10 `user_rating_examples` + retrieval notes.

Techniques to demonstrate:

- System vs user split
- JSON-only (`response_format=json_object`) + Pydantic validate + one repair pass
- Few-shot ratings in the generate payload (not in the system prompt)
- Grounding rule: never mention a title that is not in CONTEXT
- Explicit empty-context behavior

---

## Structured output

```json
{
  "task": "recommend | compare | user_summary | rating_predict | search",
  "answer": "2-5 sentences for the user",
  "movies": [{"movie_id": 0, "title": "exact title", "reason": "one sentence"}],
  "caveats": ["data or join caveats"],
  "predicted_rating": null
}
```

After generate, **code** drops any `movies[]` entry whose `movie_id` is not in retrieved catalog IDs or user-rating IDs.

---

## Evaluation

`python -m movie_system eval` for this stage:

- Golden intent parses for the brief queries plus compare / predict / search
- Recommendation `movie_id`s ⊆ retrieved IDs
- Non-empty `answer` for recommend and user_summary
- Compare with unknown titles → empty retrieve, no invented substitutes
- Rating predict returns `predicted_rating` in 0.5–5 when a target exists

This is schema + grounding eval, not a large human rubric. Hallucinated titles are the failure mode that breaks a catalog assistant.

---

## Observability

Local structured traces, not a hosted APM.

Each `run_query` emits one JSON line to `data/query_traces.jsonl` and a one-line `movie_system` log: task, engine, cache hit, retrieved count, wall time, prompt/completion tokens. Spans cover guardrails → cache → intent → retrieve → generate → ground. Secret queries are stored as `[redacted]`. Chainlit footer shows latency and tokens. `python -m movie_system traces` prints the tail. `OBSERVE=0` disables writes.

---

## Non-goals

- Auth, hosted deploy, or a custom React UI (Chainlit is the demo shell)
- Embeddings / LangChain
- Writing into `db/movies.db`
- Using enrichment tier overrides as retrieve filters
- Fixing the MovieLens vs TMDB `movieId` map (document only)

---

## Files

| File | Role |
|------|------|
| `movie_system/guardrails.py` | secret/injection blocks, rating clamp, ID grounding |
| `movie_system/system.py` | intent + generate + grounding |
| `movie_system/retrieve.py` | catalog filters + user rating snapshot |
| `movie_system/prompts.py` | `INTENT_SYSTEM`, `GENERATE_SYSTEM` |
| `movie_system/models.py` | `QueryIntent`, `SystemResponse` |
| `movie_system/evaluate.py` | golden queries + grounding checks |
| `movie_system/observe.py` | request traces + JSONL |
| `movie_system/__main__.py` | `ask` / `prepare` / `eval` / `traces` CLI |
| `app.py` | Chainlit chatbot over `run_query` |

Re-run `prepare --llm` before demoing recommendations that need real overview sentiment.
