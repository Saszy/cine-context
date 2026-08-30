# Data preparation decision record

This file locks how we generate the brief’s data-enrichment attributes. [README.md](README.md) is the original spec and is not modified here. System-level notes live in [SOLUTION.md](SOLUTION.md).

**Split:** code owns numbers and cutoffs (reproducible, testable). The LLM owns language judgment and short rationales (what “use prompts” means). The model must not invent budget, revenue, or rating.

```
sample 80 movies
  -> compute facts in code (tiers, ROI, PES, label)
  -> LLM JSON (sentiment, rationales, themes, optional tier overrides)
  -> schema validate + repair
  -> persist JSON + SQLite (never mutate db/movies.db or db/ratings.db)
```

Prompt version: `data-prep-v1`. Batch size: 8. Temperature: 0.2. `source` is `llm` or `heuristic`.

## Sample (inputs we require)

80 **Released** titles, stratified by primary genre (seed 42), with:

- overview length ≥ 40
- budget ≥ $1,000 and revenue ≥ $1,000
- at least 5 joined ratings (`ratings.movieId = movies.movieId`)

`avg_rating` comes from that join. The ID space is noisy (MovieLens vs TMDB); we still require it so PES has a rating, and we treat it as a data-quality caveat rather than a blocker.

Rows that fail these filters are not enriched.

## Five attributes

| # | Attribute | Owner |
|---|-----------|--------|
| 1 | `overview_sentiment` | LLM (heuristic: `neutral`) |
| 2 | `budget_tier` | Code (rule bands) |
| 3 | `revenue_tier` | Code (rule bands) |
| 4 | `production_effectiveness_score` | Code (formula) |
| 5 | `themes` + `intended_audience` | LLM (heuristic: genres) |

---

### 1. Overview sentiment

| Option | Verdict |
|--------|---------|
| Keyword lists on plot words (`murder`, `death`, `love`, …) | Rejected. Confuses plot content with tone (horror overviews look “negative”). |
| Dedicated sentiment model (VADER / transformers) | Rejected. Extra dependency; the brief asks for prompts. |
| LLM classifies overview only, 4 labels + rationale | **Chosen.** |
| Fallback: `neutral` + honest rationale when no API key | **Chosen** for offline runs. |

**Locked**

- Labels: `positive` | `negative` | `mixed` | `neutral`. Keep `mixed` — conflict in a plot is not a “negative movie.”
- Use the **overview only**. Do not use box office or rating.
- Require `sentiment_rationale` (one sentence grounded in the overview).
- Horror/crime vocabulary is not negative by default.
- Without an API key: `overview_sentiment = neutral`, rationale states that sentiment is LLM-only. Do not fake analysis with keywords.

---

### 2. Budget and revenue tiers

Two attributes. Bands are deterministic so reviewers can audit them.

| Option | Verdict |
|--------|---------|
| Percentiles of the 80-movie sample | Rejected. A $50M film could be “high” only because the sample is small. |
| LLM free-form tiers with no bands | Rejected. Inconsistent and untestable. |
| LLM assigns tier using published bands | Rejected as source of truth. Easy to drift from the dollars. |
| Code assigns tier; LLM writes rationale + optional override | **Chosen.** |

**Locked bands**

- Budget: low &lt; $15M, medium $15–80M, high &gt; $80M
- Revenue: low &lt; $40M, medium $40–200M, high &gt; $200M

Code sets `budget_tier` and `revenue_tier`. The LLM writes `budget_rationale` and `revenue_rationale` (cite the dollar amount and the band).

Optional exception fields (not used for retrieval unless we opt in later):

- `budget_tier_override` / `revenue_tier_override` — only when era or industry context makes the band misleading (e.g. 1970s $11M as “high” for its time)
- `budget_override_rationale` / `revenue_override_rationale`

Downstream retrieval uses the **rule tier**.

---

### 3. Production effectiveness score

| Option | Verdict |
|--------|---------|
| LLM picks a free-form 0–100 | Rejected. Usual failure mode: unrepeatable scores. |
| Multiplicative `ROI * (rating/5)` only | Rejected. Dominated by outliers; no calibrated “50 = as expected.” |
| Formula in code; LLM writes the narrative | **Chosen.** |

**Locked formula** (0–100)

- `roi = revenue / budget`
- `roi_score`: log10 ROI scaled so 0.1x → 0, 1x → 50, 10x → 100, then clipped
- `rating_score = avg_rating / 5 * 100`
- `PES = round(0.55 * roi_score + 0.45 * rating_score)`, clipped to 0–100

**50 = as expected for its budget.** A cheap high-ROI, well-rated film can outscore an expensive modest success.

Labels (from the integer, in code):

| PES | Label |
|-----|--------|
| ≥ 80 | `highly_effective` |
| ≥ 60 | `effective` |
| ≥ 45 | `as_expected` |
| else | `underperformed` |

The LLM copies the provided score and label and writes `production_effectiveness_rationale` using ROI, rating, and score. It does not choose the integer.

---

## Output schema

Each enriched row is movie metadata plus:

| Field | Set by |
|-------|--------|
| `overview_sentiment`, `sentiment_rationale` | LLM / heuristic |
| `budget_tier`, `revenue_tier` | Code |
| `budget_rationale`, `revenue_rationale` | LLM / heuristic |
| `budget_tier_override`, `budget_override_rationale` | LLM (optional) |
| `revenue_tier_override`, `revenue_override_rationale` | LLM (optional) |
| `production_effectiveness_score`, `production_effectiveness_label` | Code |
| `production_effectiveness_rationale` | LLM / heuristic |
| `themes`, `intended_audience` | LLM / heuristic |
| `source` | `llm` or `heuristic` |
| `model` | OpenAI model id, or null |
| `prompt_version` | `data-prep-v1` |

Written to `data/enriched_movies.json` and `data/enriched.db`. Original databases are never mutated.

## Evaluation

`python -m movie_system eval` must include:

- sample size 50–100, unique ids, all five attributes present
- `production_effectiveness_score` in 0–100 **and** equal to the formula (rounding)
- `budget_tier` / `revenue_tier` agree with the dollar bands
- heuristic rows: sentiment is `neutral` (no keyword fake-out)
- spot-check: sentiment rationale should not cite revenue or budget as the reason for tone (enforced in the prompt; sampled manually)

## Non-goals

- Embeddings / vector index for 80 rows
- Mutating `db/movies.db` or `db/ratings.db`
- Using override tiers in retrieval (rule tiers only)
