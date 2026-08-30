from __future__ import annotations

import argparse
import json
import sys

from movie_system.config import ENRICHED_JSON, SAMPLE_SIZE
from movie_system.enrich import enrich_movies, load_enriched, save_enriched
from movie_system.evaluate import run_eval
from movie_system.llm import llm_available
from movie_system.observe import last_trace, read_traces
from movie_system.sample import sample_movies
from movie_system.system import run_query


def cmd_prepare(args: argparse.Namespace) -> int:
    use_llm = bool(args.llm) or (args.prefer_llm and llm_available())
    if args.llm and not llm_available():
        print("OPENAI_API_KEY is not set; cannot use --llm.", file=sys.stderr)
        return 1
    movies = sample_movies(size=args.size)
    print(f"Sampled {len(movies)} movies (stratified by primary genre).")
    enriched = enrich_movies(movies, use_llm=use_llm)
    save_enriched(enriched)
    source = enriched[0].source if enriched else "n/a"
    print(f"Wrote {ENRICHED_JSON} via {source} enrichment.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    catalog = load_enriched()
    use_llm = not args.no_llm and llm_available()
    intent, response, retrieved = run_query(args.query, catalog, use_llm=use_llm)
    trace = last_trace()
    out = {
        "query": args.query,
        "used_llm": use_llm,
        "intent": intent.model_dump(),
        "retrieved_ids": [m.movie_id for m in retrieved],
        "response": response.model_dump(),
        "trace": trace.to_dict() if trace else None,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    events = read_traces(limit=args.last)
    if not events:
        print("No traces yet. Ask a question in the chat or via `ask`.")
        return 0
    print(json.dumps(events, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    use_llm = args.llm and llm_available()
    report = run_eval(use_llm=use_llm)
    print(json.dumps(report, indent=2))
    failed = report["total"] - report["passed"]
    print(f"\n{report['passed']}/{report['total']} checks passed.")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-enriched movie catalog and query system"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Sample 50-100 movies and enrich 5 attributes")
    p_prep.add_argument("--size", type=int, default=SAMPLE_SIZE)
    p_prep.add_argument("--llm", action="store_true", help="Require OpenAI enrichment")
    p_prep.add_argument(
        "--prefer-llm",
        action="store_true",
        help="Use OpenAI if OPENAI_API_KEY is set, else heuristics",
    )
    p_prep.set_defaults(func=cmd_prepare)

    p_ask = sub.add_parser("ask", help="Run a natural-language catalog question")
    p_ask.add_argument("query")
    p_ask.add_argument("--no-llm", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run schema, intent, and grounding checks")
    p_eval.add_argument("--llm", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_tr = sub.add_parser("traces", help="Print recent request traces (JSONL)")
    p_tr.add_argument("--last", type=int, default=20)
    p_tr.set_defaults(func=cmd_traces)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
