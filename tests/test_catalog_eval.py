from __future__ import annotations

from movie_system.evaluate import run_eval


def test_offline_eval_passes() -> None:
    report = run_eval(use_llm=False)
    failed = [c for c in report["checks"] if not c["passed"]]
    assert report["passed"] == report["total"], failed
