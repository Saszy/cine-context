from __future__ import annotations

from movie_system.sample_prompts import SAMPLE_PROMPTS, prompts_except


def test_at_least_ten_sample_prompts() -> None:
    assert len(SAMPLE_PROMPTS) >= 10
    labels = [label for label, _ in SAMPLE_PROMPTS]
    queries = [q for _, q in SAMPLE_PROMPTS]
    assert len(set(labels)) == len(labels)
    assert len(set(queries)) == len(queries)


def test_followups_skip_current() -> None:
    current = SAMPLE_PROMPTS[0][1]
    follow = prompts_except(current)
    assert len(follow) == len(SAMPLE_PROMPTS) - 1
    assert current not in {q for _, q in follow}


def test_followup_chips_match_labels() -> None:
    from movie_system.sample_prompts import followup_chips

    current = SAMPLE_PROMPTS[0][1]
    chips = followup_chips(current)
    assert chips[0] == {
        "label": SAMPLE_PROMPTS[1][0],
        "query": SAMPLE_PROMPTS[1][1],
    }
    assert current not in {chip["query"] for chip in chips}
