from __future__ import annotations

from contextvars import Context

from movie_system.chainlit_compat import patch_local_steps


def test_local_steps_get_does_not_raise_in_empty_context() -> None:
    var = patch_local_steps()
    empty = Context()
    assert empty.run(var.get) is None
