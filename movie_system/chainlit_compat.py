from __future__ import annotations

import sys
from contextvars import ContextVar
from types import ModuleType
from typing import Any, Optional


def _context_module() -> ModuleType:
    # chainlit/__init__.py binds `context` onto the package, so
    # `import chainlit.context` returns a LazyProxy, not the module.
    import chainlit.context as _  # noqa: F401

    return sys.modules["chainlit.context"]


def patch_local_steps() -> ContextVar[Optional[Any]]:
    """Give Chainlit's local_steps ContextVar a default.

    Chainlit 2.3 defines ``ContextVar("local_steps")`` with no default and
    only sets it on the main thread. On Python 3.9, ``on_chat_start`` and
    action callbacks run in a context where ``.get()`` raises LookupError.
    """
    cl_context = _context_module()
    import chainlit.message as cl_message
    import chainlit.step as cl_step

    try:
        current = cl_context.local_steps.get()
    except LookupError:
        current = None

    fixed: ContextVar[Optional[Any]] = ContextVar("local_steps", default=None)
    if current is not None:
        fixed.set(current)

    cl_context.local_steps = fixed
    cl_step.local_steps = fixed
    cl_message.local_steps = fixed
    try:
        import chainlit.openai as cl_openai

        cl_openai.local_steps = fixed
    except Exception:
        pass
    return fixed
