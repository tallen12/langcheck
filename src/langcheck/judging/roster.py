"""Which models sit on the panel: different families, none the corrector's own.

A judge from the corrector's family tends to favour its corrections, so it gives way to
a standby from another family.
"""

import re
from collections.abc import Sequence

DEFAULT_PANEL = ("qwen3:14b", "gemma4:12b-it-q4_K_M", "llama3.1:latest")
DEFAULT_STANDBY = "qwen3.5:9b-q8_0"

_FAMILY = re.compile(r"[a-z]+")


class RosterError(ValueError):
    """No panel without the corrector's family can be formed."""


def model_family(model: str) -> str:
    """The family of a model name: its leading letters, after any namespace.

    ``gemma4:12b`` and ``gemma3`` are both ``gemma``; ``user/qwen3.5:9b`` is ``qwen``.
    """
    name = model.rsplit("/", 1)[-1].lower()
    match = _FAMILY.match(name)
    return match.group(0) if match is not None else name


def choose_panel(
    models: Sequence[str], *, corrector: str | None, standby: str = DEFAULT_STANDBY
) -> tuple[str, ...]:
    """``models`` with any member of ``corrector``'s family replaced by ``standby``."""
    if len(set(models)) != len(models) or not models:
        msg = f"the panel needs distinct models, got {list(models)}"
        raise RosterError(msg)
    if corrector is None:
        return tuple(models)
    family = model_family(corrector)
    kept = tuple(m for m in models if model_family(m) != family)
    if len(kept) == len(models):
        return kept
    if len(models) - len(kept) > 1 or model_family(standby) == family or standby in kept:
        msg = f"cannot keep {family} models off the panel {list(models)} with {standby}"
        raise RosterError(msg)
    return tuple(standby if model_family(m) == family else m for m in models)
