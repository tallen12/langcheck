"""What the tagger and proposer train on: clean prose corrupted with error tables, and
pairs passed in, with a held-out share of each to validate on.

No annotated evaluation set is read here (see :mod:`langcheck.evaluation.datasets`). The
held-out sentences come from the same mix, so they show how well training fits its own
noise; pass realistic pairs to validate on as well.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from langcheck.evaluation.m2 import AnnotatedSentence
from langcheck.synthesis.clean_text import clean_sentences
from langcheck.synthesis.corruption import (
    Confusions,
    corrupt_in_context,
    known_clean,
    table_sets,
)
from langcheck.synthesis.default_tables import default_confusions

type Pair = tuple[AnnotatedSentence, str]
"""A sentence with its edits and the text before it."""

HELD_SEED = 1
"""Held-out prose is corrupted with its own random draws."""


class TrainingMixError(Exception):
    """The mix names nothing to train on, or too little to hold any out."""


@dataclass(frozen=True, slots=True)
class TrainingMix:
    """``clean`` prose to corrupt, ready ``pairs`` with their context, more ``tables`` to
    learn errors from beside the built-in ones, and the ``errors`` counts a corrupted
    sentence draws from. With ``corrected``, the clean version of each training sentence
    with an error is added too, with nothing to fix.
    """

    corrected: bool = False
    clean: tuple[str, ...] = ()
    pairs: tuple[Pair, ...] = ()
    tables: tuple[AnnotatedSentence, ...] = ()
    errors: tuple[int, ...] = ()


def corruption_tables(mix: TrainingMix) -> tuple[Confusions, ...]:
    """The built-in tables and, with ``mix.tables``, those learned from them; each
    corrupted sentence takes one at random. ``mix.errors`` replaces each one's counts.
    """
    tables = (default_confusions(), *table_sets([mix.tables]))
    if mix.errors:
        return tuple(replace(t, error_counts=mix.errors) for t in tables)
    return tables


def _with_corrected(pairs: Sequence[Pair], *, corrected: bool) -> list[Pair]:
    found = list(pairs)
    if corrected:
        found += [
            (AnnotatedSentence(fixed, ((),)), context)
            for sentence, context in pairs
            if sentence.annotators
            and sentence.annotators[0]
            and (fixed := known_clean(sentence)) is not None
        ]
    return found


def assemble(mix: TrainingMix, *, context_sentences: int) -> tuple[list[Pair], list[Pair]]:
    """The training sentences ``mix`` names, and the held-out ones. Prose and pairs are
    split before corrupting, so no sentence is on both sides.
    """
    # Imported here: the training module needs PyTorch.
    from langcheck.tagging.training import split_holdout

    if not mix.clean and not mix.pairs:
        msg = "nothing to train on: pass --clean prose or --pairs"
        raise TrainingMixError(msg)
    prose = clean_sentences(mix.clean, context_sentences=context_sentences)
    found = [(c.words, c.context) for c in prose]
    clean_train, clean_held = split_holdout(found)
    pairs_train, pairs_held = split_holdout(mix.pairs)
    tables = corruption_tables(mix)
    training = _with_corrected(
        [*corrupt_in_context(clean_train, tables), *pairs_train], corrected=mix.corrected
    )
    held = [*corrupt_in_context(clean_held, tables, seed=HELD_SEED), *pairs_held]
    if not held:
        msg = f"{len(training)} training sentences leave none for validation; use more"
        raise TrainingMixError(msg)
    return training, held
