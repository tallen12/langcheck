"""What a noise model is asked for, and what it gave back.

A request names the kinds of its errors, one per error, by ERRANT's part of speech or
operation (``PUNCT``, ``DET``, ``SPELL``...). The model reads it as plain words before
the clean sentence, as Stahlberg and Kumar's tagged corruption models do, and learns
from corrected sentences written back with their real errors.
"""

import difflib
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.synthesis.corruption import known_clean
from langcheck.tagging.alignment import split_words

KINDS: Final = (
    "PUNCT",
    "DET",
    "PREP",
    "VERB",
    "NOUN",
    "ORTH",
    "SPELL",
    "MORPH",
    "PRON",
    "CONTR",
    "CONJ",
    "WO",
    "PART",
    "ADJ",
    "ADV",
    "OTHER",
)
OTHER: Final = "OTHER"
SEPARATOR: Final = " | "
NOISE: Final = "NOISE"
"""The code of an edit the model made: its kind is what was asked, not what was checked."""
OPERATIONS: Final = {"insert": "M", "delete": "U", "replace": "R"}
"""Going from the noisy words to the clean ones, words to insert were dropped (missing)
and words to delete were added (unnecessary)."""


@dataclass(frozen=True, slots=True)
class NoiseRequest:
    """One error of each of ``kinds``, in a fixed order so equal requests read alike."""

    kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KindMix:
    """How many errors a sentence gets and how often each kind is asked for."""

    counts: tuple[int, ...]
    kinds: tuple[str, ...]
    weights: tuple[float, ...]

    def draw(self, rng: random.Random) -> NoiseRequest:
        """A request with a count from ``counts`` and kinds as ``weights`` say."""
        count = rng.choice(self.counts)
        return NoiseRequest(tuple(sorted(rng.choices(self.kinds, self.weights, k=count))))


def kind_of(code: str) -> str:
    """The kind of an ERRANT code: ``R:VERB:SVA`` is a ``VERB`` error."""
    parts = code.split(":")
    kind = parts[1] if len(parts) > 1 else code
    return kind if kind in KINDS else OTHER


def requested(sentence: AnnotatedSentence) -> NoiseRequest:
    """The request ``sentence``'s first annotator's edits would answer."""
    edits = sentence.annotators[0] if sentence.annotators else ()
    return NoiseRequest(tuple(sorted(kind_of(edit.code) for edit in edits)))


def prompt(clean: str, request: NoiseRequest) -> str:
    """What the model reads: the count, the kinds, then the clean sentence."""
    return " ".join([str(len(request.kinds)), *request.kinds]) + SEPARATOR + clean


def training_example(sentence: AnnotatedSentence) -> tuple[str, str] | None:
    """The prompt for ``sentence`` corrected and, as the answer, the sentence as written;
    None when it cannot be corrected.
    """
    clean = known_clean(sentence)
    if clean is None:
        return None
    return prompt(detokenize(clean), requested(sentence)), detokenize(sentence.tokens)


def kind_mix(sentences: Sequence[AnnotatedSentence], *, counts: Sequence[int]) -> KindMix:
    """Kinds as often as ``sentences``' edits show them, and error counts from ``counts``."""
    found = Counter(kind_of(e.code) for s in sentences for e in s.annotators[0])
    if not found:
        msg = "no edits to learn which kinds of error to ask for"
        raise ValueError(msg)
    kinds = tuple(found)
    return KindMix(tuple(counts), kinds, tuple(float(found[k]) for k in kinds))


def noisy_sentence(clean: Sequence[str], written: str) -> AnnotatedSentence:
    """``written`` as words, with the edits that turn it back into ``clean``."""
    noisy = split_words(written)
    matcher = difflib.SequenceMatcher(None, noisy, list(clean), autojunk=False)
    edits = tuple(
        GoldEdit(i1, i2, tuple(clean[j1:j2]), f"{OPERATIONS[tag]}:{NOISE}")
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    return AnnotatedSentence(noisy, (edits,))
