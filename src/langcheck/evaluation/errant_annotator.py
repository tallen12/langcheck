"""``EditAnnotator`` over ERRANT with spaCy's ``en_core_web_sm``."""

from collections.abc import Collection, Sequence

import errant
import spacy
from errant.annotator import Annotator
from errant.en import classifier
from spacy.tokens import Doc, Span

from langcheck.evaluation.m2 import GoldEdit, corrected
from langcheck.evaluation.models import ScoredEdit
from langcheck.taxonomy.errant_map import (
    Dropped,
    EditEvidence,
    Mapped,
    UnknownErrantCodeError,
    map_edit,
)
from langcheck.taxonomy.kinds import ErrorKind

DEFAULT_MODEL = "en_core_web_sm"


class ErrantEditAnnotator:
    """Types edits with ERRANT and maps their codes onto our kinds."""

    def __init__(self, annotator: Annotator, word_list: Collection[str]) -> None:
        self._annotator = annotator
        self._word_list = word_list

    def system_edits(self, source: Sequence[str], corrected: str) -> list[ScoredEdit]:
        """ERRANT tokenizes ``corrected`` and aligns it with ``source``."""
        original = self._annotator.parse(" ".join(source), tokenise=False)
        correction = self._annotator.parse(corrected, tokenise=True)
        return self._annotate(original, correction)

    def reference_edits(self, source: Sequence[str], reference: Sequence[str]) -> list[ScoredEdit]:
        """Both sides are tokenized already, as in JFLEG."""
        original = self._annotator.parse(" ".join(source), tokenise=False)
        correction = self._annotator.parse(" ".join(reference), tokenise=False)
        return self._annotate(original, correction)

    def gold_edits(self, source: Sequence[str], edits: Sequence[GoldEdit]) -> list[ScoredEdit]:
        """Kinds for M2 edits, with the tags and dependencies both sides need."""
        original = self._annotator.parse(" ".join(source), tokenise=False)
        target = corrected(source, edits)
        correction = self._annotator.parse(" ".join(target), tokenise=False) if target else None
        scored: list[ScoredEdit] = []
        shift = 0
        for edit in sorted(edits, key=lambda e: (e.start, e.end)):
            c_start = edit.start + shift
            c_end = c_start + len(edit.correction)
            shift += len(edit.correction) - (edit.end - edit.start)
            clean = correction[c_start:c_end] if correction is not None else None
            kind = self._kind(edit.code, original[edit.start : edit.end], clean)
            scored.append(ScoredEdit(edit.start, edit.end, edit.correction, kind))
        return scored

    def _annotate(self, original: Doc, correction: Doc) -> list[ScoredEdit]:
        edits: list[ScoredEdit] = []
        for edit in self._annotator.annotate(original, correction):
            noisy, clean = _span(edit.o_toks), _span(edit.c_toks)
            edits.append(
                ScoredEdit(
                    start=int(edit.o_start),
                    end=int(edit.o_end),
                    correction=tuple(str(edit.c_str).split()),
                    kind=self._kind(str(edit.type), noisy, clean),
                )
            )
        return edits

    def _kind(self, code: str, noisy: Span, clean: Span | None) -> ErrorKind | None:
        clean_tokens = list(clean) if clean is not None else []
        words = tuple(token.text for token in noisy)
        evidence = EditEvidence(
            noisy=words,
            clean=tuple(token.text for token in clean_tokens),
            noisy_tags=tuple(token.tag_ for token in noisy),
            clean_tags=tuple(token.tag_ for token in clean_tokens),
            noisy_deps=tuple(token.dep_ for token in noisy),
            clean_deps=tuple(token.dep_ for token in clean_tokens),
            noisy_is_word=len(words) == 1
            and (words[0] in self._word_list or words[0].lower() in self._word_list),
        )
        try:
            mapping = map_edit(code, evidence)
        except UnknownErrantCodeError:
            return None
        match mapping:
            case Mapped(kind=kind):
                return kind
            case Dropped():
                return None


def load_errant_annotator(model: str = DEFAULT_MODEL) -> ErrantEditAnnotator:
    """Load ``model`` for ERRANT; one per process, since ERRANT keeps it in a global."""
    nlp = spacy.load(model, exclude=["ner"])
    return ErrantEditAnnotator(errant.load("en", nlp), classifier.spell)


def _span(tokens: object) -> Span:
    # ERRANT is untyped; its edit sides are slices of the parsed docs.
    if not isinstance(tokens, Span):
        msg = f"expected an ERRANT edit side as a spaCy Span, got {type(tokens)}"
        raise TypeError(msg)
    return tokens
