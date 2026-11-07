"""Showing a sentence, and a correction of it, as a reader would see them.

Sets store sentences as tokens, and tokenizers split words such as "build-up" in ways
that cannot be undone from the tokens alone, so a judge shown "build - up" reports an
error the writer never made. When the sentence as published is known, each token is found
in it and edits replace the published characters, keeping the writer's spacing elsewhere,
even where it is wrong: only the spacing next to an edit is chosen anew.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from langcheck.evaluation.detokenize import attaches_left, detokenize

HYPHEN = "-"

type Edit = tuple[int, int, tuple[str, ...]]
"""Source tokens ``[start, end)`` become the tokens given."""


@dataclass(frozen=True, slots=True)
class Piece:
    """Published text of a token, or an edit's text, and the spacing after it."""

    text: str
    gap: str
    first: str
    """The first token, which decides whether a space comes before the piece."""
    respace: bool = False
    """Whether the spacing after the piece borders an edit and is chosen anew."""


def pieces(tokens: Sequence[str], written: str) -> tuple[str, tuple[Piece, ...]] | None:
    """Text before the first token and each token with the spacing after it, or ``None``
    when the tokens are not found in ``written`` in order with only spaces between them.
    """
    spans: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        start = written.find(token, position)
        if start < 0 or written[position:start].strip():
            return None
        spans.append((start, start + len(token)))
        position = start + len(token)
    if written[position:].strip():
        return None
    ends = [start for start, _ in spans[1:]] + [len(written)]
    found = tuple(
        Piece(written[start:end], written[end:following], tokens[i])
        for i, ((start, end), following) in enumerate(zip(spans, ends, strict=True))
    )
    return written[: spans[0][0]] if spans else written, found


def render(tokens: Sequence[str], edits: Sequence[Edit], written: str | None = None) -> str:
    """``tokens`` with non-overlapping ``edits`` applied, in the published text's spacing
    when ``written`` is given and its tokens can be found, else with normal spacing.
    """
    found = pieces(tokens, written) if written is not None else None
    if found is None:
        result: list[str] = []
        position = 0
        for start, end, correction in sorted(edits):
            result += [*tokens[position:start], *correction]
            position = end
        return detokenize([*result, *tokens[position:]])
    lead, published = found
    out: list[Piece] = []
    position = 0
    for start, end, correction in sorted(edits):
        out += published[position:start]
        if out:
            # Deleting a comma keeps the space after it until the boundary is chosen anew.
            gap = out[-1].gap or (published[end - 1].gap if end > start else "")
            out[-1] = replace(out[-1], gap=gap, respace=True)
        if end == start and correction == (HYPHEN,):
            # Tokenized sets write a hyphen as its own token, so an inserted one joins the
            # words on both sides: "larger than life" becomes "larger-than-life".
            if out:
                out[-1] = replace(out[-1], gap="", respace=False)
            out.append(Piece(HYPHEN, "", HYPHEN))
        elif correction:
            gap = published[end - 1].gap if end > start else " "
            out.append(Piece(detokenize(correction), gap, correction[0], respace=True))
        position = end
    out += published[position:]
    for i in range(len(out) - 1):
        before, after = out[i], out[i + 1]
        if not before.respace:
            continue
        if attaches_left(after.first):
            out[i] = replace(before, gap="")
        elif not before.gap:
            out[i] = replace(before, gap=" ")
    return (lead + "".join(piece.text + piece.gap for piece in out)).strip()


def written_span(tokens: Sequence[str], start: int, end: int, written: str | None = None) -> str:
    """Source tokens ``[start, end)`` as published, or with normal spacing."""
    found = pieces(tokens, written) if written is not None else None
    if found is None:
        return detokenize(tokens[start:end])
    span = found[1][start:end]
    return "".join(piece.text + piece.gap for piece in span[:-1]) + (span[-1].text if span else "")
