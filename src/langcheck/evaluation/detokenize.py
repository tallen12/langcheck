"""Turning a set's tokenized sentence back into text as a writer would type it.

The sets store sentences as space-separated tokens ("The law 's spirit ."). Checked
that way, the LLM misses errors and suggests spacing fixes, so the checker gets the
sentence with normal spacing; ERRANT re-tokenizes its correction for scoring.
"""

from collections.abc import Sequence

_ATTACH_LEFT = frozenset({".", ",", ";", ":", "!", "?", "%", ")", "]", "}", "..."})
"""Tokens written without a space before them."""
_ATTACH_RIGHT = frozenset({"(", "[", "{", "$"})
"""Tokens written without a space after them."""
_CLITICS = frozenset({"'s", "'re", "'ve", "'d", "'ll", "'m", "n't"})
"""Contractions and possessives, matched case-insensitively."""
_QUOTES = frozenset({'"', "'"})


def detokenize(tokens: Sequence[str]) -> str:
    """``tokens`` with the spaces a writer would not type removed.

    Punctuation and clitics ("'s", "n't") attach to the word before them, opening
    brackets to the word after. Straight quotes alternate between opening (attached to
    the word after) and closing (attached to the word before).
    """
    parts: list[str] = []
    glue_next = False
    open_quotes: set[str] = set()
    for token in tokens:
        glue = glue_next or not parts
        glue_next = False
        if token in _QUOTES:
            if token in open_quotes:
                open_quotes.discard(token)
                glue = True
            else:
                open_quotes.add(token)
                glue_next = True
        elif token in _ATTACH_LEFT or token.lower() in _CLITICS:
            glue = True
        elif token in _ATTACH_RIGHT:
            glue_next = True
        parts.append(token if glue else f" {token}")
    return "".join(parts)
