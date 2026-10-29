import pytest

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, M2Error, corrected, read_m2

M2 = """S He go to school every days .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0
A 5 6|||R:NOUN:NUM|||day|||REQUIRED|||-NONE-|||0
A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||1

S It is fine .

S I like the the cats .
A 3 4|||U:DET|||-NONE-|||REQUIRED|||-NONE-|||0
"""


def test_reads_sentences_edits_and_annotators() -> None:
    first, second, third = read_m2(M2)

    assert first == AnnotatedSentence(
        tokens=("He", "go", "to", "school", "every", "days", "."),
        annotators=(
            (GoldEdit(1, 2, ("goes",), "R:VERB:SVA"), GoldEdit(5, 6, ("day",), "R:NOUN:NUM")),
            (),
        ),
    )
    assert second.annotators == ((),)
    assert third.annotators == ((GoldEdit(3, 4, (), "U:DET"),),)


@pytest.mark.parametrize(
    "text",
    [
        "A 1 2|||R:X|||y|||REQUIRED|||-NONE-|||0",
        "S a b\nB nope",
        "S a b\nA 1 2|||R:X|||y",
        "S a b\nA x y|||R:X|||y|||REQUIRED|||-NONE-|||0",
        "S a b\nA 1 9|||R:X|||y|||REQUIRED|||-NONE-|||0",
    ],
)
def test_rejects_malformed_m2(text: str) -> None:
    with pytest.raises(M2Error):
        read_m2(text)


def test_corrected_applies_edits_in_order() -> None:
    [sentence, *_] = read_m2(M2)

    assert corrected(sentence.tokens, sentence.annotators[0]) == (
        "He",
        "goes",
        "to",
        "school",
        "every",
        "day",
        ".",
    )
    assert corrected(("a", "b"), [GoldEdit(1, 1, ("x",), "M:DET")]) == ("a", "x", "b")
    with pytest.raises(M2Error, match="overlap"):
        corrected(("a", "b", "c"), [GoldEdit(0, 2, (), "U:X"), GoldEdit(1, 2, (), "U:X")])
