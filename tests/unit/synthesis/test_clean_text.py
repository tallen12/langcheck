import json
from pathlib import Path

from langcheck.synthesis.clean_text import clean_sentences, looks_finished, read_texts


def test_an_ordinary_sentence_looks_finished() -> None:
    text = "The cat sat on the mat."
    assert looks_finished(text.split(), text)


def test_unfinished_short_or_marked_up_text_does_not() -> None:
    for text in (
        "the cat sat on the mat.",
        "The cat sat on the mat",
        "Yes it is.",
        "The ((cat)) sat on it.",
        "The cat ( ) sat on it.",
        "The cat (born 1951 sat on it.",
        "The cat born 1951) sat on it.",
    ):
        assert not looks_finished(text.split(), text)


def test_a_plain_parenthetical_looks_finished() -> None:
    text = "Reagan Louie (born 1951) is an American photographer (and artist)."
    assert looks_finished(text.split(), text)


def test_sentences_keep_what_came_before_them() -> None:
    found = list(
        clean_sentences(["The cat sat on the mat. The dog ran into the yard."], context_sentences=1)
    )
    assert [s.context for s in found] == ["", "The cat sat on the mat."]
    assert found[1].words[:2] == ("The", "dog")


def test_read_texts_takes_only_text_fields(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    lines = [{"text": "Hello there."}, {"text": " "}, {"id": 1}, [1]]
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    assert read_texts(path) == ["Hello there."]
