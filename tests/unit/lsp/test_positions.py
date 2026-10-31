import pytest

from langcheck.lsp.positions import LineIndex, Position, Range


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (0, Position(0, 0)),
        (3, Position(0, 3)),
        (4, Position(1, 0)),
        (6, Position(1, 2)),
        (7, Position(2, 0)),
        (9, Position(3, 0)),
        (100, Position(3, 2)),
        (-5, Position(0, 0)),
    ],
)
def test_lines_end_at_lf_crlf_and_cr(offset: int, expected: Position) -> None:
    assert LineIndex("abc\nd\r\ne\rfg").position(offset) == expected


def test_astral_characters_take_two_utf16_columns() -> None:
    text = "a😀b café"
    index = LineIndex(text)

    assert index.position(text.index("b")) == Position(0, 3)
    assert index.position(text.index("é")) == Position(0, 8)
    assert index.range(1, 2) == Range(Position(0, 1), Position(0, 3))


def test_ranges_overlap_when_they_share_a_position() -> None:
    word = Range(Position(1, 4), Position(1, 8))

    assert word.overlaps(Range(Position(1, 6), Position(1, 6)))
    assert word.overlaps(Range(Position(0, 0), Position(1, 4)))
    assert not word.overlaps(Range(Position(1, 9), Position(2, 0)))


@pytest.mark.parametrize("offset", [0, 1, 2, 5, 6, 7])
def test_offset_is_the_inverse_of_position(offset: int) -> None:
    index = LineIndex("a😀b\ncd\r\n")

    assert index.offset(index.position(offset)) == offset


def test_offset_past_the_text_is_clamped() -> None:
    index = LineIndex("ab\ncd")

    assert index.offset(Position(9, 0)) == 5
    assert index.offset(Position(0, 9)) == 5
