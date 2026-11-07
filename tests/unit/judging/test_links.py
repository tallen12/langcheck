from langcheck.judging.links import shape_links

TOKENS = ("We", "see", "failures", "often", ".")


def test_a_moved_word_links_its_removal_and_insertion() -> None:
    removal, insertion = (3, 4, ()), (1, 1, ("often",))

    assert shape_links(TOKENS, [removal, insertion]) == {
        removal: {insertion},
        insertion: {removal},
    }


def test_paired_quotes_and_brackets_are_linked() -> None:
    opening, closing = (0, 0, ('"',)), (5, 5, ('"',))
    bracket, close = (1, 1, ("(",)), (3, 3, (")",))

    links = shape_links(TOKENS, [opening, closing, bracket, close])

    assert links[opening] == {closing}
    assert links[bracket] == {close}


def test_edits_that_stand_alone_are_not_linked() -> None:
    assert shape_links(TOKENS, [(1, 2, ("saw",)), (3, 4, ()), (0, 0, ("So",))]) == {}
    assert shape_links(TOKENS, [(0, 0, ('"',))]) == {}
