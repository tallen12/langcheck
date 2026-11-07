from langcheck.judging.render import pieces, render, written_span

TOKENS = ("A", "build", "-", "up", ",", "he", "said", ".")
WRITTEN = "A build-up, he  said."


def test_tokens_are_found_in_the_published_sentence_with_the_spacing_after_them() -> None:
    found = pieces(("He", "said", "."), " He said.")

    assert found is not None
    lead, published = found
    assert lead == " "
    assert [(p.text, p.gap) for p in published] == [("He", " "), ("said", ""), (".", "")]
    assert pieces(("He", "went"), "He said") is None
    assert pieces(("He",), "He said") is None


def test_without_edits_the_published_sentence_is_kept_as_written() -> None:
    assert render(TOKENS, [], WRITTEN) == WRITTEN


def test_a_replacement_keeps_the_writers_spacing_around_it() -> None:
    assert render(TOKENS, [(6, 7, ("says",))], WRITTEN) == "A build-up, he  says."
    assert render(TOKENS, [(1, 4, ("buildup",))], WRITTEN) == "A buildup, he  said."


def test_insertions_and_deletions_space_only_the_boundaries_they_touch() -> None:
    assert render(TOKENS, [(5, 5, ("then",))], WRITTEN) == "A build-up, then he  said."
    assert render(TOKENS, [(4, 4, (";",))], WRITTEN) == "A build-up;, he  said."
    assert render(TOKENS, [(4, 5, ())], WRITTEN) == "A build-up he  said."
    assert render(TOKENS, [(0, 1, ())], WRITTEN) == "build-up, he  said."
    assert render(("Hi", "there"), [(1, 1, (",",))], "Hi there") == "Hi, there"
    assert render(("Hi",), [(1, 1, ("!",))], "Hi") == "Hi!"


def test_without_the_published_sentence_tokens_get_normal_spacing() -> None:
    assert render(TOKENS, [(6, 7, ("says",))]) == "A build - up, he says."
    assert render(TOKENS, [], "Something else entirely.") == "A build - up, he said."


def test_a_span_reads_as_published() -> None:
    assert written_span(TOKENS, 1, 4, WRITTEN) == "build-up"
    assert written_span(TOKENS, 5, 7, WRITTEN) == "he  said"
    assert written_span(TOKENS, 1, 4) == "build - up"
    assert written_span(TOKENS, 2, 2, WRITTEN) == ""
