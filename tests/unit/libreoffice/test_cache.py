from langcheck.libreoffice.cache import Checked, DocumentCache, ParagraphError
from langcheck.libreoffice.schema import Suggestion

URI = "file:///a.odt"
TEXT = "Intro here.\nRead teh book.\nRead teh book."


def typo(start: int) -> Suggestion:
    return Suggestion(f"sg_{start}", start, start + 3, "teh", "Spelling", "Misspelt.", ("the",))


def test_errors_are_relative_to_the_first_matching_paragraph() -> None:
    cache = DocumentCache()
    cache.store(URI, Checked(TEXT, (typo(17), typo(32))))

    assert cache.errors_for(URI, "Read teh book.") == [
        ParagraphError(5, 3, "Spelling", "Misspelt.", ("the",), "sg_17")
    ]
    assert cache.errors_for(URI, "Intro here.") == []


def test_unknown_documents_and_edited_paragraphs_get_nothing() -> None:
    cache = DocumentCache()
    cache.store(URI, Checked(TEXT, (typo(17),)))

    assert cache.errors_for("file:///other.odt", "Read teh book.") == []
    assert cache.errors_for(URI, "Read teh book!") == []
    assert cache.errors_for(URI, "") == []
    cache.forget(URI)
    assert cache.errors_for(URI, "Read teh book.") == []


def test_store_reports_whether_the_suggestions_changed() -> None:
    cache = DocumentCache()

    assert cache.store(URI, Checked(TEXT, (typo(17),)))
    assert not cache.store(URI, Checked(TEXT + " More.", (typo(17),)))
    assert cache.store(URI, Checked(TEXT, ()))
