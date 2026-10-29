from dataclasses import replace

from langcheck.check.cache_models import CacheSetting, SentenceResult, cache_key, move
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span

SETTING = CacheSetting("m", "check-v1", "ctx", ("Bilbo", "Shire"))


def key_for(**changes: object) -> str:
    fields: dict[str, object] = {
        "user": "ana",
        "sentence": "Read teh book.",
        "previous": "",
        "following": "",
    } | changes
    setting = fields.pop("setting", SETTING)
    assert isinstance(setting, CacheSetting)
    return cache_key(
        user=str(fields["user"]),
        sentence=str(fields["sentence"]),
        previous=str(fields["previous"]),
        following=str(fields["following"]),
        setting=setting,
    ).digest


def test_key_depends_on_text_neighbours_and_setting() -> None:
    base = key_for()

    assert key_for() == base
    assert key_for(sentence="Read the book.") != base
    assert key_for(previous="Hi.") != base
    assert key_for(following="Bye.") != base
    for setting in [
        replace(SETTING, model="other"),
        replace(SETTING, prompt_version="check-v2"),
        replace(SETTING, context_version="ctx2"),
        replace(SETTING, dictionary=("Bilbo",)),
    ]:
        assert key_for(setting=setting) != base


def test_key_ignores_dictionary_order_and_scopes_by_user() -> None:
    reordered = replace(SETTING, dictionary=("Shire", "Bilbo"))
    key = cache_key(user="ana", sentence="x", previous="", following="", setting=SETTING)
    other = cache_key(user="ben", sentence="x", previous="", following="", setting=reordered)

    assert key.digest == other.digest
    assert key != other


def suggestion(start: int) -> Suggestion:
    span = Span(start, start + 3)
    return Suggestion(
        id=suggestion_id(span, "teh", ErrorKind.SPELLING),
        span=span,
        original="teh",
        kind=ErrorKind.SPELLING,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=(Option("the"),),
    )


def test_results_are_stored_relative_and_placed_with_fresh_ids() -> None:
    stored = SentenceResult.relative_to(10, [suggestion(15)])

    assert stored.suggestions[0].span == Span(5, 8)
    assert stored.placed_at(40) == (suggestion(45),)
    assert move(suggestion(15), -15) == suggestion(0)
