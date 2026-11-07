from pydantic import TypeAdapter

from langcheck.judging.models import (
    Candidate,
    EditJudgment,
    Item,
    Judgment,
    SentenceJudgment,
    Source,
    Verdict,
    sentence_key,
)


def test_a_sentence_key_depends_only_on_the_tokens() -> None:
    first = Item(("He", "go", "."), "", ())
    second = Item(("He", "go", "."), "Earlier.", (Candidate(1, 2, ("went",), frozenset()),))

    assert first.key == second.key == sentence_key(["He", "go", "."])
    assert first.key != sentence_key(["He", "goes", "."])
    assert len(first.key) == 16


def test_a_candidate_key_is_its_span_and_correction() -> None:
    candidate = Candidate(1, 2, ("went",), frozenset({Source.GOLD}))

    assert candidate.key == (1, 2, ("went",))


def test_judgments_round_trip_by_their_type() -> None:
    adapter: TypeAdapter[Judgment] = TypeAdapter(Judgment)
    edit = EditJudgment(
        sentence="k",
        start=1,
        end=2,
        correction=("went",),
        judge="j",
        verdict=Verdict.VALID,
        reason="tense",
    )
    sentence = SentenceJudgment(sentence="k", judge="j", remaining_error=True, reason="comma")

    assert adapter.validate_json(adapter.dump_json(edit)) == edit
    assert adapter.validate_json(adapter.dump_json(sentence)) == sentence
    assert edit.edit == (1, 2, ("went",))
