import pytest

from langcheck.libreoffice.schema import (
    DocumentState,
    SchemaError,
    Suggestion,
    parse_document,
    parse_version,
)

SUGGESTION = {
    "id": "sg_1",
    "start": 5,
    "end": 8,
    "original": "teh",
    "kind": "Spelling",
    "operation": "R",
    "explanation": "Misspelt.",
    "options": [{"replacement": "the", "score": None, "delta": None}],
}


def test_parses_a_document_reply_ignoring_unknown_fields() -> None:
    state = parse_document(
        {"version": 2, "state": "done", "suggestions": [SUGGESTION], "unchecked": []}
    )

    assert state == DocumentState(
        version=2,
        state="done",
        suggestions=(Suggestion("sg_1", 5, 8, "teh", "Spelling", "Misspelt.", ("the",)),),
    )
    assert state.settled
    assert not DocumentState(1, "checking", ()).settled
    assert parse_version({"version": 7}) == 7


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"version": "2", "state": "done", "suggestions": []},
        {"version": True, "state": "done", "suggestions": []},
        {"version": 2, "state": "done", "suggestions": {}},
        {"version": 2, "state": "done", "suggestions": [{**SUGGESTION, "options": "the"}]},
        {"version": 2, "state": "done", "suggestions": [{**SUGGESTION, "start": 9}]},
        {"version": 2, "state": "done", "suggestions": [{**SUGGESTION, "id": 1}]},
        {"version": 2, "state": "done", "suggestions": [{**SUGGESTION, "options": [1]}]},
    ],
)
def test_rejects_malformed_replies(data: object) -> None:
    with pytest.raises(SchemaError):
        parse_document(data)
