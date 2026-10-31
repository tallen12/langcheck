import pytest
from pydantic import ValidationError

from langcheck.correction.schemas import CorrectionReply


def test_original_defaults_to_the_flagged_text() -> None:
    reply = CorrectionReply.model_validate_json(
        '{"flags": [{"flag": "F1", "corrections": '
        '[{"replacement": "goes", "kind": "VerbForm", "explanation": "Agreement."}]}]}'
    )

    assert reply.flags[0].corrections[0].original == ""


def test_a_flag_may_have_no_corrections() -> None:
    reply = CorrectionReply.model_validate_json('{"flags": [{"flag": "F1", "corrections": []}]}')

    assert reply.flags[0].corrections == []


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        CorrectionReply.model_validate_json('{"flags": [], "rewrite": "Everything."}')
