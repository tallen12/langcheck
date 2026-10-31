import pytest
from pydantic import ValidationError

from langcheck.api.schemas import MAX_TEXT_LENGTH, CheckRequest, ContextBody, OptionBody
from langcheck.check.suggestion import Option, Scored
from langcheck.context.models import DocumentContext, Register


def test_context_body_round_trips_and_uses_the_register_key() -> None:
    context = DocumentContext("S", Register.FICTION_DIALOGUE, "en-GB", "n")

    body = ContextBody.of(context)

    assert body.to_context() == context
    assert body.model_dump(by_alias=True)["register"] == "fiction-dialogue-heavy"


def test_check_request_limits_text_and_rejects_unknown_fields() -> None:
    assert CheckRequest(text="Hi.").dictionary == []
    with pytest.raises(ValidationError):
        CheckRequest.model_validate({"text": "x" * (MAX_TEXT_LENGTH + 1)})
    with pytest.raises(ValidationError):
        CheckRequest.model_validate({"text": "x", "extra": 1})


def test_option_body_carries_the_rating_when_scored() -> None:
    assert OptionBody.of(Option("the")) == OptionBody(replacement="the")
    assert OptionBody.of(Option("the", Scored(score=-1.0, delta=2.0))) == OptionBody(
        replacement="the", score=-1.0, delta=2.0
    )
