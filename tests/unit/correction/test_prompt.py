from langcheck.correction.models import FlaggedSentence, FlagType
from langcheck.correction.prompt import (
    correction_messages,
    correction_request,
    flag_ids,
    system_prompt,
)
from langcheck.correction.schemas import CorrectionRequest
from langcheck.llm.models import Role
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier

# "He go store." starting at offset 10 of its text.
FLAGGED = FlaggedSentence(
    SentenceInText(CharSpan(10, 22), "He go store.", "Hi there."),
    (
        TextFlag(CharSpan(13, 15), 0.91234, Tier.ERROR, FlagKind.GENERAL),
        TextFlag(CharSpan(15, 15), 0.4, Tier.SUGGESTION, FlagKind.GENERAL),
    ),
)


def test_flags_are_numbered_in_reading_order() -> None:
    assert list(flag_ids(FLAGGED)) == ["F1", "F2"]


def test_the_request_has_offsets_into_the_sentence_and_each_flags_type() -> None:
    request = correction_request(FLAGGED, count=2)

    assert request.context == "Hi there."
    assert request.corrections_per_flag == 2
    first, second = request.flags
    assert (first.start, first.end, first.text, first.type) == (3, 5, "go", FlagType.WRONG_TEXT)
    assert first.confidence is Tier.ERROR
    assert first.score == 0.912
    assert (second.start, second.end, second.text) == (5, 5, "")
    assert second.type is FlagType.MISSING_TEXT


def test_the_request_marks_flags_in_the_sentence() -> None:
    assert correction_request(FLAGGED, count=1).marked == "He <F1>go</F1><F2/> store."


def test_the_user_message_is_the_request_as_json() -> None:
    system, user = correction_messages(FLAGGED, count=3)

    assert system.role is Role.SYSTEM
    assert user.role is Role.USER
    assert CorrectionRequest.model_validate_json(user.content).sentence == "He go store."


def test_the_system_prompt_lists_kinds_and_the_reply_schema() -> None:
    prompt = system_prompt()

    assert "- SubjectVerbAgreement" in prompt
    assert '"corrections"' in prompt
