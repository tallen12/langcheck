import pytest
from pydantic import ValidationError

from langcheck.check.reply import CheckReply, ReplyIssue


def test_parses_the_documented_reply() -> None:
    reply = CheckReply.model_validate_json(
        """
        {"issues": [{
          "sentence": "S3", "original": "their going", "occurrence": 1,
          "kind": "Apostrophe", "explanation": "Needs the contraction.",
          "options": ["they're going", "they are going"]
        }]}
        """
    )

    assert reply.issues[0].sentence == "S3"
    assert reply.issues[0].options == ["they're going", "they are going"]


def test_occurrence_defaults_to_the_first_and_kind_is_not_checked_here() -> None:
    issue = ReplyIssue(sentence="S1", original="teh", kind="Typo", explanation="", options=["the"])

    assert issue.occurrence == 1
    assert issue.kind == "Typo"


@pytest.mark.parametrize(
    "changes",
    [
        {"options": []},
        {"options": ["a", "b", "c", "d", "e"]},
        {"original": ""},
        {"occurrence": 0},
        {"rewritten": "The whole sentence."},
    ],
)
def test_rejects_malformed_issues(changes: dict[str, object]) -> None:
    data: dict[str, object] = {
        "sentence": "S1",
        "original": "teh",
        "kind": "Spelling",
        "explanation": "",
        "options": ["the"],
    } | changes

    with pytest.raises(ValidationError):
        CheckReply.model_validate({"issues": [data]})
