from dataclasses import replace

from langcheck.context.models import DocumentContext, Register


def test_empty_context_says_nothing() -> None:
    context = DocumentContext.empty()

    assert (context.summary, context.dialect, context.notes) == ("", "", "")
    assert context.register is Register.NEUTRAL


def test_version_is_a_fingerprint_of_the_content() -> None:
    context = DocumentContext.empty()

    assert context.version == DocumentContext.empty().version
    assert len(context.version) == 16
    assert replace(context, register=Register.FORMAL).version != context.version


def test_register_values() -> None:
    assert [r.value for r in Register] == [
        "formal",
        "neutral",
        "informal",
        "fiction-dialogue-heavy",
    ]
