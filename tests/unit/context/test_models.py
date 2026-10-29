from langcheck.context.models import DocumentContext, Register


def test_empty_context_says_nothing() -> None:
    context = DocumentContext.empty()

    assert (context.summary, context.dialect, context.notes) == ("", "", "")
    assert context.glossary == ()
    assert context.register is Register.NEUTRAL
    assert context.version == 0


def test_register_values() -> None:
    assert [r.value for r in Register] == [
        "formal",
        "neutral",
        "informal",
        "fiction-dialogue-heavy",
    ]
