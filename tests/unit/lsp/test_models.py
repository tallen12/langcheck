from langcheck.lsp.models import CommandAction, Diagnostic, ProxyCommand, ReplaceAction
from langcheck.lsp.positions import Position, Range

DIAGNOSTIC = Diagnostic(
    Range(Position(0, 0), Position(0, 3)), "m", "Spelling", "sg_1", "teh", ("the",)
)


def test_command_names_match_the_extension() -> None:
    assert ProxyCommand.DISMISS.value == "langcheck.dismiss"
    assert ProxyCommand.ADD_TO_DICTIONARY.value == "langcheck.addToDictionary"


def test_actions_carry_their_diagnostic() -> None:
    replace = ReplaceAction("Replace", "file:///a.md", DIAGNOSTIC, "the", preferred=True)
    command = CommandAction("Dismiss", DIAGNOSTIC, ProxyCommand.DISMISS, ("file:///a.md", "sg_1"))

    assert replace.diagnostic == command.diagnostic == DIAGNOSTIC
