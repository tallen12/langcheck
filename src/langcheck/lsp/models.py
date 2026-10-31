"""What the proxy hands to the editor, independent of any LSP library."""

from dataclasses import dataclass
from enum import StrEnum

from langcheck.check.suggestion import Confidence
from langcheck.lsp.positions import Range

SOURCE = "langcheck"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One suggestion at an editor range, with its options best first."""

    range: Range
    message: str
    kind: str
    suggestion_id: str
    original: str
    options: tuple[str, ...]
    confidence: Confidence = Confidence.HIGH


class ProxyCommand(StrEnum):
    """Commands the proxy registers with the editor."""

    DISMISS = "langcheck.dismiss"
    ADD_TO_DICTIONARY = "langcheck.addToDictionary"


@dataclass(frozen=True, slots=True)
class ReplaceAction:
    """A quick fix that replaces the diagnostic's range with ``replacement``."""

    title: str
    uri: str
    diagnostic: Diagnostic
    replacement: str
    preferred: bool


@dataclass(frozen=True, slots=True)
class CommandAction:
    """A code action that runs a proxy command with string arguments."""

    title: str
    diagnostic: Diagnostic
    command: ProxyCommand
    arguments: tuple[str, ...]


type CodeAction = ReplaceAction | CommandAction
