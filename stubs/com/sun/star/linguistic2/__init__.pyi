# Minimal stubs for com.sun.star.linguistic2, covering what langcheck.libreoffice uses.
from com.sun.star.lang import Locale

class XProofreader: ...
class XSupportedLocales: ...
class XLinguServiceEventBroadcaster: ...

class SingleProofreadingError:
    nErrorStart: int
    nErrorLength: int
    nErrorType: int
    aRuleIdentifier: str
    aShortComment: str
    aFullComment: str
    aSuggestions: tuple[str, ...]
    aProperties: tuple[object, ...]
    def __init__(self) -> None: ...

class ProofreadingResult:
    aDocumentIdentifier: str
    xFlatParagraph: object
    aText: str
    aLocale: Locale
    nStartOfSentencePosition: int
    nBehindEndOfSentencePosition: int
    nStartOfNextSentencePosition: int
    aErrors: tuple[SingleProofreadingError, ...]
    aProperties: tuple[object, ...]
    xProofreader: object
    def __init__(self) -> None: ...

class LinguServiceEvent:
    Source: object
    nEvent: int
    def __init__(self) -> None: ...
