"""The UNO grammar checker LibreOffice loads: ``org.langcheck.libreoffice.Proofreader``.

``doProofreading`` answers each paragraph from the cache at once and hands the whole
document to the worker; when the worker's results change a document's suggestions, the
component fires ``PROOFREAD_AGAIN`` and LibreOffice asks again for every paragraph.
"""

import os
import sys
import threading
import uuid
from collections.abc import Callable, Sequence

import unohelper
from com.sun.star.beans import PropertyValue
from com.sun.star.lang import Locale, XServiceDisplayName, XServiceInfo, XServiceName
from com.sun.star.linguistic2 import (
    LinguServiceEvent,
    ProofreadingResult,
    SingleProofreadingError,
    XLinguServiceEventBroadcaster,
    XProofreader,
    XSupportedLocales,
)

from langcheck.libreoffice.cache import DocumentCache, ParagraphError
from langcheck.libreoffice.client import ServiceClient
from langcheck.libreoffice.settings import connection_from
from langcheck.libreoffice.uno_types import (
    ComponentContext,
    LinguServiceEventListener,
    TextDocument,
)
from langcheck.libreoffice.worker import CheckWorker

IMPLEMENTATION = "org.langcheck.libreoffice.Proofreader"
SERVICES = ("com.sun.star.linguistic2.Proofreader",)
SETTINGS_NODE = "/org.langcheck.libreoffice.Settings/Server"
PROOFREAD_AGAIN = 8
"""``com.sun.star.linguistic2.LinguServiceEventFlags.PROOFREAD_AGAIN``."""
PROOFREADING = 4
"""``com.sun.star.text.TextMarkupType.PROOFREADING``."""
LANGUAGES = ("en",)
UNTITLED = "untitled:"

type WorkerFactory = Callable[[DocumentCache, Callable[[str], None]], CheckWorker | None]
"""Makes the running worker on first use; ``None`` when no server is configured."""


class LangcheckProofreader(
    unohelper.Base,
    XProofreader,
    XServiceInfo,
    XServiceName,
    XServiceDisplayName,
    XSupportedLocales,
    XLinguServiceEventBroadcaster,
):
    """One instance per LibreOffice process (LibreOffice creates it once)."""

    def __init__(
        self, context: ComponentContext, *args: object, workers: WorkerFactory | None = None
    ) -> None:
        self._context = context
        self._cache = DocumentCache()
        self._listeners: list[LinguServiceEventListener] = []
        self._lock = threading.Lock()
        self._make_worker = workers if workers is not None else self._default_worker
        self._worker: CheckWorker | None = None
        self._started = False
        self._untitled: dict[int, str] = {}

    # XServiceInfo, XServiceName, XServiceDisplayName

    def getImplementationName(self) -> str:
        """The implementation name LibreOffice registers."""
        return IMPLEMENTATION

    def supportsService(self, name: str) -> bool:
        """Whether this is a ``Proofreader``."""
        return name in SERVICES

    def getSupportedServiceNames(self) -> tuple[str, ...]:
        """The services implemented."""
        return SERVICES

    def getServiceName(self) -> str:
        """The service name, for LibreOffice's linguistic settings."""
        return IMPLEMENTATION

    def getServiceDisplayName(self, locale: Locale) -> str:
        """The name in Tools > Options > Language Settings > Writing Aids."""
        return "langcheck"

    # XSupportedLocales

    def getLocales(self) -> tuple[Locale, ...]:
        """English, in the regions LibreOffice offers."""
        return tuple(Locale("en", country, "") for country in ("US", "GB", "AU", "CA", "NZ"))

    def hasLocale(self, locale: Locale) -> bool:
        """Whether ``locale`` is English."""
        return locale.Language in LANGUAGES

    # XProofreader

    def isSpellChecker(self) -> bool:
        """langcheck reports spelling through grammar results, not as a spell checker."""
        return False

    def doProofreading(
        self,
        document_id: str,
        text: str,
        locale: Locale,
        start: int,
        end: int,
        properties: Sequence[object],
    ) -> ProofreadingResult:
        """The cached errors in the paragraph; queue the whole document for checking."""
        result = ProofreadingResult()
        result.aDocumentIdentifier = document_id
        result.aText = text
        result.aLocale = locale
        result.nStartOfSentencePosition = start
        # The whole paragraph is answered at once, so LibreOffice does not ask per sentence.
        result.nBehindEndOfSentencePosition = len(text)
        result.nStartOfNextSentencePosition = len(text)
        result.xProofreader = self
        result.aProperties = ()
        errors: list[SingleProofreadingError] = []
        found = self._document_for(text)
        worker = self._ensure_worker()
        if found is not None:
            uri, whole = found
            if worker is not None:
                worker.submit(uri, whole)
            errors = [
                to_uno_error(error)
                for error in self._cache.errors_for(uri, text)
                if error.start >= start
            ]
        result.aErrors = tuple(errors)
        return result

    def ignoreRule(self, rule: str, locale: Locale) -> None:
        """Rules are not ignorable yet; dismiss suggestions in other clients."""

    def resetIgnoreRules(self) -> None:
        """Nothing to reset."""

    # XLinguServiceEventBroadcaster

    def addLinguServiceEventListener(self, listener: LinguServiceEventListener) -> bool:
        """LibreOffice listens for ``PROOFREAD_AGAIN`` here."""
        with self._lock:
            self._listeners.append(listener)
        return True

    def removeLinguServiceEventListener(self, listener: LinguServiceEventListener) -> bool:
        """Stop notifying ``listener``."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
        return True

    def proofread_again(self, uri: str) -> None:
        """Ask LibreOffice to check every paragraph again (called by the worker)."""
        event = LinguServiceEvent()
        event.Source = self
        event.nEvent = PROOFREAD_AGAIN
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.processLinguServiceEvent(event)

    def _ensure_worker(self) -> CheckWorker | None:
        with self._lock:
            if not self._started:
                self._started = True
                self._worker = self._make_worker(self._cache, self.proofread_again)
            return self._worker

    def _document_for(self, paragraph: str) -> tuple[str, str] | None:
        """The URI and full text of the open document that holds ``paragraph``.

        The spike showed ``aDocumentIdentifier`` is not the component and there may be
        no current component, so the open text documents are searched instead.
        """
        desktop = self._context.ServiceManager.createInstanceWithContext(
            "com.sun.star.frame.Desktop", self._context
        )
        components = desktop.getComponents().createEnumeration()
        while components.hasMoreElements():
            component = components.nextElement()
            if not isinstance(component, TextDocument):
                continue
            whole = component.getText().getString()
            if paragraph in whole:
                return self._uri_of(component), whole
        return None

    def _uri_of(self, document: TextDocument) -> str:
        url = document.getURL()
        if url:
            return url
        # Unsaved documents get a stable temporary URI, so nothing about them persists.
        return self._untitled.setdefault(id(document), f"{UNTITLED}{uuid.uuid4()}")

    def _default_worker(
        self, cache: DocumentCache, on_changed: Callable[[str], None]
    ) -> CheckWorker | None:
        connection = connection_from(
            os.environ,
            configured_url=self._configured("Url"),
            configured_key=self._configured("ApiKey"),
        )
        if connection is None:
            return None
        worker = CheckWorker(ServiceClient(connection), cache, on_changed=on_changed, log=_log)
        worker.start()
        return worker

    def _configured(self, name: str) -> str:
        provider = self._context.ServiceManager.createInstanceWithContext(
            "com.sun.star.configuration.ConfigurationProvider", self._context
        )
        node = PropertyValue()
        node.Name = "nodepath"
        node.Value = SETTINGS_NODE
        access = provider.createInstanceWithArguments(
            "com.sun.star.configuration.ConfigurationAccess", (node,)
        )
        value = access.getByName(name)
        return value if isinstance(value, str) else ""


def to_uno_error(error: ParagraphError) -> SingleProofreadingError:
    """A ``SingleProofreadingError`` LibreOffice underlines, options as suggestions."""
    uno_error = SingleProofreadingError()
    uno_error.nErrorStart = error.start
    uno_error.nErrorLength = error.length
    uno_error.nErrorType = PROOFREADING
    uno_error.aRuleIdentifier = f"langcheck.{error.kind}"
    uno_error.aShortComment = error.kind
    uno_error.aFullComment = error.explanation or error.kind
    uno_error.aSuggestions = error.options
    uno_error.aProperties = ()
    return uno_error


def _log(message: str) -> None:
    # LibreOffice shows Python extensions' stderr when started from a terminal.
    sys.stderr.write(f"{message}\n")


g_ImplementationHelper = unohelper.ImplementationHelper()
g_ImplementationHelper.addImplementation(LangcheckProofreader, IMPLEMENTATION, SERVICES)
