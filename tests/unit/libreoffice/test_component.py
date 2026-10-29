import importlib
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from langcheck.libreoffice.cache import DocumentCache
from langcheck.libreoffice.schema import DocumentState, Suggestion
from langcheck.libreoffice.worker import CheckWorker

if TYPE_CHECKING:
    # Typing only: at run time the module loads under the fake_uno fixture.
    from com.sun.star.linguistic2 import ProofreadingResult

    from langcheck.libreoffice.component import LangcheckProofreader

TEXT = "Read teh book.\nSecond paragraph."
TYPO = Suggestion("sg_5", 5, 8, "teh", "Spelling", "Misspelt.", ("the", "tea"))


class Text:
    def __init__(self, value: str) -> None:
        self.value = value

    def getString(self) -> str:
        return self.value


class Document:
    def __init__(self, text: str, url: str = "") -> None:
        self.text = text
        self.url = url
        self.RuntimeUID = ""

    def getText(self) -> Text:
        return Text(self.text)

    def getURL(self) -> str:
        return self.url


class Components:
    def __init__(self, items: list[object]) -> None:
        self.items = items

    def createEnumeration(self) -> Components:
        return Components(list(self.items))

    def hasMoreElements(self) -> bool:
        return bool(self.items)

    def nextElement(self) -> object:
        return self.items.pop(0)


class Configuration:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def getByName(self, name: str) -> object:
        return self.values[name]


class Services:
    """Answers as the Desktop and the ConfigurationProvider."""

    def __init__(self, documents: list[object], config: dict[str, object]) -> None:
        self.documents = documents
        self.config = config
        self.node_paths: list[object] = []

    def getComponents(self) -> Components:
        return Components(self.documents)

    def createInstanceWithArguments(self, name: str, arguments: Sequence[object]) -> Configuration:
        self.node_paths += [getattr(a, "Value", None) for a in arguments]
        return Configuration(self.config)


class ServiceManager:
    def __init__(self, services: Services) -> None:
        self.services = services

    def createInstanceWithContext(self, name: str, context: object) -> Services:
        return self.services


class Context:
    def __init__(self, services: Services) -> None:
        self.ServiceManager = ServiceManager(services)


class Listener:
    def __init__(self) -> None:
        self.events: list[object] = []

    def processLinguServiceEvent(self, event: object) -> None:
        self.events.append(event)


class Service:
    def __init__(self) -> None:
        self.puts: list[tuple[str, str]] = []

    def put_document(self, uri: str, text: str) -> int:
        self.puts.append((uri, text))
        return len(self.puts)

    def document(self, uri: str) -> DocumentState:
        return DocumentState(len(self.puts), "done", (TYPO,))


@pytest.fixture
def component(fake_uno: None) -> ModuleType:
    return importlib.import_module("langcheck.libreoffice.component")


def make(
    module: ModuleType,
    documents: list[object],
    *,
    service: Service | None = None,
    config: dict[str, object] | None = None,
) -> tuple[LangcheckProofreader, list[CheckWorker]]:
    workers: list[CheckWorker] = []

    def factory(cache: DocumentCache, on_changed: Callable[[str], None]) -> CheckWorker | None:
        if service is None:
            return None
        worker = CheckWorker(service, cache, on_changed=on_changed, log=lambda line: None)
        workers.append(worker)
        return worker

    # Imported here, after fake_uno installed the UNO modules it needs.
    from langcheck.libreoffice.component import LangcheckProofreader

    context = Context(Services(documents, config or {"Url": "", "ApiKey": ""}))
    return LangcheckProofreader(context, workers=factory), workers


def proofread(proofreader: LangcheckProofreader, text: str, start: int = 0) -> ProofreadingResult:
    from com.sun.star.lang import Locale

    return proofreader.doProofreading("1", text, Locale("en", "US", ""), start, len(text), ())


def test_registers_one_proofreader_implementation(component: ModuleType) -> None:
    [(ctor, name, services)] = component.g_ImplementationHelper.implementations

    assert ctor is component.LangcheckProofreader
    assert name == "org.langcheck.libreoffice.Proofreader"
    assert services == ("com.sun.star.linguistic2.Proofreader",)


def test_service_info_and_locales(component: ModuleType) -> None:
    proofreader, _ = make(component, [])
    locale = component.Locale

    assert proofreader.supportsService("com.sun.star.linguistic2.Proofreader")
    assert proofreader.getServiceDisplayName(locale("en")) == "langcheck"
    assert proofreader.hasLocale(locale("en", "GB"))
    assert not proofreader.hasLocale(locale("de", "DE"))
    assert {locale_.Country for locale_ in proofreader.getLocales()} >= {"US", "GB"}
    assert proofreader.isSpellChecker() is False


def test_answers_from_the_cache_and_asks_libreoffice_to_check_again(
    component: ModuleType,
) -> None:
    service = Service()
    listener = Listener()
    proofreader, workers = make(
        component, [object(), Document(TEXT, "file:///a.odt")], service=service
    )
    proofreader.addLinguServiceEventListener(listener)

    first = proofread(proofreader, "Read teh book.")
    [worker] = workers
    assert worker.run_once()
    second = proofread(proofreader, "Read teh book.")
    later_sentence = proofread(proofreader, "Read teh book.", start=9)

    assert first.aErrors == ()
    assert first.nBehindEndOfSentencePosition == len("Read teh book.")
    assert service.puts == [("file:///a.odt", TEXT)]
    [event] = listener.events
    assert vars(event)["nEvent"] == component.PROOFREAD_AGAIN
    [error] = second.aErrors
    assert (error.nErrorStart, error.nErrorLength) == (5, 3)
    assert error.aSuggestions == ("the", "tea")
    assert error.aFullComment == "Misspelt."
    assert later_sentence.aErrors == ()


def test_unsaved_documents_get_one_stable_untitled_uri(component: ModuleType) -> None:
    service = Service()
    # pyuno returns a new wrapper on every lookup; the RuntimeUID stays the same.
    first, again = Document(TEXT), Document(TEXT + " More.")
    first.RuntimeUID = again.RuntimeUID = "7"
    documents: list[object] = [first]
    proofreader, workers = make(component, documents, service=service)

    proofread(proofreader, "Read teh book.")
    workers[0].run_once()
    documents[0] = again
    proofread(proofreader, "Read teh book.")
    workers[0].run_once()

    uris = {uri for uri, _ in service.puts}
    [uri] = uris
    assert uri.startswith("untitled:")


def test_paragraphs_outside_any_document_and_missing_servers_get_nothing(
    component: ModuleType,
) -> None:
    proofreader, workers = make(component, [Document("Other.")])

    assert proofread(proofreader, "Read teh book.").aErrors == ()
    assert workers == []


def test_listeners_can_be_removed(component: ModuleType) -> None:
    proofreader, _ = make(component, [])
    listener = Listener()
    proofreader.addLinguServiceEventListener(listener)
    proofreader.removeLinguServiceEventListener(listener)
    proofreader.removeLinguServiceEventListener(listener)

    proofreader.proofread_again("file:///a.odt")

    assert listener.events == []


def test_the_default_worker_reads_the_configuration(
    component: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LANGCHECK_SERVER__URL", raising=False)
    services = Services([], {"Url": "", "ApiKey": ""})
    proofreader = component.LangcheckProofreader(Context(services))

    assert proofreader._ensure_worker() is None
    assert services.node_paths == [component.SETTINGS_NODE, component.SETTINGS_NODE]

    configured = component.LangcheckProofreader(
        Context(Services([], {"Url": "http://127.0.0.1:9", "ApiKey": 7}))
    )
    worker = configured._ensure_worker()
    assert isinstance(worker, CheckWorker)
    worker.stop()


def test_to_uno_error_falls_back_to_the_kind(component: ModuleType) -> None:
    from langcheck.libreoffice.cache import ParagraphError

    error = component.to_uno_error(ParagraphError(1, 2, "Spelling", "", ("x",), "sg"))

    assert (error.aShortComment, error.aFullComment) == ("Spelling", "Spelling")
    assert error.nErrorType == component.PROOFREADING
