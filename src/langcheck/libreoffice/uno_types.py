"""The parts of the UNO API the extension uses, as protocols.

UNO objects are dynamic; these protocols name only the calls the extension makes, so
the rest of the code is type-checked and tests can pass plain fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Enumeration(Protocol):
    """``com.sun.star.container.XEnumeration``."""

    def hasMoreElements(self) -> bool:
        """Whether another element follows."""
        ...

    def nextElement(self) -> object:
        """The next element."""
        ...


class ComponentList(Protocol):
    """``com.sun.star.container.XEnumerationAccess`` of open components."""

    def createEnumeration(self) -> Enumeration:
        """The open components."""
        ...


class NameAccess(Protocol):
    """``com.sun.star.container.XNameAccess`` of a configuration node."""

    def getByName(self, name: str) -> object:
        """The value of a property."""
        ...


class Service(Protocol):
    """A service instance; each is used only for the calls its role needs."""

    def getComponents(self) -> ComponentList:
        """The open documents."""
        ...

    def createInstanceWithArguments(self, name: str, arguments: Sequence[object]) -> NameAccess:
        """A configuration access for a node."""
        ...


class MultiComponentFactory(Protocol):
    """``com.sun.star.lang.XMultiComponentFactory``."""

    def createInstanceWithContext(self, name: str, context: ComponentContext) -> Service:
        """A new service instance."""
        ...


class ComponentContext(Protocol):
    """``com.sun.star.uno.XComponentContext``."""

    @property
    def ServiceManager(self) -> MultiComponentFactory:
        """The service factory."""
        ...


class LinguServiceEventListener(Protocol):
    """``com.sun.star.linguistic2.XLinguServiceEventListener``."""

    def processLinguServiceEvent(self, event: object) -> None:
        """React to a linguistic service event."""
        ...


def document_text(component: object) -> tuple[str, str] | None:
    """The URL (empty when unsaved) and full text of a Writer document, or ``None`` for
    any other component.

    Duck-typed with ``getattr``: pyuno objects break ``isinstance`` checks against
    runtime-checkable protocols (the spike's integration test caught this).
    """
    get_text = getattr(component, "getText", None)
    get_url = getattr(component, "getURL", None)
    if not callable(get_text) or not callable(get_url):
        return None
    get_string = getattr(get_text(), "getString", None)
    if not callable(get_string):
        return None
    url: object = get_url()
    text: object = get_string()
    if not isinstance(url, str) or not isinstance(text, str):
        return None
    return url, text
