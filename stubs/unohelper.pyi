# Minimal stubs for LibreOffice's unohelper, covering what langcheck.libreoffice uses.
from collections.abc import Callable

class Base: ...

class ImplementationHelper:
    def addImplementation(
        self, ctor: Callable[..., object], implementationName: str, serviceNames: tuple[str, ...]
    ) -> None: ...
