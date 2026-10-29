# Minimal stubs for com.sun.star.beans, covering what langcheck.libreoffice uses.
class PropertyValue:
    Name: str
    Value: object
    def __init__(
        self, Name: str = "", Handle: int = 0, Value: object = None, State: int = 0
    ) -> None: ...
