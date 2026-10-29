from langcheck.libreoffice.uno_types import TextDocument


class Document:
    def getText(self) -> Document:
        return self

    def getString(self) -> str:
        return "Hi."

    def getURL(self) -> str:
        return ""


def test_text_documents_are_recognised_at_run_time() -> None:
    assert isinstance(Document(), TextDocument)
    assert not isinstance(object(), TextDocument)
