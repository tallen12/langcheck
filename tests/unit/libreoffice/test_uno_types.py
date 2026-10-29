from langcheck.libreoffice.uno_types import document_text


class Text:
    def __init__(self, value: object) -> None:
        self.value = value

    def getString(self) -> object:
        return self.value


class Document:
    def __init__(self, text: object, url: object = "") -> None:
        self.text = text
        self.url = url

    def getText(self) -> Text:
        return Text(self.text)

    def getURL(self) -> object:
        return self.url


def test_reads_writer_documents_by_their_methods() -> None:
    assert document_text(Document("Hi.", "file:///a.odt")) == ("file:///a.odt", "Hi.")
    assert document_text(Document("Hi.")) == ("", "Hi.")


def test_other_components_are_not_documents() -> None:
    assert document_text(object()) is None
    assert document_text(Document(7)) is None
    assert document_text(Document("Hi.", None)) is None
