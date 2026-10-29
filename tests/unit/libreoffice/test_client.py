import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from langcheck.libreoffice.client import Connection, ServiceClient, ServiceError, urlopen

DOCUMENT = {"version": 3, "state": "done", "suggestions": [], "unchecked": []}


class Recorder:
    def __init__(self, reply: bytes | Exception) -> None:
        self.reply = reply
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> bytes:
        self.requests.append((request, timeout))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_puts_the_text_with_the_bearer_key() -> None:
    opener = Recorder(b'{"version": 4}')
    client = ServiceClient(Connection("http://svc.test/", "lc_key"), opener=opener, timeout=5)

    assert client.put_document("file:///a b.odt", "Hi.") == 4

    [(request, timeout)] = opener.requests
    assert request.get_method() == "PUT"
    assert request.full_url == "http://svc.test/documents/file%3A%2F%2F%2Fa%20b.odt"
    assert request.get_header("Authorization") == "Bearer lc_key"
    assert request.data == json.dumps({"text": "Hi."}).encode()
    assert timeout == 5


def test_reads_the_document_without_a_key() -> None:
    opener = Recorder(json.dumps(DOCUMENT).encode())
    client = ServiceClient(Connection("http://svc.test"), opener=opener)

    assert client.document("file:///a.odt").version == 3
    request = opener.requests[0][0]
    assert request.full_url.endswith("/suggestions")
    assert request.get_header("Authorization") is None
    assert "lc_" not in repr(Connection("http://x", "lc_secret"))


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (urllib.error.HTTPError("http://svc.test", 401, "no", Message(), None), "HTTP 401"),
        (urllib.error.URLError("refused"), "cannot reach"),
        (b"not json", "invalid JSON"),
    ],
)
def test_failures_become_service_errors(reply: bytes | Exception, message: str) -> None:
    client = ServiceClient(Connection("http://svc.test"), opener=Recorder(reply))

    with pytest.raises(ServiceError, match=message):
        client.document("file:///a.odt")


def test_urlopen_reads_a_local_file(tmp_path: Path) -> None:
    path = tmp_path / "reply.json"
    path.write_text('{"version": 1}', encoding="utf-8")

    assert urlopen(urllib.request.Request(path.as_uri()), 5) == b'{"version": 1}'
