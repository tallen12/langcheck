import httpx

from langcheck.api.client import ServiceClient
from langcheck.lsp.protocols import DocumentsService


def test_the_service_client_satisfies_the_protocol() -> None:
    client: DocumentsService = ServiceClient(
        httpx.AsyncClient(), base_url="http://svc.test", api_key=None
    )

    assert client is not None
