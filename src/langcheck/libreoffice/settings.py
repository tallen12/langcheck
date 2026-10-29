"""Where the extension finds the service: environment variables, then its own
LibreOffice configuration (Tools > Options > Advanced > Expert Configuration,
``org.langcheck.libreoffice.Settings``).
"""

from __future__ import annotations

from collections.abc import Mapping

from langcheck.libreoffice.client import Connection

URL_VARIABLE = "LANGCHECK_SERVER__URL"
KEY_VARIABLE = "LANGCHECK_SERVER__API_KEY"


def connection_from(
    environ: Mapping[str, str], *, configured_url: str, configured_key: str
) -> Connection | None:
    """The service connection, or ``None`` when no URL is set anywhere.

    The environment wins, using the same variables as the ``langcheck`` CLI.
    """
    url = environ.get(URL_VARIABLE, "").strip() or configured_url.strip()
    if not url:
        return None
    key = environ.get(KEY_VARIABLE, "").strip() or configured_key.strip()
    return Connection(url, key)
