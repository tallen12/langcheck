"""``LanguageServerRunner`` over stdio, as editors start language servers."""

from pygls.lsp.server import LanguageServer


class StdioRunner:
    """Serves the editor that started this process on stdin and stdout."""

    def run(self, server: LanguageServer) -> None:
        """Serve until shut down."""
        server.start_io()
