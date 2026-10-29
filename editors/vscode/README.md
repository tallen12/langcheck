# langcheck for VS Code

Spelling and grammar suggestions from a shared [langcheck](https://github.com/tallen12/langcheck)
server, as diagnostics with quick fixes. Each option is a quick fix, best first, followed by
"Add to the dictionary" and "Dismiss".

## Setup

1. Install `langcheck` so the `langcheck` command is on your `PATH` (or set
   `langcheck.command` and `langcheck.args`, for example `uv` with `["run", "langcheck"]`).
2. Set `langcheck.serverUrl` to your server.
3. Run **langcheck: Set API Key** and paste the key from `langcheck admin add-user`. It is
   kept in VS Code's secret storage and passed to the language server in its environment.

Markdown and plain text are checked by default; change `langcheck.languages` for others.

## Build

From the repository root, `just vscode` builds `editors/vscode/langcheck.vsix`; install it
with **Extensions: Install from VSIX...**.
