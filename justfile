# List available tasks.
default:
    @just --list

# Install locked dependencies with a PyTorch build (cpu, cuda or rocm), the evaluation tools and the pre-commit hooks.
setup build="cpu":
    uv sync --locked --extra {{build}} --extra eval
    uv run --no-sync pre-commit install

# Format Python code.
format:
    uv run ruff format .

# Check formatting and lint rules.
lint:
    uv run ruff format --check .
    uv run ruff check .

# Check types.
typecheck:
    uv run ty check

# Check that tests mirror src/langcheck/ and the skills link is intact.
layout:
    uv run python scripts/check_layout.py

# Run the default test suite (no model downloads, no LLM).
test:
    uv run pytest

# Run every test, including those that download models or need an LLM.
test-all:
    uv run pytest -m ""

# Run the test suite with branch coverage (writes coverage.xml).
coverage:
    uv run pytest --cov --cov-report=term --cov-report=xml

# Run all required checks.
check: lint typecheck layout test

# Build the VS Code extension into editors/vscode/langcheck.vsix (needs Node.js 22).
[working-directory: 'editors/vscode']
vscode:
    npm ci --no-audit --no-fund
    npm run package

# Build the LibreOffice extension into dist/langcheck.oxt, after checking it runs on Python 3.8.
libreoffice:
    uv run --no-project --python 3.8 python scripts/check_libreoffice_python.py
    uv run python scripts/build_oxt.py

# Run the tests that drive a real LibreOffice (needs soffice and its Python uno).
test-libreoffice:
    uv run pytest -m libreoffice

# Remove finished worktrees and merged branches (flags: --fetch, --dry-run).
prune-worktrees *flags:
    uv run python scripts/prune_worktrees.py {{flags}}

# Run the CLI (pass arguments after the recipe name; quoted arguments stay whole).
[positional-arguments]
cli *args:
    uv run langcheck "$@"

# Render design-docs/images/*.dot to PNG next to each source (needs Graphviz).
diagrams:
    uv run python -c "import pathlib, subprocess; [subprocess.run(['dot', '-Tpng', '-Gdpi=110', str(f), '-o', str(f.with_suffix('.png'))], check=True) for f in sorted(pathlib.Path('design-docs/images').glob('*.dot'))]"
