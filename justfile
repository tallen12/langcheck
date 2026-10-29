# List available tasks.
default:
    @just --list

# Install locked dependencies and the pre-commit hooks.
setup:
    uv sync --locked
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

# Run the CLI (pass arguments after the recipe name; quoted arguments stay whole).
[positional-arguments]
cli *args:
    uv run langcheck "$@"

# Render design-docs/images/*.dot to PNG next to each source (needs Graphviz).
diagrams:
    uv run python -c "import pathlib, subprocess; [subprocess.run(['dot', '-Tpng', '-Gdpi=110', str(f), '-o', str(f.with_suffix('.png'))], check=True) for f in sorted(pathlib.Path('design-docs/images').glob('*.dot'))]"
