"""Check the repository layout that tools and agents rely on.

Tests mirror ``src/langcheck/``: the test for ``src/langcheck/<path>/<name>.py`` is
``tests/<type>/<path>/test_<name>.py``, where ``<type>`` is one of ``TEST_TYPES``. Every
module needs a unit test; other test types are optional. Tests of the shared fixtures in
``tests/conftest.py`` are ``tests/<type>/test_conftest.py``.

It also checks that ``langcheck.libreoffice`` imports only what LibreOffice's Python has.

Run with ``uv run python scripts/check_layout.py``. Exits 1 and lists every violation.
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

TEST_TYPES = ("unit", "integration")
REQUIRED_TEST_TYPE = "unit"
PACKAGE = "langcheck"
INIT = "__init__.py"
CONFTEST = "conftest.py"


def python_files(root: Path) -> Iterator[Path]:
    """Yield ``.py`` files under ``root``, relative to it, skipping caches."""
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "__pycache__" not in relative.parts:
            yield relative


def source_modules(package: Path) -> set[Path]:
    """Modules that need tests: every ``.py`` file except package ``__init__.py`` files."""
    return {path for path in python_files(package) if path.name != INIT}


def source_packages(package: Path) -> set[Path]:
    """Package directories relative to ``package``, including the package itself."""
    return {path.parent for path in python_files(package) if path.name == INIT}


def test_file_for(module: Path) -> Path:
    """Map ``<path>/<name>.py`` to ``<path>/test_<name>.py``."""
    return module.with_name(f"test_{module.name}")


def check_tests(root: Path) -> list[str]:
    """Return a message for every test that breaks the mirror rule or is missing."""
    package = root / "src" / PACKAGE
    tests = root / "tests"
    modules = source_modules(package)
    packages = source_packages(package)
    allowed_tests = {test_file_for(module) for module in modules}
    errors: list[str] = []

    for entry in sorted(tests.iterdir()):
        if entry.name == "__pycache__" or entry.name in {INIT, CONFTEST}:
            continue
        if not (entry.is_dir() and entry.name in TEST_TYPES):
            errors.append(
                f"tests/{entry.name}: tests/ holds only {INIT}, {CONFTEST} and the test "
                f"type directories {', '.join(TEST_TYPES)}"
            )

    for test_type in TEST_TYPES:
        type_root = tests / test_type
        if not type_root.is_dir():
            errors.append(f"tests/{test_type}: missing test type directory")
            continue
        for directory in sorted(
            path.relative_to(type_root) for path in [type_root, *type_root.rglob("*")]
        ):
            if not (type_root / directory).is_dir() or "__pycache__" in directory.parts:
                continue
            shown = Path("tests", test_type, directory).as_posix()
            if directory not in packages:
                errors.append(f"{shown}: no matching package src/{PACKAGE}/{directory}")
            elif not (type_root / directory / INIT).is_file():
                # Without it, same-named test modules in different directories collide.
                errors.append(f"{shown}: missing {INIT}")
        for path in python_files(type_root):
            if path.name in {INIT, CONFTEST} or path == Path(f"test_{CONFTEST}"):
                continue
            if path not in allowed_tests:
                expected = path.with_name(path.name.removeprefix("test_"))
                errors.append(
                    f"tests/{test_type}/{path.as_posix()}: does not mirror a module; "
                    f"expected src/{PACKAGE}/{expected.as_posix()} to exist"
                )

    required_root = tests / REQUIRED_TEST_TYPE
    for module in sorted(modules):
        if not (required_root / test_file_for(module)).is_file():
            errors.append(
                f"src/{PACKAGE}/{module.as_posix()}: missing "
                f"tests/{REQUIRED_TEST_TYPE}/{test_file_for(module).as_posix()}"
            )
    return errors


def check_skills_link(root: Path) -> list[str]:
    """Claude Code loads the shared skills only through the ``.claude/skills`` symlink."""
    link = root / ".claude" / "skills"
    target = root / ".agents" / "skills"
    # A clone with core.symlinks=false checks the link out as a plain file.
    if not link.is_symlink() or link.resolve() != target.resolve():
        return [".claude/skills: must be a symlink to .agents/skills"]
    return []


LIBREOFFICE = ("libreoffice",)
UNO_MODULES = frozenset({"uno", "unohelper", "com"})


def check_libreoffice_imports(root: Path) -> list[str]:
    """LibreOffice runs ``langcheck.libreoffice`` in its own Python, which cannot see our
    virtualenv: return a message for every import there of anything but the standard
    library, the UNO modules and ``langcheck.libreoffice`` itself.
    """
    package = root / "src" / PACKAGE / Path(*LIBREOFFICE)
    own = ".".join((PACKAGE, *LIBREOFFICE))
    errors: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.partition(".")[0]
                allowed = (
                    top in sys.stdlib_module_names
                    or top in UNO_MODULES
                    or name == own
                    or name.startswith(f"{own}.")
                )
                if not allowed:
                    shown = path.relative_to(root).as_posix()
                    errors.append(
                        f"{shown}:{node.lineno}: imports {name}; LibreOffice's Python "
                        "has only the standard library, uno and this package"
                    )
    return errors


def main() -> int:
    """Print every layout violation and return the exit status."""
    root = Path(__file__).resolve().parent.parent
    errors = [*check_tests(root), *check_skills_link(root), *check_libreoffice_imports(root)]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"{len(errors)} layout error(s)", file=sys.stderr)
        return 1
    print("layout ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
