"""Build the LibreOffice extension ``dist/langcheck.oxt``.

The package ``src/langcheck/libreoffice`` goes into ``pythonpath/``, which LibreOffice
adds to its Python's ``sys.path``; the static files come from ``editors/libreoffice``.

Run with ``uv run python scripts/build_oxt.py [OUTPUT]``.
"""

import sys
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "editors" / "libreoffice"
PACKAGE = ROOT / "src" / "langcheck"
DEFAULT_OUTPUT = ROOT / "dist" / "langcheck.oxt"


def build(output: Path) -> list[str]:
    """Write the extension to ``output``; the archive names it holds."""
    output.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STATIC.rglob("*")):
            if path.is_dir() or path.name == "README.md":
                continue
            name = path.relative_to(STATIC).as_posix()
            text = path.read_text(encoding="utf-8")
            if name == "description.xml":
                text = text.replace("@VERSION@", version("langcheck"))
            archive.writestr(name, text)
            names.append(name)
        sources = [PACKAGE / "__init__.py", *sorted((PACKAGE / "libreoffice").glob("*.py"))]
        for path in sources:
            name = (Path("pythonpath") / path.relative_to(PACKAGE.parent)).as_posix()
            archive.write(path, name)
            names.append(name)
    return names


def main() -> int:
    """Build to the path given, or ``dist/langcheck.oxt``."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    names = build(output)
    print(f"built {output} ({len(names)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
