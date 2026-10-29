"""Check ``langcheck.libreoffice`` runs on the oldest Python LibreOffice ships (3.8).

Compiles every module and imports those that do not need UNO. Run with
``uv run --no-project --python 3.8 python scripts/check_libreoffice_python.py``; this file
itself must stay Python 3.8 compatible.
"""

import compileall
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "langcheck" / "libreoffice"
NEEDS_UNO = {"component"}


def main() -> int:
    """Compile and import; 1 on any failure."""
    if not compileall.compile_dir(str(PACKAGE), quiet=1):
        return 1
    sys.path.insert(0, str(ROOT / "src"))
    for path in sorted(PACKAGE.glob("*.py")):
        if path.stem not in NEEDS_UNO:
            importlib.import_module(f"langcheck.libreoffice.{path.stem}".replace(".__init__", ""))
    print(f"langcheck.libreoffice works on Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
