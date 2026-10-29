# The component LibreOffice's Python loader imports; the code lives in
# pythonpath/langcheck/libreoffice, which LibreOffice adds to sys.path.
from langcheck.libreoffice.component import g_ImplementationHelper

__all__ = ["g_ImplementationHelper"]
