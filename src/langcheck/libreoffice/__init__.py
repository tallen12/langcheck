"""LibreOffice proofreader extension.

LibreOffice runs this package in its own Python, which cannot see our virtualenv, so it
imports only the standard library, ``uno``/``unohelper``/``com.sun.star`` and itself.
``scripts/check_layout.py`` enforces this.
"""
