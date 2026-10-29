"""Versioned prompt templates, loaded with ``importlib.resources``.

Templates live next to this file as ``<name>.md`` and use ``string.Template``
placeholders. A changed template gets a new file name (``check-v2.md``), never an edit in
place, because the prompt version is part of every result cache key.
"""
