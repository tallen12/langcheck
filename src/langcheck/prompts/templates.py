"""Loading prompt templates shipped inside the package."""

from importlib import resources
from string import Template


def load_template(name: str) -> Template:
    """The template ``<name>.md`` from this package, for example ``check-v1``.

    Raises ``FileNotFoundError`` when there is no such template.
    """
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    return Template(text)
