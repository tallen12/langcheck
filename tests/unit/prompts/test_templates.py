import pytest

from langcheck.prompts.templates import load_template


def test_loads_a_shipped_template() -> None:
    template = load_template("check-v1")

    assert "$kinds" in template.template
    assert "$schema" in template.template


def test_missing_templates_raise() -> None:
    with pytest.raises(FileNotFoundError):
        load_template("no-such-prompt")
