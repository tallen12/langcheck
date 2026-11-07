import pytest

from langcheck.judging.roster import (
    DEFAULT_PANEL,
    DEFAULT_STANDBY,
    RosterError,
    choose_panel,
    model_family,
)


def test_families_ignore_versions_sizes_and_namespaces() -> None:
    assert model_family("gemma4:12b-it-q4_K_M") == "gemma"
    assert model_family("qwen3.5:9b-q8_0") == model_family("user/Qwen3:14b") == "qwen"
    assert model_family("llama3.1:latest") == "llama"


def test_without_a_corrector_the_panel_is_kept() -> None:
    assert choose_panel(DEFAULT_PANEL, corrector=None) == DEFAULT_PANEL


def test_the_correctors_family_gives_way_to_the_standby() -> None:
    panel = choose_panel(DEFAULT_PANEL, corrector="gemma4:12b-it-q4_K_M")

    assert panel == ("qwen3:14b", DEFAULT_STANDBY, "llama3.1:latest")
    assert choose_panel(DEFAULT_PANEL, corrector="mistral:7b") == DEFAULT_PANEL


def test_a_panel_that_cannot_avoid_the_family_is_refused() -> None:
    with pytest.raises(RosterError):
        choose_panel(DEFAULT_PANEL, corrector="qwen3:8b")
    with pytest.raises(RosterError):
        choose_panel(("gemma3", "gemma4"), corrector="gemma4")
    with pytest.raises(RosterError):
        choose_panel(("a", "a"), corrector=None)
