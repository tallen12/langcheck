import pytest
from pydantic import SecretStr, ValidationError

from langcheck.llm.models import ClientConfig
from langcheck.scoring.factory import ScorerAdapter, ScorerConfig
from langcheck.scoring.policy import PLL_DEFAULT, KindPolicy, ScoringPolicy
from langcheck.settings.models import HttpSettings, LLMSettings, ScorerSettings
from langcheck.taxonomy.kinds import ErrorKind


def test_llm_settings_convert_to_the_client_config() -> None:
    settings = LLMSettings(
        base_url="http://x/v1", model="m", api_key=SecretStr("k"), retries=1, concurrency=2
    )

    assert settings.client_config() == ClientConfig(
        base_url="http://x/v1", model="m", api_key="k", retries=1, concurrency=2
    )
    assert LLMSettings().client_config().api_key is None


def test_sections_validate_values_and_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        HttpSettings(port=0)
    with pytest.raises(ValidationError):
        LLMSettings.model_validate({"retries": -1})
    with pytest.raises(ValidationError):
        LLMSettings.model_validate({"modle": "x"})


def test_scorer_settings_default_to_no_scorer() -> None:
    assert ScorerSettings().scorer_config() == ScorerConfig()


def test_scorer_settings_convert_adapters_and_policy() -> None:
    settings = ScorerSettings.model_validate(
        {
            "adapters": ["pll", "cola", "pll"],
            "device": "cuda",
            "policy": {"Punctuation": {"filter": True, "margin": 0.25}},
        }
    )

    config = settings.scorer_config()

    assert config.adapters == (ScorerAdapter.PLL, ScorerAdapter.COLA)
    assert config.device == "cuda"
    assert config.policy == ScoringPolicy(
        {ErrorKind.PUNCTUATION: KindPolicy(filter=True, margin=0.25)}
    )
    with pytest.raises(ValidationError):
        ScorerSettings.model_validate({"adapters": ["bert"]})


def test_pll_alone_defaults_to_the_evaluated_policy() -> None:
    def policy(settings: dict[str, object]) -> ScoringPolicy:
        return ScorerSettings.model_validate(settings).scorer_config().policy

    assert policy({"adapters": ["pll"]}) == PLL_DEFAULT
    assert policy({"adapters": ["pll", "cola"]}) == ScoringPolicy()
    assert policy({"adapters": ["cola"]}) == ScoringPolicy()
    assert policy({"adapters": ["pll"], "policy": {}}) == ScoringPolicy()
