import pytest
from pydantic import SecretStr, ValidationError

from langcheck.llm.models import ClientConfig
from langcheck.settings.models import HttpSettings, LLMSettings


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
