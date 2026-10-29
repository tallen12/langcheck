from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from langcheck.settings.loading import (
    CONFIG_ENV,
    ClientSettings,
    ServerSettings,
    SettingsFileError,
    load_settings,
    masked_dump,
    read_yaml,
    resolve_config_path,
)
from langcheck.settings.models import AuthMode

YAML = {"llm": {"model": "qwen3:8b", "base_url": "http://yaml/v1"}}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["LANGCHECK_LLM__MODEL", "LANGCHECK_LLM__BASE_URL", "LANGCHECK_AUTH__MODE"]:
        monkeypatch.delenv(name, raising=False)


def test_defaults_apply_without_any_source() -> None:
    settings = load_settings(ServerSettings, overrides={}, yaml_data={})

    assert settings.auth.mode is AuthMode.API_KEY
    assert settings.http.port == 8000
    assert settings.database.url == "sqlite:///langcheck.db"


def test_precedence_is_overrides_then_environment_then_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_yaml = load_settings(ClientSettings, overrides={}, yaml_data=YAML)
    monkeypatch.setenv("LANGCHECK_LLM__MODEL", "llama3")
    from_env = load_settings(ClientSettings, overrides={}, yaml_data=YAML)
    from_option = load_settings(
        ClientSettings, overrides={"llm": {"model": "gemma3"}}, yaml_data=YAML
    )

    assert from_yaml.llm.model == "qwen3:8b"
    assert from_env.llm.model == "llama3"
    assert from_option.llm.model == "gemma3"
    assert from_option.llm.base_url == "http://yaml/v1"


def test_environment_values_are_validated_and_name_the_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGCHECK_AUTH__MODE", "magic")

    with pytest.raises(ValidationError, match=r"auth\.mode"):
        load_settings(ServerSettings, overrides={}, yaml_data={})


def test_unknown_yaml_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="lmm"):
        load_settings(ServerSettings, overrides={}, yaml_data={"lmm": {}})
    with pytest.raises(ValidationError, match="modle"):
        load_settings(ServerSettings, overrides={}, yaml_data={"llm": {"modle": "x"}})


def test_client_settings_do_not_take_server_sections() -> None:
    with pytest.raises(ValidationError, match="auth"):
        load_settings(ClientSettings, overrides={}, yaml_data={"auth": {"mode": "none"}})


def test_masked_dump_hides_secrets() -> None:
    settings = load_settings(
        ClientSettings,
        overrides={"server": {"api_key": "s3cret"}, "llm": {"api_key": "k3y"}},
        yaml_data={},
    )

    dumped = masked_dump(settings)

    assert settings.server.api_key == SecretStr("s3cret")
    assert "s3cret" not in str(dumped)
    assert "k3y" not in str(dumped)
    assert dumped["server"] == {"url": None, "api_key": "**********", "timeout": 300.0}


def test_masked_dump_hides_passwords_in_urls() -> None:
    settings = load_settings(
        ServerSettings,
        overrides={"database": {"url": "postgresql://app:p4ss@db.test:5432/langcheck"}},
        yaml_data={},
    )

    dumped = masked_dump(settings)

    assert "p4ss" not in str(dumped)
    assert dumped["database"] == {"url": "postgresql://app:**********@db.test:5432/langcheck"}
    assert masked_dump(load_settings(ServerSettings, overrides={}, yaml_data={}))["database"] == {
        "url": "sqlite:///langcheck.db"
    }


def test_resolve_config_path_prefers_the_option_then_the_environment(tmp_path: Path) -> None:
    option = tmp_path / "option.yaml"
    from_env = tmp_path / "env.yaml"
    default = tmp_path / "langcheck.yaml"
    for path in (option, from_env, default):
        path.write_text("{}", encoding="utf-8")
    environ = {CONFIG_ENV: str(from_env)}

    assert resolve_config_path(option, environ=environ, cwd=tmp_path) == option
    assert resolve_config_path(None, environ=environ, cwd=tmp_path) == from_env
    assert resolve_config_path(None, environ={}, cwd=tmp_path) == default
    assert resolve_config_path(None, environ={CONFIG_ENV: ""}, cwd=tmp_path) == default


def test_resolve_config_path_without_a_default_file(tmp_path: Path) -> None:
    assert resolve_config_path(None, environ={}, cwd=tmp_path) is None


def test_an_explicit_config_path_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SettingsFileError, match="not found"):
        resolve_config_path(missing, environ={}, cwd=tmp_path)
    with pytest.raises(SettingsFileError, match="not found"):
        resolve_config_path(None, environ={CONFIG_ENV: str(missing)}, cwd=tmp_path)


def test_read_yaml(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("llm:\n  model: qwen3:8b\n", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    assert read_yaml(good) == {"llm": {"model": "qwen3:8b"}}
    assert read_yaml(empty) == {}
    assert read_yaml(None) == {}


@pytest.mark.parametrize(
    ("content", "message"),
    [("- a\n- b\n", "must hold a mapping"), ("llm: [unclosed\n", "invalid YAML")],
)
def test_read_yaml_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SettingsFileError, match=message):
        read_yaml(path)
