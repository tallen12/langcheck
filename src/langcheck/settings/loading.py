"""The two top-level settings classes and how they load.

Sources, highest first, merged field by field: values passed in (CLI options), then
``LANGCHECK_``-prefixed environment variables (nested with ``__``), then a YAML file,
then defaults. ``.env`` files and secrets directories are not read.
"""

import json
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from langcheck.settings.models import (
    AuthSettings,
    CheckerSettings,
    DatabaseSettings,
    HttpSettings,
    LLMSettings,
    ScorerSettings,
    ServerConnection,
)

ENV_PREFIX = "LANGCHECK_"
CONFIG_ENV = f"{ENV_PREFIX}CONFIG"
DEFAULT_CONFIG_NAME = "langcheck.yaml"

_CONFIG = SettingsConfigDict(
    env_prefix=ENV_PREFIX,
    env_nested_delimiter="__",
    extra="forbid",
    frozen=True,
)


@dataclass(frozen=True, slots=True)
class _Layers:
    overrides: Mapping[str, object]
    yaml: Mapping[str, object]


_NO_LAYERS = _Layers(overrides={}, yaml={})
# The caller's layers for the settings object being built. pydantic-settings builds its
# sources inside the class, so ``load_settings`` hands them over through this variable,
# set only for the duration of one construction.
_LAYERS: ContextVar[_Layers] = ContextVar("langcheck_settings_layers")


class SettingsFileError(Exception):
    """The YAML settings file is missing or is not a mapping."""


class _LayeredSettings(BaseSettings):
    """Reads init values, then the environment, then the YAML layer."""

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        layers = _LAYERS.get(_NO_LAYERS)
        return (
            _MappingSource(settings_cls, layers.overrides),
            env_settings,
            _MappingSource(settings_cls, layers.yaml),
        )


class ClientSettings(_LayeredSettings):
    """Settings for the ``langcheck`` client commands."""

    model_config = _CONFIG

    server: ServerConnection = Field(default_factory=ServerConnection)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    checker: CheckerSettings = Field(default_factory=CheckerSettings)
    scorer: ScorerSettings = Field(default_factory=ScorerSettings)


class ServerSettings(_LayeredSettings):
    """Settings for ``langcheck serve`` and ``langcheck admin``."""

    model_config = _CONFIG

    http: HttpSettings = Field(default_factory=HttpSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    checker: CheckerSettings = Field(default_factory=CheckerSettings)
    scorer: ScorerSettings = Field(default_factory=ScorerSettings)


class _MappingSource(PydanticBaseSettingsSource):
    """Settings values from an already parsed mapping, such as a YAML file."""

    def __init__(self, settings_cls: type[BaseSettings], data: Mapping[str, object]) -> None:
        super().__init__(settings_cls)
        self._data = dict(data)

    @override
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[object, str, bool]:
        return self._data.get(field_name), field_name, False

    @override
    def __call__(self) -> dict[str, object]:
        return self._data


def resolve_config_path(
    explicit: Path | None, *, environ: Mapping[str, str], cwd: Path
) -> Path | None:
    """The YAML file to read: ``--config``, else ``LANGCHECK_CONFIG``, else
    ``langcheck.yaml`` in ``cwd`` if it exists. A path given explicitly must exist.
    """
    from_env = environ.get(CONFIG_ENV, "")
    chosen = explicit or (Path(from_env) if from_env != "" else None)
    if chosen is not None:
        if not chosen.is_file():
            msg = f"settings file not found: {chosen}"
            raise SettingsFileError(msg)
        return chosen
    default = cwd / DEFAULT_CONFIG_NAME
    return default if default.is_file() else None


def read_yaml(path: Path | None) -> dict[str, object]:
    """The mapping in the YAML file at ``path``; empty without a file."""
    if path is None:
        return {}
    try:
        data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        msg = f"invalid YAML in {path}: {error}"
        raise SettingsFileError(msg) from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"{path} must hold a mapping of settings, not {type(data).__name__}"
        raise SettingsFileError(msg)
    return {str(key): value for key, value in data.items()}


def load_settings[S: _LayeredSettings](
    settings_cls: type[S],
    *,
    overrides: Mapping[str, object],
    yaml_data: Mapping[str, object],
) -> S:
    """Load ``settings_cls`` from ``overrides``, the environment, ``yaml_data`` and defaults.

    ``overrides`` holds only the values the user gave, nested like the settings, so an
    option left out does not mask the layers below. Raises ``pydantic.ValidationError``
    naming the bad field.
    """
    token = _LAYERS.set(_Layers(overrides=overrides, yaml=yaml_data))
    try:
        return settings_cls()
    finally:
        _LAYERS.reset(token)


MASK = "**********"


def masked_dump(settings: BaseModel) -> dict[str, object]:
    """``settings`` as plain JSON values with every secret masked, for ``config show``.

    Besides ``SecretStr`` fields, this masks passwords inside URLs, such as a database
    URL ``postgresql://user:password@host/db``.
    """
    data: object = _mask_url_passwords(json.loads(settings.model_dump_json()))
    if not isinstance(data, dict):
        msg = "settings must dump to a mapping"
        raise TypeError(msg)
    return {str(key): value for key, value in data.items()}


def _mask_url_passwords(value: object) -> object:
    match value:
        case dict():
            return {key: _mask_url_passwords(item) for key, item in value.items()}
        case list():
            return [_mask_url_passwords(item) for item in value]
        case str():
            return _mask_url_password(value)
        case _:
            return value


def _mask_url_password(text: str) -> str:
    try:
        parts = urlsplit(text)
        password = parts.password
    except ValueError:
        return text
    if password is None:
        return text
    user, _, host = parts.netloc.rpartition("@")
    name = user.partition(":")[0]
    return urlunsplit(parts._replace(netloc=f"{name}:{MASK}@{host}"))
