from langcheck.libreoffice.client import Connection
from langcheck.libreoffice.settings import KEY_VARIABLE, URL_VARIABLE, connection_from


def test_the_environment_wins_over_the_configuration() -> None:
    environ = {URL_VARIABLE: " http://env.test ", KEY_VARIABLE: "lc_env"}

    assert connection_from(environ, configured_url="http://cfg.test", configured_key="lc_cfg") == (
        Connection("http://env.test", "lc_env")
    )


def test_falls_back_to_the_configuration_or_nothing() -> None:
    assert connection_from({}, configured_url="http://cfg.test", configured_key="k") == (
        Connection("http://cfg.test", "k")
    )
    assert connection_from({}, configured_url=" ", configured_key="k") is None
