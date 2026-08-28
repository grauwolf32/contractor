import pytest

from contractor_runtime.settings import parse_settings


def test_settings_use_environment_and_cli_override() -> None:
    settings = parse_settings(
        ["--listen", "127.0.0.1:9001", "--log-level", "debug"],
        {"CONTRACTOR_RUNTIME_LISTEN": "127.0.0.1:9000"},
    )
    assert settings.listen_address == "127.0.0.1:9001"
    assert settings.log_level == "debug"


@pytest.mark.parametrize("listen", ["", "localhost", ":9000", "localhost:not-a-port", "x:70000"])
def test_invalid_listen_is_rejected(listen: str) -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--listen", listen], {})
