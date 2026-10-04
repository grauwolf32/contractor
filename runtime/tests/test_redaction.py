from __future__ import annotations

from contractor_runtime.contracts import (
    CaidoSettings,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.redaction import (
    REDACTED,
    contains_private_value,
    redact_private_values,
    runtime_secrets,
    runtime_setting_values,
    substring_values,
)

LONG = "private-canary-value-0f3a9"
LONGER = f"{LONG}-extended"


def test_short_values_match_whole_text_and_long_values_match_anywhere() -> None:
    values = ("a", "1", LONG, LONGER, "")
    assert redact_private_values("a", values) == REDACTED
    assert redact_private_values("alpha 1", values) == "alpha 1"
    # The longest value is replaced first, so no fragment of it survives.
    assert redact_private_values(f"x {LONGER} y {LONG}", values) == f"x {REDACTED} y {REDACTED}"
    assert substring_values(values) == (LONGER, LONG)
    assert contains_private_value("1", values)
    assert not contains_private_value("10", values)
    assert contains_private_value(f"..{LONG}..", values)
    assert not contains_private_value("", ("",))


def test_runtime_secrets_hold_credentials_and_setting_values_add_endpoints() -> None:
    settings = RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        httpProxy=HTTPProxySettings(
            adapter="http-proxy@1",
            proxyUrl="https://proxy.example",
            basicAuth={"username": "ops", "password": "proxy-password"},
            targets=["tool-http"],
        ),
        caido=CaidoSettings(
            adapter="caido-graphql@1",
            endpoint="https://caido.example",
            bearerToken="caido-token",
            requestTimeoutSeconds=5,
        ),
        httpOriginTarget=HTTPOriginTargetSettings(
            url="https://target.example", bearerToken="target-token"
        ),
        requestTimeoutSeconds=10,
    )
    assert runtime_secrets(settings) == (
        "gateway-token",
        "ops",
        "proxy-password",
        "caido-token",
        "target-token",
    )
    assert runtime_setting_values(settings) == (
        "https://gateway.example/v1",
        "https://control.example/private/v1",
        "https://proxy.example",
        "https://caido.example",
        *runtime_secrets(settings),
    )
