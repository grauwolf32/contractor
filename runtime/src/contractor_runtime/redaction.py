"""Runtime private values and the one policy for matching them in recorded text.

Metrics, allocation checks, Worker results, the summarizer and toolsets share
this module. It imports nothing from the toolsets, allocation or Worker
packages, so each of them can use it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable

from contractor_runtime.contracts import RuntimeSettings

# A private value this long is specific enough to be matched anywhere in
# model- or Worker-authored text; a shorter one (a proxy username, a short
# password) only as a complete string, so ordinary words cannot fail closed.
MIN_PRIVATE_SUBSTRING_BYTES = 16
# Retained in place of a redacted private value.
REDACTED = "[REDACTED]"


def runtime_secrets(settings: RuntimeSettings) -> tuple[str, ...]:
    """Only the RuntimeSettings credentials, without endpoints or CA bundles.

    Worker results may legitimately name an endpoint (a same-host deployment
    audits services next to its own), but never a credential.
    """

    values: list[str] = []
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if settings.telemetry is not None:
        values.extend(secret.get_secret_value() for secret in settings.telemetry.headers.values())
    if settings.http_proxy is not None:
        proxy = settings.http_proxy
        if proxy.basic_auth is not None:
            values.extend(
                (
                    proxy.basic_auth.username.get_secret_value(),
                    proxy.basic_auth.password.get_secret_value(),
                )
            )
        if proxy.bearer_token is not None:
            values.append(proxy.bearer_token.get_secret_value())
    if settings.caido is not None and settings.caido.bearer_token is not None:
        values.append(settings.caido.bearer_token.get_secret_value())
    if settings.http_origin_target is not None:
        # The target URL is the audited application and legitimately appears in
        # results; only the credentials Runtime injects for it are private.
        target = settings.http_origin_target
        if target.basic_auth is not None:
            values.extend(
                (
                    target.basic_auth.username.get_secret_value(),
                    target.basic_auth.password.get_secret_value(),
                )
            )
        if target.bearer_token is not None:
            values.append(target.bearer_token.get_secret_value())
    return tuple(value for value in values if value)


def runtime_setting_values(settings: RuntimeSettings) -> tuple[str, ...]:
    """Every private RuntimeSettings value: credentials, endpoints, CA bundles."""

    values = [settings.llm_gateway_url, settings.artifact_api_url]
    if settings.telemetry is not None:
        values.append(settings.telemetry.endpoint)
    if settings.http_proxy is not None:
        values.append(settings.http_proxy.proxy_url)
        if settings.http_proxy.ca_bundle_pem is not None:
            values.append(settings.http_proxy.ca_bundle_pem)
    if settings.caido is not None:
        values.append(settings.caido.endpoint)
        if settings.caido.ca_bundle_pem is not None:
            values.append(settings.caido.ca_bundle_pem)
    values.extend(runtime_secrets(settings))
    return tuple(value for value in values if value)


def substring_values(values: Iterable[str]) -> tuple[str, ...]:
    """The values long enough to match anywhere, longest first and unique.

    Replacing the longest first redacts a private value that contains a
    shorter one completely.
    """

    selected = {
        value for value in values if len(value.encode("utf-8")) >= MIN_PRIVATE_SUBSTRING_BYTES
    }
    return tuple(sorted(selected, key=lambda value: (-len(value.encode("utf-8")), value)))


def contains_private_value(text: str, values: Iterable[str]) -> bool:
    """Whether text equals a private value or contains a long one."""

    return any(
        value == text
        or (len(value.encode("utf-8")) >= MIN_PRIVATE_SUBSTRING_BYTES and value in text)
        for value in values
        if value
    )


def redact_private_values(text: str, values: Iterable[str]) -> str:
    """Return text with private values replaced under the shared policy.

    Text equal to a private value is replaced whole; otherwise only values of
    at least MIN_PRIVATE_SUBSTRING_BYTES are replaced where they occur, so a
    short credential never garbles ordinary words or identifiers.
    """

    selected = tuple(value for value in values if value)
    if text in selected:
        return REDACTED
    for value in substring_values(selected):
        text = text.replace(value, REDACTED)
    return text
