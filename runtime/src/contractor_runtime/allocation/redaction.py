"""Values and URL hosts used to redact allocation observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts import RuntimeSettings

# A private value this long is specific enough to be matched anywhere in
# model- or Worker-authored text; a shorter one (a proxy username, a short
# password) only as a complete string, so ordinary words cannot fail closed.
MIN_PRIVATE_SUBSTRING_BYTES = 16


def _runtime_setting_values(settings: RuntimeSettings) -> tuple[str, ...]:
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
    values.extend(_runtime_secret_values(settings))
    return tuple(value for value in values if value)


def _runtime_secret_values(settings: RuntimeSettings) -> tuple[str, ...]:
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


def _substring_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        value for value in values if len(value.encode("utf-8")) >= MIN_PRIVATE_SUBSTRING_BYTES
    )


def _contains_private_value(text: str, values: Iterable[str]) -> bool:
    """Whether text exposes a private value under the shared matching policy."""

    return any(
        value == text
        or (len(value.encode("utf-8")) >= MIN_PRIVATE_SUBSTRING_BYTES and value in text)
        for value in values
    )


def _url_hosts(urls: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for value in urls:
        parsed = urlsplit(value)
        if parsed.hostname is not None:
            values.add(parsed.hostname)
        if parsed.netloc:
            values.add(parsed.netloc)
    return tuple(sorted(values))


def _nested_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {item for nested in value.values() for item in _nested_strings(nested)}
    if isinstance(value, (list, tuple)):
        return {item for nested in value for item in _nested_strings(nested)}
    return set()
