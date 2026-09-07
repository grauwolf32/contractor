"""Values and URL hosts used to redact allocation observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts import RuntimeSettings


def _runtime_setting_values(settings: RuntimeSettings) -> tuple[str, ...]:
    values = [settings.llm_gateway_url, settings.artifact_api_url]
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if settings.telemetry is not None:
        values.append(settings.telemetry.endpoint)
        values.extend(secret.get_secret_value() for secret in settings.telemetry.headers.values())
    if settings.http_proxy is not None:
        proxy = settings.http_proxy
        values.append(proxy.proxy_url)
        if proxy.basic_auth is not None:
            values.extend(
                (
                    proxy.basic_auth.username.get_secret_value(),
                    proxy.basic_auth.password.get_secret_value(),
                )
            )
        if proxy.bearer_token is not None:
            values.append(proxy.bearer_token.get_secret_value())
        if proxy.ca_bundle_pem is not None:
            values.append(proxy.ca_bundle_pem)
    if settings.caido is not None:
        caido = settings.caido
        values.append(caido.endpoint)
        if caido.bearer_token is not None:
            values.append(caido.bearer_token.get_secret_value())
        if caido.ca_bundle_pem is not None:
            values.append(caido.ca_bundle_pem)
    return tuple(value for value in values if value)


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
