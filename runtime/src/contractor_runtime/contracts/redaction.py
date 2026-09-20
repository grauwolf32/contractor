"""Private RuntimeSettings values shared by allocation and telemetry redaction."""

from __future__ import annotations

from contractor_runtime.contracts.settings import RuntimeSettings


def runtime_setting_values(settings: RuntimeSettings) -> tuple[str, ...]:
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
    for toolset in settings.toolsets.values():
        values.append(toolset.endpoint)
        values.extend(secret.get_secret_value() for secret in toolset.headers.values())
        if toolset.ca_bundle_pem is not None:
            values.append(toolset.ca_bundle_pem)
    if settings.http_origin_target is not None:
        target = settings.http_origin_target
        values.append(target.url)
        if target.basic_auth is not None:
            values.extend(
                (
                    target.basic_auth.username.get_secret_value(),
                    target.basic_auth.password.get_secret_value(),
                )
            )
        if target.bearer_token is not None:
            values.append(target.bearer_token.get_secret_value())
    return tuple(dict.fromkeys(value for value in values if value))
