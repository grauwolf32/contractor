"""URL hosts and Agent Card text used to check allocation observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit


def _url_hosts(urls: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for value in urls:
        parsed = urlsplit(value)
        if parsed.hostname is not None:
            values.add(parsed.hostname)
        if parsed.netloc:
            values.add(parsed.netloc)
    return tuple(sorted(values))


def _untrusted_agent_card_strings(
    card: Mapping[str, Any],
    *,
    allocation_id: str,
    logical_agent_name: str,
    description: str,
    version: str,
    endpoint: str,
) -> set[str]:
    """Card text beyond exact protocol and Server-supplied values.

    Keep this path policy aligned with Go's trustedWorkerCardString. A changed
    value at a normally fixed path is still scanned for private data.
    """
    trusted: dict[tuple[str, ...], tuple[str, ...]] = {
        ("name",): (logical_agent_name, f"Contractor Worker {logical_agent_name}"),
        ("description",): (description,),
        ("version",): (version,),
        ("url",): (endpoint,),
        ("protocolVersion",): ("1.0",),
        ("supportedInterfaces", "0", "url"): (endpoint,),
        ("supportedInterfaces", "0", "protocolBinding"): ("JSONRPC",),
        ("supportedInterfaces", "0", "protocolVersion"): ("1.0",),
        ("supportedInterfaces", "0", "tenant"): (allocation_id,),
        ("defaultInputModes", "0"): (
            "application/vnd.contractor.stage-content+json",
            "application/json",
        ),
        ("defaultOutputModes", "0"): (
            "application/vnd.contractor.worker-completion+json",
            "application/json",
        ),
        ("skills", "0", "id"): ("contractor_stage_content",),
        ("skills", "0", "name"): ("Execute Contractor stage content",),
        ("skills", "0", "description"): ("Execute one strict Contractor StageContentRequest.",),
        ("skills", "0", "tags", "0"): ("contractor",),
        ("skills", "0", "tags", "1"): ("stage",),
        ("skills", "0", "inputModes", "0"): ("application/vnd.contractor.stage-content+json",),
        ("skills", "0", "outputModes", "0"): ("application/vnd.contractor.worker-completion+json",),
        ("securitySchemes", "mutualTLS", "mtlsSecurityScheme", "description"): (
            "Deployment-CA mutual TLS with a Contractor Control Plane peer",
        ),
    }
    result: set[str] = set()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, str):
            if value not in trusted.get(path, ()):
                result.add(value)
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, (*path, key))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))

    visit(card, ())
    return result
