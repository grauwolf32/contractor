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


def _trusted_agent_card_values(
    *,
    allocation_id: str,
    logical_agent_name: str,
    description: str,
    version: str,
    endpoint: str,
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Exact Agent Card strings fixed by the protocol or the Server, by card path.

    Go's workerCardTrustedValues holds the same table, and
    api/testdata/v1alpha1/agent-card-secret-scan-cases.json pins both.
    """

    return {
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


def _untrusted_agent_card_text(
    card: Mapping[str, Any],
    *,
    allocation_id: str,
    logical_agent_name: str,
    description: str,
    version: str,
    endpoint: str,
) -> tuple[set[str], set[str]]:
    """Card values beyond exact protocol and Server-supplied text, and every key.

    A changed value at a normally fixed path is still scanned for private data.
    Object keys carry no path exemption; they are returned separately because
    nearly all of them are protocol vocabulary, which only a private value long
    enough to match anywhere is checked against.
    """

    trusted = _trusted_agent_card_values(
        allocation_id=allocation_id,
        logical_agent_name=logical_agent_name,
        description=description,
        version=version,
        endpoint=endpoint,
    )
    values: set[str] = set()
    keys: set[str] = set()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, str):
            if value not in trusted.get(path, ()):
                values.add(value)
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                keys.add(str(key))
                visit(nested, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))

    visit(card, ())
    return values, keys
