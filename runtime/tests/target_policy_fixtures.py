"""Deterministic target policy for scanner tests that never contact real hosts."""

from __future__ import annotations

import hashlib
import ipaddress

from contractor_runtime.toolsets.common.target_policy import (
    IPAddress,
    TargetPolicyConfig,
    TargetUnresolved,
    parse_allowed_networks,
)

# RFC 2544 benchmarking space: never routed, allowed by the default policy, and
# large enough that distinct fixture names (targets versus Runtime service
# endpoints) do not collide.
FIXTURE_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def fixture_address(host: str) -> IPAddress:
    digest = int.from_bytes(hashlib.sha256(host.encode()).digest()[:4], "big")
    return FIXTURE_NETWORK.network_address + digest % FIXTURE_NETWORK.num_addresses


async def fixture_resolver(host: str, port: int) -> tuple[IPAddress, ...]:
    """Resolve names without DNS; names ending in `.unresolvable` fail."""

    del port
    if host.endswith(".unresolvable"):
        raise TargetUnresolved
    return (fixture_address(host),)


SCAN_TEST_POLICY = TargetPolicyConfig(
    allowed_networks=parse_allowed_networks(["127.0.0.0/8", "::1/128"]),
    resolver=fixture_resolver,
)
