"""Address classification shared by direct HTTP and scanner destinations."""

from __future__ import annotations

import asyncio
import ipaddress

import pytest

from contractor_runtime.contracts import (
    CaidoSettings,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.toolsets.common.target_policy import (
    IPAddress,
    TargetDenied,
    TargetPolicy,
    TargetPolicyConfig,
    TargetUnresolved,
    literal_address,
    parse_private_networks,
)


def resolver(table: dict[str, tuple[str, ...]]):
    async def resolve(host: str, port: int) -> tuple[IPAddress, ...]:
        del port
        if host not in table:
            raise TargetUnresolved
        return tuple(ipaddress.ip_address(value) for value in table[host])

    return resolve


def settings(**overrides) -> RuntimeSettings:
    values = {
        "llmGatewayUrl": "https://gateway.internal:8443/v1",
        "artifactApiUrl": "https://artifacts.internal:9443/private/v1",
        "requestTimeoutSeconds": 30,
    }
    values.update(overrides)
    return RuntimeSettings(**values)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.1", "127.0.0.1"),
        ("2130706433", "127.0.0.1"),
        ("0x7f000001", "127.0.0.1"),
        ("0x7f.1", "127.0.0.1"),
        ("017700000001", "127.0.0.1"),
        ("0", "0.0.0.0"),
        ("127.0.0.1.", "127.0.0.1"),
        ("[::ffff:127.0.0.1]", "127.0.0.1"),
        ("::ffff:a9fe:a9fe", "169.254.169.254"),
        ("64:ff9b::a9fe:a9fe", "169.254.169.254"),
        ("fe80::1%eth0", "fe80::1"),
        ("example.com", None),
        ("app-1.example", None),
    ],
)
def test_literal_hosts_normalize_like_resolvers(host: str, expected: str | None) -> None:
    address = literal_address(host)
    assert (None if address is None else str(address)) == expected


@pytest.mark.parametrize("host", ["1.2.3.4.5", "256.1", "08", "host.0x", "1.2.3.0x1000000"])
def test_ambiguous_numeric_hosts_are_denied(host: str) -> None:
    with pytest.raises(TargetDenied):
        literal_address(host)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.0.0.2",
        "::1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "100.64.0.1",
        "169.254.10.10",
        "fe80::1",
        "fd12:3456::1",
        "::ffff:10.0.0.5",
    ],
)
def test_restricted_addresses_are_denied_by_default_and_allowed_by_network(address: str) -> None:
    selected = ipaddress.ip_address(address)
    with pytest.raises(TargetDenied):
        TargetPolicy().check_address(selected, 80)
    allowed = TargetPolicy(
        allowed_networks=parse_private_networks(
            [
                "127.0.0.0/8",
                "::1/128",
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16",
                "100.64.0.0/10",
                "169.254.0.0/16",
                "fe80::/10",
                "fc00::/7",
            ]
        )
    )
    allowed.check_address(selected, 80)


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "169.254.170.2",
        "100.100.100.200",
        "fd00:ec2::254",
        "64:ff9b::a9fe:a9fe",
        "0.0.0.0",
        "0.1.2.3",
        "::",
        "224.0.0.1",
        "ff02::1",
        "240.0.0.1",
        "255.255.255.255",
    ],
)
def test_metadata_unspecified_multicast_and_reserved_ignore_allowlists(address: str) -> None:
    policy = TargetPolicy(allowed_networks=parse_private_networks(["0.0.0.0/0", "::/0"]))
    with pytest.raises(TargetDenied):
        policy.check_address(ipaddress.ip_address(address), 80)


def test_global_addresses_and_names_pass_without_dns() -> None:
    policy = TargetPolicy()
    policy.check_address(ipaddress.ip_address("93.184.215.14"), 443)
    assert policy.check_host("app.example", 443) is None


@pytest.mark.parametrize(
    "host",
    ["metadata.google.internal", "METADATA.GOOGLE.INTERNAL.", "metadata", "instance-data"],
)
def test_metadata_names_are_denied_textually(host: str) -> None:
    with pytest.raises(TargetDenied):
        TargetPolicy(allowed_networks=parse_private_networks(["0.0.0.0/0"])).check_host(host, 80)


def test_runtime_endpoints_are_denied_by_name_and_resolved_address() -> None:
    async def scenario() -> None:
        config = TargetPolicyConfig(
            protected_urls=("https://control.internal:8443", "https://0.0.0.0:9443"),
            private_networks=parse_private_networks(["127.0.0.0/8", "10.0.0.0/8"]),
            resolver=resolver(
                {
                    "gateway.internal": ("10.0.0.7",),
                    "artifacts.internal": ("127.0.0.1",),
                    "control.internal": ("10.0.0.8",),
                    "caido.internal": ("10.0.0.9",),
                    "proxy.internal": ("10.0.0.10",),
                }
            ),
        )
        policy = await config.build(
            settings(
                httpProxy=HTTPProxySettings(
                    adapter="http-proxy@1",
                    proxyUrl="http://proxy.internal:3128",
                    targets=["llm-gateway"],
                ),
                caido=CaidoSettings(
                    adapter="caido-graphql@1",
                    endpoint="http://caido.internal:8080",
                    requestTimeoutSeconds=10,
                ),
            )
        )
        for host, port in (
            ("gateway.internal", 8443),
            ("artifacts.internal", 9443),
            ("control.internal", 8443),
            ("proxy.internal", 3128),
            ("caido.internal", 8080),
        ):
            with pytest.raises(TargetDenied):
                policy.check_host(host, port)
        for address, port in (
            ("10.0.0.7", 8443),
            ("10.0.0.8", 8443),
            ("10.0.0.9", 8080),
            ("::ffff:10.0.0.10", 3128),
            # Loopback endpoints protect every loopback alias on that port.
            ("127.0.0.1", 9443),
            ("127.0.0.2", 9443),
            ("::1", 9443),
            # The unspecified listen address protects loopback on its port.
            ("127.0.0.1", 9443),
        ):
            with pytest.raises(TargetDenied):
                policy.check_address(ipaddress.ip_address(address), port)
        for literal in ("127.1", "2130706433", "0x7f000001"):
            with pytest.raises(TargetDenied):
                policy.check_host(literal, 9443)
        # Other ports on the same allowed addresses remain reachable.
        policy.check_address(ipaddress.ip_address("10.0.0.7"), 8080)
        policy.check_address(ipaddress.ip_address("127.0.0.1"), 3000)

    asyncio.run(scenario())


def test_project_target_allows_only_its_resolved_private_origin() -> None:
    async def scenario() -> None:
        config = TargetPolicyConfig(
            resolver=resolver({"app.local": ("127.0.0.1",), "artifacts.internal": ("10.1.1.1",)})
        )
        policy = await config.build(
            settings(httpOriginTarget=HTTPOriginTargetSettings(url="http://app.local:3000/"))
        )
        policy.check_address(ipaddress.ip_address("127.0.0.1"), 3000)
        assert policy.check_host("127.1", 3000) == ipaddress.ip_address("127.0.0.1")
        for address, port in (("127.0.0.1", 3001), ("127.0.0.2", 3000), ("10.1.1.2", 3000)):
            with pytest.raises(TargetDenied):
                policy.check_address(ipaddress.ip_address(address), port)

        # A target on a Runtime endpoint does not unlock that endpoint.
        protected = await config.build(
            settings(
                artifactApiUrl="https://127.0.0.1:3000/private/v1",
                httpOriginTarget=HTTPOriginTargetSettings(url="http://127.0.0.1:3000/"),
            )
        )
        with pytest.raises(TargetDenied):
            protected.check_address(ipaddress.ip_address("127.0.0.1"), 3000)

    asyncio.run(scenario())


def test_resolve_filters_candidates_and_require_needs_every_address() -> None:
    async def scenario() -> None:
        policy = TargetPolicy(
            allowed_networks=parse_private_networks(["10.0.0.0/8"]),
            resolver=resolver(
                {
                    "mixed.example": ("127.0.0.1", "10.0.0.5"),
                    "loopback.example": ("127.0.0.1",),
                    "empty.example": (),
                }
            ),
        )
        assert await policy.resolve("mixed.example", 80) == (ipaddress.ip_address("10.0.0.5"),)
        with pytest.raises(TargetDenied):
            await policy.resolve("loopback.example", 80)
        with pytest.raises(TargetDenied):
            await policy.require("mixed.example", (80,))
        with pytest.raises(TargetUnresolved):
            await policy.require("missing.example", (80,))
        with pytest.raises(TargetUnresolved):
            await policy.resolve("empty.example", 80)
        await policy.require("10.0.0.5", (80, 443))
        with pytest.raises(TargetDenied):
            await policy.require("localhost", (80,))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "values",
    [["not-a-network"], ["10.0.0.1/8"], ["10.0.0.0/33"], [f"10.{i}.0.0/16" for i in range(65)]],
)
def test_invalid_private_networks_are_rejected(values: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_private_networks(values)


def test_mapped_private_networks_classify_as_ipv4() -> None:
    assert parse_private_networks(["::ffff:10.0.0.0/104", "10.0.0.0/8"]) == (
        ipaddress.ip_network("10.0.0.0/8"),
    )
