"""Destination policy shared by model-selected HTTP requests and scanner launches.

The policy classifies resolved IP addresses, never URL text alone. Literal hosts
are normalized the way resolvers interpret them (shortened, decimal, octal and
hexadecimal IPv4; IPv4-mapped and NAT64 IPv6) before classification:

- Runtime service endpoints (by host name and by resolved address plus port)
  and cloud metadata, unspecified, multicast and reserved destinations are
  always denied.
- Loopback, link-local, private and other non-global addresses are denied
  unless the allocation's project HTTP target resolves to that exact address
  and port, or an operator-configured network contains the address.
- Global addresses are allowed.

Direct HTTP transports call :meth:`TargetPolicy.resolve` at connect time and
connect only to an address it returned. Scanner subprocesses resolve names
themselves, so :meth:`TargetPolicy.require` can only check before launch.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from contractor_runtime.contracts import RuntimeSettings

type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
type IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
type Resolver = Callable[[str, int], Awaitable[Sequence[IPAddress]]]

MAX_RESOLVED_ADDRESSES = 32
MAX_PRIVATE_TARGET_NETWORKS = 64
PROTECTED_RESOLUTION_TIMEOUT_SECONDS = 5.0

# Instance metadata services reachable from common cloud and container hosts.
METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(value)
    for value in (
        "169.254.169.254",
        "169.254.170.2",
        "169.254.170.23",
        "100.100.100.200",
        "fd00:ec2::254",
        "fd00:ec2::23",
    )
)
METADATA_HOSTNAMES = frozenset(
    {
        "instance-data",
        "instance-data.ec2.internal",
        "metadata",
        "metadata.goog",
        "metadata.google.internal",
    }
)
_THIS_NETWORK = ipaddress.ip_network("0.0.0.0/8")
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_LOOPBACK_ADDRESSES = (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1"))
_IPV4_PART = re.compile(r"0[xX][0-9a-fA-F]*|0[0-7]*|[1-9][0-9]*")
_NUMERIC_LABEL = re.compile(r"[0-9]+|0[xX][0-9a-fA-F]*")


class TargetDenied(RuntimeError):
    """The destination is outside the allocation's target policy."""

    def __init__(self) -> None:
        super().__init__("target destination denied by policy")


class TargetUnresolved(RuntimeError):
    """The destination host did not resolve to any address."""

    def __init__(self) -> None:
        super().__init__("target destination did not resolve")


def canonical_address(address: IPAddress) -> IPAddress:
    """Return the IPv4 address an IPv6 translation form reaches, without a zone."""

    if address.version == 6:
        assert isinstance(address, ipaddress.IPv6Address)
        if address.scope_id is not None:
            address = ipaddress.IPv6Address(address.compressed.split("%", 1)[0])
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
        if address in _NAT64_PREFIX:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def literal_address(host: str) -> IPAddress | None:
    """Parse a URL host as an address, including legacy IPv4 notations.

    Raises TargetDenied for a host whose final label is numeric but which is not
    a valid IPv4 address: resolvers and proxies disagree about such names.
    """

    value = host.strip("[]")
    try:
        return canonical_address(ipaddress.ip_address(value))
    except ValueError:
        pass
    labels = value.removesuffix(".").split(".")
    if not labels or _NUMERIC_LABEL.fullmatch(labels[-1]) is None:
        return None
    if len(labels) > 4 or not all(_IPV4_PART.fullmatch(label) for label in labels):
        raise TargetDenied
    numbers = [_ipv4_part(label) for label in labels]
    if any(number > 255 for number in numbers[:-1]) or numbers[-1] >= 256 ** (5 - len(labels)):
        raise TargetDenied
    result = numbers[-1]
    for index, number in enumerate(numbers[:-1]):
        result += number << (8 * (3 - index))
    return ipaddress.IPv4Address(result)


def _ipv4_part(label: str) -> int:
    if label[:2].lower() == "0x":
        return int(label[2:] or "0", 16)
    if len(label) > 1 and label[0] == "0":
        return int(label, 8)
    return int(label)


def normalized_host(host: str) -> str:
    return host.strip("[]").rstrip(".").lower()


def url_endpoint(url: str) -> tuple[str, int]:
    """Return the normalized host and effective port of an absolute HTTP(S) URL."""

    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = normalized_host(parsed.hostname or "")
    if scheme not in {"http", "https"} or not host:
        raise ValueError("URL must be absolute HTTP(S)")
    port = parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
    return host, port


def parse_private_networks(values: Iterable[str]) -> tuple[IPNetwork, ...]:
    """Validate operator-allowed networks; raises ValueError without echoing input."""

    selected = tuple(values)
    if len(selected) > MAX_PRIVATE_TARGET_NETWORKS:
        raise ValueError(f"at most {MAX_PRIVATE_TARGET_NETWORKS} private target networks")
    result: list[IPNetwork] = []
    for value in selected:
        try:
            network = ipaddress.ip_network(value.strip(), strict=True)
        except ValueError:
            raise ValueError("private target networks must be CIDR networks") from None
        if network.version == 6:
            assert isinstance(network, ipaddress.IPv6Network)
            first = canonical_address(network.network_address)
            if first.version == 4 and network.prefixlen >= 96:
                # Classification uses the IPv4 form of mapped/NAT64 addresses.
                network = ipaddress.ip_network(f"{first}/{network.prefixlen - 96}")
        if network not in result:
            result.append(network)
    return tuple(result)


async def system_resolver(host: str, port: int) -> tuple[IPAddress, ...]:
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise TargetUnresolved from None
    result: list[IPAddress] = []
    for record in records:
        try:
            address = ipaddress.ip_address(str(record[4][0]))
        except ValueError:
            continue
        if address not in result:
            result.append(address)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TargetPolicyConfig:
    """Process-wide policy inputs supplied by the Runtime Agent entry point."""

    protected_urls: tuple[str, ...] = ()
    private_networks: tuple[IPNetwork, ...] = ()
    resolver: Resolver | None = field(default=None, repr=False)

    async def build(self, settings: RuntimeSettings) -> TargetPolicy:
        """Resolve allocation service endpoints and the project target once."""

        resolver = self.resolver or system_resolver
        timeout = min(PROTECTED_RESOLUTION_TIMEOUT_SECONDS, float(settings.request_timeout_seconds))
        names: set[tuple[str, int]] = set()
        for url in (*self.protected_urls, *runtime_service_urls(settings)):
            try:
                names.add(url_endpoint(url))
            except ValueError:
                continue
        target: tuple[str, int] | None = None
        if settings.http_origin_target is not None:
            try:
                target = url_endpoint(settings.http_origin_target.url)
            except ValueError:
                target = None
        endpoints = sorted(names)
        # Lookups run concurrently so a slow resolver costs one bounded wait.
        resolved = await asyncio.gather(
            *(
                _resolve_quietly(resolver, host, port, timeout)
                for host, port in (*endpoints, *([target] if target is not None else []))
            )
        )
        addresses: set[tuple[IPAddress, int]] = set()
        local_ports: set[int] = set()
        for (_, port), candidates in zip(endpoints, resolved, strict=False):
            for address in candidates:
                addresses.add((address, port))
                if address.is_loopback or address.is_unspecified:
                    local_ports.add(port)
        allowed_origins: set[tuple[IPAddress, int]] = set()
        if target is not None:
            allowed_origins.update(
                (address, target[1]) for address in resolved[-1] if not address.is_global
            )
        return TargetPolicy(
            protected_names=frozenset(names),
            protected_addresses=frozenset(addresses),
            protected_local_ports=frozenset(local_ports),
            allowed_origins=frozenset(allowed_origins),
            allowed_networks=self.private_networks,
            resolver=resolver,
        )


@dataclass(frozen=True, slots=True)
class TargetPolicy:
    protected_names: frozenset[tuple[str, int]] = frozenset()
    protected_addresses: frozenset[tuple[IPAddress, int]] = frozenset()
    protected_local_ports: frozenset[int] = frozenset()
    allowed_origins: frozenset[tuple[IPAddress, int]] = frozenset()
    allowed_networks: tuple[IPNetwork, ...] = ()
    resolver: Resolver = field(default=system_resolver, repr=False)

    def check_address(self, address: IPAddress, port: int) -> None:
        selected = canonical_address(address)
        if (
            (selected, port) in self.protected_addresses
            or (selected.is_loopback and port in self.protected_local_ports)
            or selected in METADATA_ADDRESSES
            or selected.is_unspecified
            or selected.is_multicast
            # IPv6 ::/8 is reserved but contains ::1, which stays allowable.
            or (selected.is_reserved and not selected.is_loopback)
            or (selected.version == 4 and selected in _THIS_NETWORK)
        ):
            raise TargetDenied
        if selected.is_global:
            return
        if (selected, port) in self.allowed_origins or any(
            selected in network for network in self.allowed_networks
        ):
            return
        raise TargetDenied

    def check_host(self, host: str, port: int) -> IPAddress | None:
        """Check a host without DNS; returns its address when it is a literal."""

        name = normalized_host(host)
        if not name or (name, port) in self.protected_names or name in METADATA_HOSTNAMES:
            raise TargetDenied
        address = literal_address(name)
        if address is not None:
            self.check_address(address, port)
            return address
        if name == "localhost" or name.endswith(".localhost"):
            # Resolver-independent: these names always mean this host.
            for loopback in _LOOPBACK_ADDRESSES:
                try:
                    self.check_address(loopback, port)
                    return None
                except TargetDenied:
                    continue
            raise TargetDenied
        return None

    def check_url(self, url: str) -> None:
        try:
            host, port = url_endpoint(url)
        except ValueError:
            raise TargetDenied from None
        self.check_host(host, port)

    async def resolve(self, host: str, port: int) -> tuple[IPAddress, ...]:
        """Return the permitted addresses for one connection, in resolver order."""

        literal = self.check_host(host, port)
        candidates = (literal,) if literal is not None else await self._lookup(host, port)
        permitted: list[IPAddress] = []
        for address in candidates:
            try:
                self.check_address(address, port)
            except TargetDenied:
                continue
            permitted.append(address)
        if not permitted:
            raise TargetDenied
        return tuple(permitted)

    async def require(self, host: str, ports: Iterable[int]) -> None:
        """Require every resolved address to be permitted for every port."""

        selected = tuple(ports)
        if not selected:
            raise TargetDenied
        for port in selected:
            self.check_host(host, port)
        literal = literal_address(normalized_host(host))
        candidates = (literal,) if literal is not None else await self._lookup(host, selected[0])
        for address in candidates:
            for port in selected:
                self.check_address(address, port)

    async def _lookup(self, host: str, port: int) -> tuple[IPAddress, ...]:
        addresses = tuple(await self.resolver(host, port))
        if not addresses:
            raise TargetUnresolved
        if len(addresses) > MAX_RESOLVED_ADDRESSES:
            raise TargetDenied
        return addresses


def runtime_service_urls(settings: RuntimeSettings) -> tuple[str, ...]:
    """Allocation-scoped Runtime infrastructure that model targets never reach."""

    values = [settings.artifact_api_url]
    if settings.llm_gateway_url is not None:
        values.append(settings.llm_gateway_url)
    if settings.telemetry is not None:
        values.append(settings.telemetry.endpoint)
    if settings.http_proxy is not None:
        values.append(settings.http_proxy.proxy_url)
    if settings.caido is not None:
        values.append(settings.caido.endpoint)
    return tuple(values)


async def _resolve_quietly(
    resolver: Resolver, host: str, port: int, timeout: float
) -> tuple[IPAddress, ...]:
    literal: IPAddress | None
    try:
        literal = literal_address(host)
    except TargetDenied:
        return ()
    if literal is not None:
        return (literal,)
    try:
        async with asyncio.timeout(timeout):
            resolved = await resolver(host, port)
    except (TargetUnresolved, TimeoutError, OSError):
        return ()
    return tuple(canonical_address(address) for address in resolved)[:MAX_RESOLVED_ADDRESSES]
