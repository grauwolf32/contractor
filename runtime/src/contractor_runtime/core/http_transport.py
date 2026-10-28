"""DNS-pinned HTTP transport primitives shared by controlled target clients."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from typing import Any, Protocol

import httpcore

from contractor_runtime.core.target_policy import (
    IPAddress,
    TargetDenied,
    TargetUnresolved,
    canonical_address,
)


class ConnectionPolicy(Protocol):
    async def resolve(self, host: str, port: int) -> tuple[IPAddress, ...]: ...


class PolicyNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect to a validated address without repeating the host DNS lookup."""

    def __init__(
        self, policy: ConnectionPolicy, backend: httpcore.AsyncNetworkBackend | None = None
    ) -> None:
        self._policy = policy
        self._backend = backend or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = await self._policy.resolve(host, port)
        except TargetUnresolved:
            raise httpcore.ConnectError("target did not resolve") from None
        failure: Exception | None = None
        for address in addresses:
            try:
                stream = await self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as error:
                failure = error
                continue
            try:
                verify_peer(stream, address, port)
            except BaseException:
                await stream.aclose()
                raise
            return stream
        if failure is None:
            raise TargetUnresolved
        raise failure

    async def connect_unix_socket(self, path: str, timeout=None, socket_options=None):
        raise TargetDenied

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def verify_peer(stream: httpcore.AsyncNetworkStream, address: IPAddress, port: int) -> None:
    peer = stream.get_extra_info("server_addr")
    try:
        peer_address = canonical_address(ipaddress.ip_address(str(peer[0])))
        peer_port = int(peer[1])
    except (TypeError, ValueError, IndexError):
        raise TargetDenied from None
    if peer_address != canonical_address(address) or peer_port != port:
        raise TargetDenied
