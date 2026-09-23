"""Direct HTTP transport that applies the target policy to each TCP connection."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from typing import Any

import httpcore
import httpx

from contractor_runtime.toolsets.common.target_policy import (
    TargetDenied,
    TargetPolicy,
    TargetUnresolved,
    canonical_address,
)


class PolicyNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve once, check every candidate, and connect only to a checked address.

    httpcore passes the URL host here and later uses that same host for TLS SNI
    and certificate verification, so pinning the socket to the checked address
    does not weaken HTTPS. There is no second implicit DNS lookup to rebind.
    """

    def __init__(self, policy: TargetPolicy, backend: httpcore.AsyncNetworkBackend | None = None):
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
                _verify_peer(stream, address, port)
            except BaseException:
                await stream.aclose()
                raise
            return stream
        assert failure is not None
        raise failure

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise TargetDenied

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class PolicyHTTPTransport(httpx.AsyncHTTPTransport):
    """httpx's default transport with the policy backend in its connection pool."""

    def __init__(self, policy: TargetPolicy, *, limits: httpx.Limits) -> None:
        super().__init__(trust_env=False, limits=limits, retries=0)
        # httpx has no public network-backend option. Fail at construction,
        # never at request time, if its private pool attribute ever changes.
        if not isinstance(getattr(self, "_pool", None), httpcore.AsyncConnectionPool):
            raise RuntimeError("httpx transport layout is unsupported")
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(trust_env=False),
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            retries=0,
            network_backend=PolicyNetworkBackend(policy),
        )


def _verify_peer(stream: httpcore.AsyncNetworkStream, address: Any, port: int) -> None:
    peer = stream.get_extra_info("server_addr")
    try:
        peer_address = canonical_address(ipaddress.ip_address(str(peer[0])))
        peer_port = int(peer[1])
    except (TypeError, ValueError, IndexError):
        raise TargetDenied from None
    if peer_address != canonical_address(address) or peer_port != port:
        raise TargetDenied
