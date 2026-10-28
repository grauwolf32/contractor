"""Ordinary HTTP transport using the shared DNS-pin and peer-check backend."""

from __future__ import annotations

import httpcore
import httpx

from contractor_runtime.core.http_transport import PolicyNetworkBackend
from contractor_runtime.core.target_policy import TargetPolicy


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
