"""Domain-test engine double; protocol conformance uses real native sockets."""

from urllib.parse import urlsplit

import httpcore

from contractor_runtime.adapters.http_engine_state import EngineFailure
from contractor_runtime.adapters.http_session import PrivateResponse
from contractor_runtime.adapters.http_transport import HTTPTransport
from fakes.controlled_stream import ControlledBackend


def transport(spec, *, backend=None, **options):
    resource = HTTPTransport(spec, **options)
    if backend is not None:
        resource._engine = Engine(backend, resource._ssl)
    return resource


class Engine:
    def __init__(self, network, ssl_context):
        self.network, self.ssl = network, ssl_context
        self.boundaries = set()

    async def request(self, url, boundary, request, policy, identity, body_limit):
        self.boundaries.add(boundary)
        try:
            async with httpcore.AsyncConnectionPool(
                ssl_context=self.ssl,
                max_connections=1,
                max_keepalive_connections=0,
                retries=0,
                network_backend=ControlledBackend(policy, boundary, self.network),
            ) as pool:
                parsed = urlsplit(url)
                target = parsed.path + ("?" + parsed.query if "?" in url else "")
                wire_url = httpcore.URL(
                    scheme=parsed.scheme, host=parsed.hostname, port=parsed.port, target=target
                )
                async with pool.stream(
                    request.method, wire_url, headers=request.headers, content=request.body
                ) as response:
                    body = bytearray()
                    truncated = False
                    async for chunk in response.aiter_stream():
                        remaining = body_limit - len(body)
                        body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            truncated = True
                            break
                    return PrivateResponse(
                        url, response.status, response.headers, bytes(body), truncated
                    )
        except httpcore.ConnectError as error:
            raise EngineFailure("tls_failed" if error.__cause__ else "connect_failed") from None
        except httpcore.RemoteProtocolError:
            raise EngineFailure("response_invalid") from None
        except (httpcore.ReadError, httpcore.WriteError):
            raise EngineFailure("transport_failed") from None
        finally:
            stream = getattr(boundary, "transport", None)
            if stream is None or stream.closed:
                self.boundaries.discard(boundary)

    async def close(self):
        for boundary in tuple(self.boundaries):
            if boundary.transport is not None:
                await boundary.transport.aclose()
            self.boundaries.discard(boundary)
