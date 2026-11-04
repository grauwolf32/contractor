"""Native message conversion: exact target components and opaque encoded bodies."""

import time
from urllib.parse import urlsplit

from mitmproxy import http
from mitmproxy.net.http import status_codes

from contractor_runtime.adapters.http_session import PrivateRequest, PrivateResponse

HOP_HEADERS = {
    b"connection",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


def end_to_end(headers):
    hop = set(HOP_HEADERS)
    for name, value in headers:
        if name.lower() == b"connection":
            hop.update(item.strip().lower() for item in value.split(b","))
    return [(k, v) for k, v in headers if k.lower() not in hop]


def request_target(url: str) -> bytes:
    """Extract the path/query of a URL already admitted by canonical_url."""
    parsed = urlsplit(url)
    return (parsed.path + ("?" + parsed.query if "?" in url else "")).encode("ascii")


def native_url(request: http.Request) -> str:
    """Render native components without losing an empty query or IPv6 brackets."""
    host = request.host
    if ":" in host:
        host = "[" + host + "]"
    default_port = 443 if request.scheme == "https" else 80
    authority = host + (f":{request.port}" if request.port != default_port else "")
    return request.scheme + "://" + authority + request.path


def request_message(url: str, request: PrivateRequest) -> http.Request:
    """Copy admitted wire headers/body without the native .content re-encoder."""
    parsed = urlsplit(url)
    return http.Request(
        host=parsed.hostname,
        port=parsed.port or (443 if parsed.scheme == "https" else 80),
        method=request.method.encode("ascii"),
        scheme=parsed.scheme.encode("ascii"),
        authority=b"",
        path=request_target(url),
        http_version=b"HTTP/1.1",
        headers=http.Headers(request.headers),
        content=request.body,
        trailers=None,
        timestamp_start=time.time(),
        timestamp_end=None,
    )


def response_message(response: PrivateResponse, method: str) -> http.Response:
    """Frame scanner IPC from raw target bytes; HEAD/304 retain length metadata."""
    headers = [(k, v) for k, v in end_to_end(response.headers) if k.lower() != b"content-length"]
    body = response.body
    if response.status == 204:
        body = b""
    elif method == "HEAD" or response.status == 304:
        body = b""
        headers.extend((k, v) for k, v in response.headers if k.lower() == b"content-length")
    else:
        headers.append((b"content-length", str(len(body)).encode()))
    headers.append((b"connection", b"close"))
    return http.Response(
        http_version=b"HTTP/1.1",
        status_code=response.status,
        reason=status_codes.RESPONSES.get(response.status, "").encode("ascii"),
        headers=http.Headers(headers),
        content=body,
        trailers=None,
        timestamp_start=time.time(),
        timestamp_end=None,
    )
