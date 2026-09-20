"""Validate one bounded, versioned request before handing it to sqlmap."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from contractor_runtime.toolsets.common.input_errors import ToolInputError

MAX_REQUEST_ARTIFACT_BYTES = 256 * 1024
MAX_BODY_BYTES = 64 * 1024
MAX_URL_BYTES = 8192
MAX_HEADERS = 64
MAX_HEADER_NAME_BYTES = 128
MAX_HEADER_VALUE_BYTES = 8192
MAX_HEADERS_BYTES = 32 * 1024
MAX_TEST_PARAMETERS = 64
MAX_TEST_PARAMETER_BYTES = 128

_FIELDS = {"schemaVersion", "method", "url", "headers", "body", "testParameters"}
_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", re.ASCII)
_PARAMETER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\[\]-]*", re.ASCII)
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", re.ASCII)
_UNSUPPORTED_HEADERS = {
    "connection",
    "content-encoding",
    "expect",
    "if-modified-since",
    "if-none-match",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class HTTPRequestInputError(ToolInputError):
    """Validation errors contain fixed diagnostics and never supplied request data."""


@dataclass(frozen=True, slots=True)
class PreparedHTTPRequest:
    raw: bytes
    method: str
    test_parameters: tuple[str, ...]


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise HTTPRequestInputError("request JSON contains duplicate object fields")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise HTTPRequestInputError("request JSON contains an unsupported constant")


def _ascii(value: object, limit: int, diagnostic: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not value and not empty):
        raise HTTPRequestInputError(diagnostic)
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        raise HTTPRequestInputError(diagnostic)
    return value


def _url(value: object) -> tuple[str, str]:
    url = _ascii(value, MAX_URL_BYTES, "request url must be a bounded HTTP(S) URL")
    if " " in url or "\\" in url or "#" in url or re.search(r"%(?![0-9A-Fa-f]{2})", url):
        raise HTTPRequestInputError("request url contains unsupported characters")
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        raise HTTPRequestInputError("request url has an invalid authority") from None
    if (
        parts.scheme not in {"http", "https"}
        or not url.startswith(f"{parts.scheme}://")
        or not hostname
        or "@" in parts.netloc
        or "%" in parts.netloc
        or parts.netloc.endswith(":")
        or port == 0
    ):
        raise HTTPRequestInputError("request url must use HTTP(S) without credentials")
    if parts.netloc.startswith("["):
        try:
            ipaddress.IPv6Address(hostname)
        except ValueError:
            raise HTTPRequestInputError("request url has an invalid host") from None
        if not re.fullmatch(r"\[[0-9A-Fa-f:.]+\](?::[0-9]+)?", parts.netloc):
            raise HTTPRequestInputError("request url has an invalid authority")
    elif len(hostname) > 253 or any(
        not _DNS_LABEL.fullmatch(label) for label in hostname.removesuffix(".").split(".")
    ):
        raise HTTPRequestInputError("request url has an invalid host")
    return url, parts.netloc


def _body(value: object) -> bytes:
    if not isinstance(value, str):
        raise HTTPRequestInputError("request body must be UTF-8 text")
    try:
        body = value.encode("utf-8")
    except UnicodeError:
        raise HTTPRequestInputError("request body must be UTF-8 text") from None
    if len(body) > MAX_BODY_BYTES:
        raise HTTPRequestInputError("request body exceeds its size limit")
    if any(ord(character) < 32 and character not in "\t\n" for character in value):
        raise HTTPRequestInputError("request body contains unsupported control characters")
    # sqlmap's request-file reader removes CR and trailing empty/whitespace lines.
    # Reject those representations instead of silently changing the supplied body.
    if value and (value.endswith("\n") or not value.rsplit("\n", 1)[-1].strip()):
        raise HTTPRequestInputError("request body has unsupported trailing whitespace lines")
    return body


def _content_type(value: str) -> None:
    charsets = [
        parameter.partition("=")[2].strip().strip('"').lower()
        for parameter in value.split(";")[1:]
        if parameter.partition("=")[0].strip().lower() == "charset"
    ]
    if len(charsets) > 1 or (charsets and charsets[0] not in {"utf-8", "utf8"}):
        raise HTTPRequestInputError("request Content-Type must use UTF-8 text")


def _headers(value: object, authority: str, body: bytes) -> bytes:
    if not isinstance(value, list) or len(value) > MAX_HEADERS:
        raise HTTPRequestInputError("request headers must be a bounded list")
    lines = []
    names = set()
    for header in value:
        if not isinstance(header, dict) or set(header) != {"name", "value"}:
            raise HTTPRequestInputError("request header must contain only name and value")
        name = _ascii(header["name"], MAX_HEADER_NAME_BYTES, "request header name is invalid")
        if not _HEADER_NAME.fullmatch(name):
            raise HTTPRequestInputError("request header name is invalid")
        content = _ascii(
            header["value"], MAX_HEADER_VALUE_BYTES, "request header value is invalid", empty=True
        )
        if content != content.strip():
            raise HTTPRequestInputError("request header value has unsupported whitespace")
        normalized = name.lower()
        if normalized in names:
            raise HTTPRequestInputError("request contains duplicate header names")
        names.add(normalized)
        if normalized in _UNSUPPORTED_HEADERS:
            raise HTTPRequestInputError("request uses unsupported encoding or framing headers")
        if "*" in name or "*" in (
            content.replace("*/*", "") if normalized == "accept" else content
        ):
            raise HTTPRequestInputError("request headers contain unsupported injection markers")
        if normalized == "host" and content.lower() != authority.lower():
            raise HTTPRequestInputError("request Host must match the URL authority and port")
        if normalized == "content-length" and content != str(len(body)):
            raise HTTPRequestInputError("request Content-Length must match its UTF-8 body size")
        if normalized == "content-type":
            _content_type(content)
        lines.append(f"{name}: {content}\r\n")
    if "host" not in names:
        lines.insert(0, f"Host: {authority}\r\n")
    # Even an empty GET needs Content-Length so sqlmap retains requests whose
    # selected test parameter is only in a header, with no query or cookie.
    if "content-length" not in names:
        lines.append(f"Content-Length: {len(body)}\r\n")
    serialized = "".join(lines).encode("ascii")
    if len(serialized) > MAX_HEADERS_BYTES:
        raise HTTPRequestInputError("request headers exceed their total size limit")
    return serialized


def _parameters(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_TEST_PARAMETERS:
        raise HTTPRequestInputError("request testParameters must be a nonempty bounded list")
    names = []
    for item in value:
        name = _ascii(
            item, MAX_TEST_PARAMETER_BYTES, "request testParameters contains an invalid name"
        )
        if not _PARAMETER.fullmatch(name) or name in names:
            raise HTTPRequestInputError("request testParameters contains an invalid name")
        names.append(name)
    return tuple(names)


def parse_http_request(data: bytes) -> PreparedHTTPRequest:
    """Return a private raw-request representation, with no permissive coercions."""
    if not isinstance(data, bytes) or len(data) > MAX_REQUEST_ARTIFACT_BYTES:
        raise HTTPRequestInputError("request artifact exceeds its size limit or is not bytes")
    try:
        request = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
    except HTTPRequestInputError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPRequestInputError("request artifact must contain valid UTF-8 JSON") from None
    if not isinstance(request, dict) or set(request) != _FIELDS:
        raise HTTPRequestInputError("request artifact contains missing or unsupported fields")
    if type(request["schemaVersion"]) is not int or request["schemaVersion"] != 1:
        raise HTTPRequestInputError("request schemaVersion must be integer 1")
    method = request["method"]
    if not isinstance(method, str) or method not in _METHODS:
        raise HTTPRequestInputError("request method is unsupported")
    url, authority = _url(request["url"])
    body = _body(request["body"])
    if "*" in url or b"*" in body:
        raise HTTPRequestInputError("request contains unsupported injection markers")
    headers = _headers(request["headers"], authority, body)
    parameters = _parameters(request["testParameters"])
    # Absolute-form retains scheme and explicit port when sqlmap reads the file.
    raw = f"{method} {url} HTTP/1.1\r\n".encode("ascii") + headers + b"\r\n" + body
    if (
        b"==========" in raw
        or b"### Conversation" in raw
        or b"<request base64=" in raw.lower()
        or re.search(rb"%INJECT[_ ]?HERE%", raw, re.IGNORECASE)
    ):
        raise HTTPRequestInputError("request contains unsupported scanner request-file markers")
    return PreparedHTTPRequest(raw=raw, method=method, test_parameters=parameters)
