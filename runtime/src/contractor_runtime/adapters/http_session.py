"""Allocation-only private session state; nothing is serialized or exposed as a tool."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from contractor_runtime.contracts.engagement import MAX_URL_BYTES, Identity
from contractor_runtime.contracts.fields import (
    is_field_value,
    valid_cookie_name,
    valid_cookie_value,
)
from contractor_runtime.contracts.settings import HTTPIdentityMaterial
from contractor_runtime.core.http_limits import MAX_CONTROLLED_HEADER_BYTES
from contractor_runtime.core.http_session import (
    MAX_COOKIE_SESSION_BYTES,
    MAX_COOKIES,
    basic_authorization,
    discard_expired_cookies,
    encode_cookies,
    matching_cookies,
)


def origin_of(url: str) -> str:
    parsed = urlsplit(url)
    return parsed.scheme + "://" + parsed.netloc


@dataclass(slots=True, repr=False)
class PrivateResponse:
    url: str
    status: int
    headers: list[tuple[bytes, bytes]]
    body: bytes = b""
    truncated: bool = False
    session_valid: bool = True
    method: str = "GET"
    redirects: int = 0
    retries: int = 0


@dataclass(slots=True, repr=False)
class PrivateRequest:
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    method: str = "GET"
    body: bytes = b""
    follow_redirects: bool = True
    url: str = ""


class IdentitySession:
    """One owner, one identity and one generation; cookies are exact-origin local."""

    def __init__(
        self,
        identity: Identity,
        material: HTTPIdentityMaterial | None,
    ):
        self.identity, self.material = identity, material
        self.epoch = object()
        self.state = "anonymous" if identity.kind == "anonymous" else "credential_configured"
        self.cookies: dict[str, httpx.Cookies] = {}
        supplied = {}
        if material is not None:
            supplied = (
                material.fields
                if identity.kind == "cookies"
                else {
                    name.removeprefix("cookie:"): value
                    for name, value in material.fields.items()
                    if identity.kind == "api-keys" and name.startswith("cookie:")
                }
            )
        self.supplied_cookies = bool(supplied)
        if supplied:
            for origin in identity.origins:
                jar = httpx.Cookies()
                host = urlsplit(origin).hostname
                if ":" in host:
                    # CookieJar uses the bracketed effective host for IPv6.
                    host = f"[{host}].local"
                for name, value in supplied.items():
                    jar.set(name, value.get_secret_value(), domain=host, path="/")
                for cookie in jar.jar:
                    cookie.domain_specified = False
                self.cookies[origin] = jar
            try:
                self._validate_cookies(self.cookies)
            except ValueError:
                self.close()
                raise

    def require_epoch(self, epoch: object) -> None:
        if epoch is not self.epoch:
            raise ValueError("engagement session generation is stale")

    def clear(self) -> None:
        for cookies in self.cookies.values():
            cookies.clear()
        self.cookies.clear()
        self.epoch = object()

    def cookie_summary(self) -> dict:
        for jar in self.cookies.values():
            discard_expired_cookies(jar, time.time())
        cookies = [cookie for jar in self.cookies.values() for cookie in jar.jar]
        return {
            "cookie_names": sorted({cookie.name for cookie in cookies}),
            "cookie_count": len(cookies),
        }

    def close(self) -> None:
        self.clear()
        if self.material is not None:
            self.material.fields.clear()
        self.material = None

    def allows_origin(self, url: str) -> bool:
        origin = origin_of(url)
        return origin in self.identity.origins

    def request(self, url: str, explicit_headers=()) -> PrivateRequest:
        request = PrivateRequest(url=url)
        headers: dict[str, tuple[str, str]] = {}
        if self.material is not None and self.identity.kind in {"basic", "bearer"}:
            values = self.material.fields
            if self.identity.kind == "basic":
                authorization = basic_authorization(
                    values["username"].get_secret_value(),
                    values["password"].get_secret_value(),
                )
            else:
                authorization = "Bearer " + values["token"].get_secret_value()
            headers["authorization"] = ("Authorization", authorization)
        if self.material is not None and self.identity.kind == "api-keys":
            query = {}
            for qualified, field in self.material.fields.items():
                location, name = qualified.split(":", 1)
                if location == "header":
                    if any(
                        key.decode("ascii").lower() == name.lower() for key, _ in explicit_headers
                    ):
                        raise ValueError("selected credential conflicts with an explicit header")
                    headers[name.lower()] = (name, field.get_secret_value())
                elif location == "query":
                    query[name] = field.get_secret_value()
            parsed = urlsplit(url)
            names = {name for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}
            if names.intersection(query):
                raise ValueError("selected credential conflicts with an explicit query field")
            if query:
                combined = parsed.query + ("&" if parsed.query else "") + urlencode(query)
                request.url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, combined, ""))
                if len(request.url.encode()) > MAX_URL_BYTES:
                    raise ValueError("engagement expanded URL exceeds bound")
        jar = self.cookies.get(origin_of(url))
        if jar is not None:
            discard_expired_cookies(jar, time.time())
            cookie_request = httpx.Request("GET", url)
            matching_cookies(jar, url).set_cookie_header(cookie_request)
            value = cookie_request.headers.get("cookie")
            if value:
                headers["cookie"] = ("Cookie", value)
        supplied_sendable = "cookie" in headers
        if self.identity.kind == "api-keys":
            # Retain explicitly selected non-auth cookie parameters. Never let a
            # scanner replace a configured or learned credential cookie.
            supplied_names = {
                key.removeprefix("cookie:")
                for key in self.material.fields
                if key.startswith("cookie:")
            }
            learned_names = {cookie.name for cookie in jar.jar} if jar is not None else set()
            explicit = {}
            for key, raw in explicit_headers:
                if key.lower() != b"cookie":
                    continue
                for pair in raw.decode("ascii").split(";"):
                    name, separator, value = pair.strip().partition("=")
                    if (
                        not separator
                        or not valid_cookie_name(name)
                        or not valid_cookie_value(value)
                        or name in explicit
                        or name in supplied_names
                        or name in learned_names
                    ):
                        raise ValueError("selected credential conflicts with an explicit cookie")
                    explicit[name] = value
            if explicit:
                value = "; ".join(f"{name}={value}" for name, value in explicit.items())
                if "cookie" in headers:
                    value += "; " + headers["cookie"][1]
                headers["cookie"] = ("Cookie", value)
        if any(not is_field_value(value) for _, value in headers.values()):
            raise ValueError("engagement expanded header is invalid")
        if self.supplied_cookies and not supplied_sendable:
            self.state = "blocked"
            raise ValueError("engagement supplied session has no sendable cookies")
        request.headers = [
            (name.encode("ascii"), value.encode("ascii")) for name, value in headers.values()
        ]
        if (
            sum(len(name) + len(value) + 4 for name, value in request.headers)
            > MAX_CONTROLLED_HEADER_BYTES
        ):
            raise ValueError("engagement expanded headers exceed bound")
        return request

    def receive(self, response: PrivateResponse, epoch: object) -> None:
        self.require_epoch(epoch)
        origin = origin_of(response.url)
        jar = httpx.Cookies(self.cookies.get(origin))
        message = httpx.Response(
            response.status, headers=response.headers, request=httpx.Request("GET", response.url)
        )
        jar.extract_cookies(message)
        prospective = {**self.cookies, origin: jar}
        self._validate_cookies(prospective)
        self.cookies[origin] = jar

    @staticmethod
    def _validate_cookies(jars: dict[str, httpx.Cookies]) -> None:
        count = size = 0
        now = time.time()
        for key, cookies in jars.items():
            parsed = urlsplit(key)
            encoded = encode_cookies(
                cookies,
                (
                    parsed.scheme,
                    parsed.hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                ),
                now,
            )
            count += len(cookies.jar)
            size += len(encoded)
        if count > MAX_COOKIES or size > MAX_COOKIE_SESSION_BYTES:
            raise ValueError("engagement session cookies exceed bound")
