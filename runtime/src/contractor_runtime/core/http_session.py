"""Bounded cookie state shared by ordinary HTTP and controlled identities."""

from __future__ import annotations

import base64
import json
from http.cookiejar import Cookie
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from contractor_runtime.contracts.fields import is_field_value
from contractor_runtime.strict_json import strict_json_loads

MAX_COOKIES = 128
MAX_COOKIE_SESSION_BYTES = 65536


class CookieStateError(ValueError):
    """Invalid private state; messages contain no cookie values."""


def basic_authorization(username: str, password: str) -> str:
    """Encode the same UTF-8 Basic pair for every HTTP caller."""
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def is_cookie_pair(name: str, value: str | None) -> bool:
    pair = name if value is None else f"{name}={value}"
    return bool(name) and is_field_value(pair)


def discard_expired_cookies(cookies: httpx.Cookies, now: float) -> None:
    """Remove expired source entries using the owning session's clock."""
    for cookie in list(cookies.jar):
        if cookie.is_expired(now):
            cookies.jar.clear(cookie.domain, cookie.path, cookie.name)


class _CookieRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[0, 1]
    name: str
    value: str | None
    port: str | None
    port_specified: bool
    domain: str
    domain_specified: bool
    domain_initial_dot: bool
    path: str
    path_specified: bool
    secure: bool
    expires: int | None
    discard: bool
    comment: str | None
    comment_url: str | None
    rest: dict[str, str | None]
    rfc2109: bool

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and type(value.get("version")) is not int:
            raise ValueError("invalid cookie version")
        return value

    @model_validator(mode="after")
    def validate_cookie(self) -> _CookieRow:
        for value in (self.name, self.value, self.domain, self.path, self.port):
            if value is not None and any(c in value for c in "\r\n\x00"):
                raise ValueError("invalid cookie")
        if not is_cookie_pair(self.name, self.value) or not self.path.startswith("/"):
            raise ValueError("invalid cookie")
        if self.domain_initial_dot and not self.domain.startswith("."):
            raise ValueError("invalid cookie domain")
        if not self.domain and self.domain_specified:
            raise ValueError("invalid cookie domain")
        if self.port_specified and self.port is None:
            raise ValueError("invalid cookie port")
        return self


class _CookieDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    target_origin: tuple[str, str, int] | None
    cookies: list[_CookieRow] = Field(max_length=MAX_COOKIES)

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value: Any) -> Any:
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("invalid cookie schema")
        return value


def encode_cookies(
    cookies: httpx.Cookies, origin: tuple[str, str, int] | None, now: float
) -> bytes:
    """Prune expired source entries and encode complete matching/lifetime attributes."""
    try:
        discard_expired_cookies(cookies, now)
        # Cookie's internal attribute is named _rest; never serialize arbitrary
        # instance members, headers, auth defaults or Runtime settings.
        rows = []
        fields = tuple(_CookieRow.model_fields)
        for cookie in cookies.jar:
            rows.append(
                _CookieRow(
                    **{
                        name: cookie._rest if name == "rest" else getattr(cookie, name)
                        for name in fields
                    }
                )
            )
        rows.sort(key=lambda row: (row.domain, row.path, row.name))
        document = _CookieDocument(schema_version=1, target_origin=origin, cookies=rows)
        data = json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(data) > MAX_COOKIE_SESSION_BYTES:
            raise ValueError("cookie document too large")
        return data
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise CookieStateError("http_cookie_state_invalid") from None


def decode_cookies(data: bytes, origin: tuple[str, str, int] | None, now: float) -> httpx.Cookies:
    """Validate the complete bounded document before exposing any cookie."""
    try:
        if len(data) > MAX_COOKIE_SESSION_BYTES:
            raise ValueError("cookie document too large")
        raw = strict_json_loads(data)
        # JSON arrays represent the fixed origin tuple; all other validation
        # remains strict (including bool versus integer).
        if isinstance(raw, dict) and isinstance(raw.get("target_origin"), list):
            raw["target_origin"] = tuple(raw["target_origin"])
        document = _CookieDocument.model_validate(raw)
        if document.target_origin != origin:
            raise ValueError("cookie origin changed")
        cookies = httpx.Cookies()
        identities = set()
        for row in document.cookies:
            identity = (row.domain, row.path, row.name)
            if identity in identities:
                raise ValueError("duplicate cookie identity")
            identities.add(identity)
            cookie = Cookie(**row.model_dump())
            if not cookie.is_expired(now):
                cookies.jar.set_cookie(cookie)
        return cookies
    except (ValueError, TypeError, OverflowError, UnicodeError, ValidationError, RecursionError):
        raise CookieStateError("http_cookie_state_invalid") from None


def extract_sendable_cookies(cookies: httpx.Cookies, response: httpx.Response) -> None:
    """Store a response's cookies, ignoring a Set-Cookie that cannot be sent back.

    A cookie whose name and value are not one Cookie field value (for example
    a UTF-8 value httpx cannot encode) would make every later request to its
    origin invalid, so it is dropped and any cookie it would replace is kept.
    """

    previous = {(cookie.domain, cookie.path, cookie.name): cookie for cookie in cookies.jar}
    cookies.extract_cookies(response)
    for cookie in list(cookies.jar):
        if is_cookie_pair(cookie.name, cookie.value):
            continue
        key = (cookie.domain, cookie.path, cookie.name)
        cookies.jar.clear(*key)
        kept = previous.get(key)
        if kept is not None:
            cookies.jar.set_cookie(kept)


def matching_cookies(cookies: httpx.Cookies, url: str) -> httpx.Cookies:
    """Enforce host-only cookies as well as CookieJar's domain/path/Secure policy.

    Python's default Netscape policy accepts host-only cookies on subdomains;
    filter those before httpx applies the remaining RFC matching rules.
    Explicit unscoped http_session_set cookies retain their documented scope.
    """
    host = urlsplit(url).hostname or ""
    if ":" in host:
        host = f"[{host}]"
    result = httpx.Cookies()
    for cookie in cookies.jar:
        if (
            cookie.domain
            and not cookie.domain_specified
            and cookie.domain not in {host, host + ".local"}
        ):
            continue
        result.jar.set_cookie(cookie)
    return result
