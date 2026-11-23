"""Closed non-secret engagement and canonical route policy (proposal 0033)."""

from __future__ import annotations

import ipaddress
import re
from typing import Annotated, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import ConfigDict, Field, model_validator

from contractor_runtime.contracts.base import WireModel
from contractor_runtime.contracts.settings import RuntimeCredentialRef

ENGAGEMENT_SCHEMA = "contractor.engagement.v1"
MAX_ENGAGEMENT_BYTES = 4 * 1024 * 1024
MAX_URL_BYTES = 8192
_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
_PATH_SAFE = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/-._~!$&'()*+,;=:@"
)


def _bad() -> ValueError:
    return ValueError("engagement is invalid")


def _unique(values: list[str], low: int, high: int) -> bool:
    return low <= len(values) <= high and len(set(values)) == len(values)


def _text(value: str, maximum: int) -> bool:
    return (
        0 < len(value.encode("utf-8")) <= maximum
        and bool(value.strip())
        and not any(ord(c) < 32 or ord(c) == 127 for c in value)
    )


def _path(raw: str) -> str:
    if not raw.startswith("/") or any(c in raw for c in "?#\\") or "//" in raw:
        raise _bad()
    for m in re.finditer("%", raw):
        token = raw[m.start() + 1 : m.start() + 3]
        if len(token) != 2 or re.fullmatch("[0-9a-fA-F]{2}", token) is None:
            raise _bad()
        byte = int(token, 16)
        if byte in {47, 92, 37, 127} or byte < 32:
            raise _bad()
    decoded = unquote(raw, encoding="utf-8", errors="strict")
    if any(part in {".", ".."} for part in decoded.split("/")):
        raise _bad()
    if any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in decoded):
        raise _bad()
    return "".join(chr(byte) if byte in _PATH_SAFE else f"%{byte:02X}" for byte in decoded.encode())


def canonical_url(raw: str) -> str:
    """Return the path used for matching/transmission, without DNS or target I/O."""
    if (
        not raw
        or len(raw.encode("utf-8")) > MAX_URL_BYTES
        or any(c in raw for c in "\\#")
        or any(ord(c) <= 32 or ord(c) == 127 for c in raw)
    ):
        raise _bad()
    try:
        parsed = urlsplit(raw)
        if (
            raw.split(":", 1)[0] not in {"http", "https"}
            or not parsed.netloc
            or "@" in parsed.netloc
        ):
            raise _bad()
        host = parsed.hostname
        if not host or "%" in host:
            raise _bad()
        # urlsplit lowercases hostname; verify literals using the original authority.
        source_host = (
            parsed.netloc[1 : parsed.netloc.index("]")]
            if parsed.netloc.startswith("[")
            else parsed.netloc.split(":")[0]
        )
        try:
            ip = ipaddress.ip_address(source_host)
        except ValueError:
            if len(source_host) > 253 or source_host.endswith(".") or ":" in source_host:
                raise _bad() from None
            labels = source_host.split(".")
            if all(label.isdecimal() and label.isascii() for label in labels):
                raise _bad() from None
            for label in labels:
                if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) is None:
                    raise _bad() from None
            try:
                for label in labels:
                    int(label[2:], 16) if label.lower().startswith("0x") else int(label, 10)
            except ValueError:
                pass
            else:
                raise _bad()
            host = source_host.lower()
        else:
            if str(ip) != source_host or (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped):
                raise _bad()
            host = f"[{ip}]" if ip.version == 6 else str(ip)
        if parsed.netloc.startswith("["):
            tail = parsed.netloc[parsed.netloc.index("]") + 1 :]
            if tail and not tail.startswith(":"):
                raise _bad()
        port = parsed.port
        if parsed.netloc.endswith(":"):
            raise _bad()
        if port is not None:
            port_text = parsed.netloc.rsplit(":", 1)[1]
            if not 1 <= port <= 65535 or str(port) != port_text:
                raise _bad()
            if not (
                (parsed.scheme == "http" and port == 80)
                or (parsed.scheme == "https" and port == 443)
            ):
                host += f":{port}"
        path = _path(parsed.path or "/")
        for m in re.finditer("%", parsed.query):
            if re.fullmatch("[0-9a-fA-F]{2}", parsed.query[m.start() + 1 : m.start() + 3]) is None:
                raise _bad()
        result = parsed.scheme + "://" + host + path
        if "?" in raw:
            result += "?" + parsed.query
        if len(result.encode()) > MAX_URL_BYTES:
            raise _bad()
        return result
    except (ValueError, UnicodeError) as error:
        raise _bad() from error


def resolve_redirect_url(base: str, reference: str) -> str:
    """Resolve without erasing syntax that the canonical URL policy rejects.

    Only the already validated base is parsed. Joining a raw reference through
    urljoin/urlsplit first would remove dot segments or control characters and
    turn a prohibited target into an apparently valid URL before admission.
    """
    if len(reference.encode("utf-8")) > MAX_URL_BYTES:
        raise _bad()
    base = canonical_url(base)
    parsed = urlsplit(base)
    origin = parsed.scheme + "://" + parsed.netloc
    if not reference:
        candidate = base
    elif ":" in reference.split("?", 1)[0].split("/", 1)[0]:
        candidate = reference
    elif reference.startswith("//"):
        candidate = parsed.scheme + ":" + reference
    elif reference.startswith("/"):
        candidate = origin + reference
    elif reference.startswith("?"):
        candidate = origin + parsed.path + reference
    else:
        candidate = origin + parsed.path.rsplit("/", 1)[0] + "/" + reference
    return canonical_url(candidate)


def _origin(value: str) -> bool:
    try:
        return (
            not any(c in value for c in "?#")
            and canonical_url(value + "/") == value + "/"
            and urlsplit(value + "/").path == "/"
        )
    except ValueError:
        return False


def _canonical_path(value: str) -> bool:
    try:
        return len(value.encode()) <= 8192 and _path(value) == value
    except (ValueError, UnicodeError):
        return False


class _EngagementModel(WireModel):
    # Authoring JSON has only its published camelCase member names; accepting
    # Python attribute spellings would disagree with Server strictjson.
    model_config = ConfigDict(populate_by_name=False, validate_by_name=False)


class Route(_EngagementModel):
    origin: str
    path_prefixes: list[str]
    methods: list[str]
    address_ranges: list[str]

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        if (
            not _origin(self.origin)
            or not _unique(self.path_prefixes, 1, 128)
            or not _unique(self.methods, 1, 9)
            or not _unique(self.address_ranges, 1, 128)
            or not all(_canonical_path(x) for x in self.path_prefixes)
            or not set(self.methods) <= _METHODS
        ):
            raise _bad()
        for value in self.address_ranges:
            prefix = ipaddress.ip_network(value, strict=True)
            if str(prefix) != value or (
                isinstance(prefix, ipaddress.IPv6Network) and prefix.network_address.ipv4_mapped
            ):
                raise _bad()
        return self

    def matches(self, origin: str, path: str, method: str) -> bool:
        try:
            address = ipaddress.ip_address(urlsplit(origin).hostname)
        except ValueError:
            pass
        else:
            if not any(address in ipaddress.ip_network(value) for value in self.address_ranges):
                return False
        return (
            self.origin == origin
            and method in self.methods
            and any(
                p == "/" or path == p or path.startswith(p.rstrip("/") + "/")
                for p in self.path_prefixes
            )
        )


class Egress(_EngagementModel):
    routes: list[Route]
    exclusions: list[Route]


class Identity(_EngagementModel):
    id: str
    kind: str
    origins: list[str]
    role: str | None = None
    tenant: str | None = None
    credential: RuntimeCredentialRef | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_credential_members(cls, value):
        if (
            isinstance(value, dict)
            and isinstance(value.get("credential"), dict)
            and set(value["credential"]) != {"credentialId", "kind"}
        ):
            raise _bad()
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (
            _ID.fullmatch(self.id) is None
            or not _unique(self.origins, 1, 16)
            or not all(_origin(x) for x in self.origins)
            or (self.role is not None and not _text(self.role, 256))
            or (self.tenant is not None and not _text(self.tenant, 256))
        ):
            raise _bad()
        if self.kind == "anonymous":
            if self.id != "anonymous" or self.credential is not None:
                raise _bad()
        else:
            kinds = {
                "basic": "target-basic@1",
                "bearer": "target-bearer@1",
                "cookies": "target-cookies@1",
                "api-keys": "target-api-keys@1",
            }
            if (
                self.id == "anonymous"
                or self.kind not in kinds
                or self.credential is None
                or self.credential.kind != kinds[self.kind]
            ):
                raise _bad()
        return self


class HTTPLimits(_EngagementModel):
    requests_per_second: Annotated[int, Field(ge=1, le=4294967295)] | None = None
    concurrent_requests: Annotated[int, Field(ge=1, le=4294967295)] | None = None
    request_body_bytes: Annotated[int, Field(ge=1, le=4294967295)] | None = None
    response_body_bytes: Annotated[int, Field(ge=1, le=4294967295)] | None = None


class Engagement(_EngagementModel):
    schema_: Literal[ENGAGEMENT_SCHEMA] = Field(alias="schema")
    egress: Egress
    identities: list[Identity]
    limits: HTTPLimits | None = None

    @model_validator(mode="after")
    def validate_engagement(self) -> Self:
        if (
            not 1 <= len(self.egress.routes) <= 128
            or not 0 <= len(self.egress.exclusions) <= 128
            or not 1 <= len(self.identities) <= 8
        ):
            raise _bad()
        origins = {route.origin for route in self.egress.routes}
        if len(origins) > 16:
            raise _bad()
        for route in self.egress.exclusions:
            if route.origin not in origins:
                raise _bad()
        ids: set[str] = set()
        credential_ids: set[str] = set()
        anonymous = 0
        for identity in self.identities:
            if identity.id in ids or not set(identity.origins) <= origins:
                raise _bad()
            ids.add(identity.id)
            if identity.credential is not None:
                if identity.credential.credential_id in credential_ids:
                    raise _bad()
                credential_ids.add(identity.credential.credential_id)
            anonymous += identity.kind == "anonymous"
        if anonymous != 1:
            raise _bad()
        if (
            len(self.model_dump_json(by_alias=True, exclude_none=True).encode())
            > MAX_ENGAGEMENT_BYTES
        ):
            raise _bad()
        return self

    def allows_url(self, raw: str, method: str) -> bool:
        try:
            parsed = urlsplit(canonical_url(raw))
        except ValueError:
            return False
        origin = parsed.scheme + "://" + parsed.netloc
        if any(r.matches(origin, parsed.path, method) for r in self.egress.exclusions):
            return False
        return any(r.matches(origin, parsed.path, method) for r in self.egress.routes)
