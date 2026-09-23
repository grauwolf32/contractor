"""Signed pagination cursors, canonical base64url and page-limit checks.

Toolsets hand cursors to models and accept them back verbatim, so the wire
format is fixed: ``base64url(JCS(document)) + "." + base64url(HMAC-SHA256)``
without padding. Every decode failure is a ``ValueError``; callers map it to
their own public error code.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from typing import Any

import jcs

MAX_CURSOR_BYTES = 2048

_BASE64URL_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def b64url_encode(value: bytes) -> str:
    """Encode ``value`` as unpadded base64url."""

    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def b64url_decode(value: str) -> bytes:
    """Decode canonical unpadded base64url or raise ``ValueError``."""

    if not value or any(character not in _BASE64URL_ALPHABET for character in value):
        raise ValueError("invalid base64url")
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if b64url_encode(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded


def query_digest(document: Mapping[str, Any]) -> str:
    """Return the digest that binds a cursor to one normalized query."""

    return "sha256:" + hashlib.sha256(jcs.canonicalize(dict(document))).hexdigest()


def encode_cursor(key: bytes | bytearray, document: Mapping[str, str | int]) -> str:
    body = jcs.canonicalize(dict(document))
    signature = hmac.digest(bytes(key), body, "sha256")
    return f"{b64url_encode(body)}.{b64url_encode(signature)}"


def decode_cursor(
    key: bytes | bytearray, value: object, fields: Mapping[str, type[str] | type[int]]
) -> dict[str, Any]:
    """Verify and decode a cursor whose fields are exactly ``fields``.

    ``str`` fields must be strings and ``int`` fields non-negative integers.
    """

    if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_BYTES:
        raise ValueError("invalid cursor")
    try:
        encoded_body, encoded_signature = value.split(".", 1)
        body = b64url_decode(encoded_body)
        signature = b64url_decode(encoded_signature)
        if not hmac.compare_digest(signature, hmac.digest(bytes(key), body, "sha256")):
            raise ValueError
        document = json.loads(body)
        if jcs.canonicalize(document) != body or set(document) != set(fields):
            raise ValueError
        for name, kind in fields.items():
            if not _field_valid(document[name], kind):
                raise ValueError
        return document
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid cursor") from None


def _field_valid(value: object, kind: type[str] | type[int]) -> bool:
    if kind is str:
        return isinstance(value, str)
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def require_limit(value: object, maximum: int, error: Callable[[], Exception]) -> int:
    """Return an integer page limit in ``1..maximum`` or raise ``error()``."""

    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise error()
    return value
