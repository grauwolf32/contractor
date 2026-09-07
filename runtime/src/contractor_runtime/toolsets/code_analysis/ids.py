"""Opaque allocation-local graph symbol identifiers (standard library only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

SYMBOL_ID_PREFIX = "cas1"
MAX_SYMBOL_ID_BYTES = 512
SYMBOL_KEY_BYTES = 32
MAX_SYMBOL_INDEX = 2_147_483_647


@dataclass(frozen=True, slots=True)
class DecodedSymbolID:
    snapshot_digest: str
    index: int
    upstream_binding: bytes


def encode_symbol_id(key: bytes, snapshot_digest: str, index: int, upstream_id: str) -> str:
    _validate_key(key)
    if not _valid_digest(snapshot_digest):
        raise ValueError("invalid snapshot digest")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= MAX_SYMBOL_INDEX:
        raise ValueError("invalid symbol index")
    upstream_binding = _upstream_binding(key, snapshot_digest, index, upstream_id)
    body = json.dumps(
        {"binding": _encode(upstream_binding), "digest": snapshot_digest, "index": index},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    signature = hmac.digest(key, SYMBOL_ID_PREFIX.encode("ascii") + b"\x00" + body, "sha256")
    return f"{SYMBOL_ID_PREFIX}.{_encode(body)}.{_encode(signature)}"


def decode_symbol_id(key: bytes, value: str) -> DecodedSymbolID:
    _validate_key(key)
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or len(value) > MAX_SYMBOL_ID_BYTES
    ):
        raise ValueError("invalid symbol ID")
    try:
        prefix, encoded_body, encoded_signature = value.split(".")
        if prefix != SYMBOL_ID_PREFIX:
            raise ValueError
        body = _decode(encoded_body)
        signature = _decode(encoded_signature)
        expected = hmac.digest(key, SYMBOL_ID_PREFIX.encode("ascii") + b"\x00" + body, "sha256")
        if len(signature) != hashlib.sha256().digest_size or not hmac.compare_digest(
            signature, expected
        ):
            raise ValueError
        document = json.loads(body)
        if (
            not isinstance(document, dict)
            or set(document) != {"binding", "digest", "index"}
            or json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii") != body
        ):
            raise ValueError
        binding_value = document["binding"]
        if not isinstance(binding_value, str):
            raise ValueError
        binding = _decode(binding_value)
        digest = document["digest"]
        index = document["index"]
        if (
            len(binding) != hashlib.sha256().digest_size
            or not isinstance(digest, str)
            or not _valid_digest(digest)
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index <= MAX_SYMBOL_INDEX
        ):
            raise ValueError
        return DecodedSymbolID(digest, index, binding)
    except (KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid symbol ID") from None


def encode_symbol_key(key: bytes) -> str:
    _validate_key(key)
    return _encode(key)


def decode_symbol_key(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("invalid symbol key")
    try:
        key = _decode(value)
    except ValueError:
        raise ValueError("invalid symbol key") from None
    _validate_key(key)
    return key


def symbol_id_matches_upstream(
    key: bytes,
    decoded: DecodedSymbolID,
    upstream_id: str,
) -> bool:
    """Verify that an already authenticated token selects this complete upstream ID."""

    try:
        expected = _upstream_binding(key, decoded.snapshot_digest, decoded.index, upstream_id)
    except ValueError:
        return False
    return hmac.compare_digest(decoded.upstream_binding, expected)


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != SYMBOL_KEY_BYTES:
        raise ValueError("invalid symbol key")


def _upstream_binding(key: bytes, snapshot_digest: str, index: int, upstream_id: str) -> bytes:
    _validate_key(key)
    if not _valid_digest(snapshot_digest):
        raise ValueError("invalid snapshot digest")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= MAX_SYMBOL_INDEX:
        raise ValueError("invalid symbol index")
    if not isinstance(upstream_id, str) or not upstream_id:
        raise ValueError("invalid upstream ID")
    try:
        encoded_id = upstream_id.encode("utf-8")
    except UnicodeError:
        raise ValueError("invalid upstream ID") from None
    return hmac.digest(
        key,
        b"upstream\x00"
        + snapshot_digest.encode("ascii")
        + b"\x00"
        + index.to_bytes(4, "big")
        + b"\x00"
        + encoded_id,
        "sha256",
    )


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    if not value or any(not (character.isalnum() or character in "-_") for character in value):
        raise ValueError("invalid base64url")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error):
        raise ValueError("invalid base64url") from None
    if _encode(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded
