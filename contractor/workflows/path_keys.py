"""Stable, collision-resistant keys for OpenAPI paths and route groups."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

# Keep enough room in a filesystem component for namespace prefixes such as
# ``trace-graph-pathpar:openapi:``.  FileArtifactService stores each artifact
# filename segment directly on disk, where common filesystems cap components at
# 255 bytes.  Encoded keys are ASCII, so character and byte limits coincide.
MAX_OPENAPI_PATH_KEY_LENGTH = 160
_PATH_KEY_VERSION = "v2"


def _is_readable_byte(value: int) -> bool:
    return (
        ord("0") <= value <= ord("9")
        or ord("a") <= value <= ord("z")
        or value == ord("-")
    )


def _encoded_path_body(path: str) -> str:
    """Encode a path as one portable filesystem component."""
    if path == "/":
        return "p-_root_"

    if path.startswith("/"):
        raw = path[1:]
        prefix = "p-"
    else:
        # OpenAPI requires a leading slash, but keep invalid input distinct
        # from its valid counterpart rather than silently colliding with it.
        raw = path
        prefix = "p-relative_"

    encoded: list[str] = [prefix]
    for value in raw.encode("utf-8"):
        if value == ord("/"):
            encoded.append("__")
        elif _is_readable_byte(value):
            encoded.append(chr(value))
        else:
            encoded.append(f"_{value:02X}")
    return "".join(encoded)


def _scoped_path_key(path: str, depth: int) -> str:
    normalized_depth = max(depth, 0)
    scope = f"{_PATH_KEY_VERSION}/d{normalized_depth}/"
    body = _encoded_path_body(path)
    body_limit = MAX_OPENAPI_PATH_KEY_LENGTH - len(scope)
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    suffix = f"_h{digest}"
    if body_limit < len(suffix) + 1:
        raise ValueError(f"OpenAPI group depth is too large: {depth}")
    if len(body) > body_limit:
        body = body[: body_limit - len(suffix)].rstrip("_") + suffix
    return scope + body


def openapi_path_key(path: str) -> str:
    """Return a filesystem-safe, collision-resistant per-path key.

    The former implementation removed braces and replaced path separators with
    underscores, so distinct paths such as ``/users/{id}`` and ``/users/id``
    shared workflow refs, memory namespaces, and artifacts. Lowercase ASCII
    letters, digits, and hyphens remain readable; slashes become ``__``; every
    other UTF-8 byte uses an ``_XX`` escape. Escaping uppercase bytes also keeps
    distinct routes distinct on case-insensitive filesystems. The ``p-`` body
    prefix avoids Windows device-name components such as ``CON`` and ``NUL``.

    Long routes are bounded to a readable prefix plus their full SHA-256 digest.
    This prevents escaped Unicode paths from exceeding filesystem component
    limits without reintroducing deterministic collisions. The ``v2/d0/``
    directory scope both separates ambiguous historical keys and identifies
    this as a per-path namespace rather than a configured route group.
    """
    return _scoped_path_key(path, 0)


def _path_prefix(path: str, depth: int) -> str:
    segments = [segment for segment in path.strip("/").split("/") if segment]
    selected = segments[:depth]
    return "/" + "/".join(selected) if selected else "/"


def openapi_group_key(path: str, depth: int) -> str:
    """Return the canonical key for ``path`` at a configured group depth.

    The requested positive depth remains in the namespace even when the route
    has fewer segments. This prevents reports produced under different grouping
    configurations from co-mingling with each other or with literal routes.
    """
    if depth <= 0:
        return openapi_path_key(path)
    return _scoped_path_key(_path_prefix(path, depth), depth)


def openapi_group_keys(
    path: str,
    depths: Iterable[int],
) -> list[str]:
    """Return canonical keys for ``depths`` in first-seen order."""
    keys: list[str] = []
    for depth in depths:
        key = openapi_group_key(path, depth)
        if key not in keys:
            keys.append(key)
    return keys
