"""Strict framed Trailmark child process.

This module must keep imports above ``main`` in the standard library.  The
address-space limit and output redirection are installed before Trailmark is
imported so a parser failure cannot contaminate Runtime stdout/stderr.
"""

from __future__ import annotations

import json
import os
import resource
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
MAX_REQUEST_ID_CHARS = 64
MAX_SYMBOLS_RESPONSE = 200
MAX_NAME_CHARS = 256
MAX_QUERY_CHARS = 256
MAX_PATH_CHARS = 4096

_COVERAGE_KEYS = {
    "analyzedFiles",
    "analyzedBytes",
    "binaryFiles",
    "unsupportedSourceFiles",
    "oversizedFiles",
    "parseErrors",
    "incomplete",
    "reasons",
}
_COVERAGE_REASONS = {"file_limit", "byte_limit", "symbol_limit", "deadline", "parse_errors"}


class _ProtocolError(RuntimeError):
    pass


class _RequestError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _TrailmarkAdapter:
    """Version-specific adapter over public Trailmark graph data only."""

    def __init__(self) -> None:
        self._graph: Any | None = None
        self._snapshot_digest: str | None = None
        self._coverage: dict[str, Any] | None = None
        self._languages: tuple[str, ...] = ()
        self._symbols: tuple[dict[str, Any], ...] = ()
        self._raw_ids: tuple[str, ...] = ()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._name_index: dict[str, tuple[str, ...]] = {}
        self._incoming: dict[str, tuple[tuple[str, str], ...]] = {}
        self._outgoing: dict[str, tuple[tuple[str, str], ...]] = {}
        self._symbol_key: bytes | None = None

    def build(self, arguments: object) -> dict[str, Any]:
        if self._graph is not None:
            raise _RequestError("invalid_state")
        document = _object(arguments, {"snapshotDigest", "coverage", "symbolKey"})
        digest = document["snapshotDigest"]
        if not isinstance(digest, str) or not _valid_digest(digest):
            raise _RequestError("invalid_request")
        coverage = _coverage(document["coverage"])

        # Deliberately deferred until after main() installs RLIMIT_AS and
        # redirects process-global output away from the protocol descriptor.
        from trailmark.models.graph import CodeGraph
        from trailmark.parse import detect_languages, parse_directory
        from trailmark.query.api import detect_entrypoints

        from contractor_runtime.toolsets.code_analysis_ids import (
            MAX_SYMBOL_INDEX,
            decode_symbol_key,
            encode_symbol_id,
        )

        try:
            symbol_key = decode_symbol_key(document["symbolKey"])
        except ValueError:
            raise _RequestError("invalid_request") from None

        languages = tuple(sorted(set(detect_languages("."))))
        if languages:
            graph = parse_directory(".", language="auto")
            graph.entrypoints.update(detect_entrypoints(graph, "."))
        else:
            graph = CodeGraph(language="", root_path=".")

        mirror_root = Path.cwd().resolve()
        if any(not isinstance(node_id, str) for node_id in graph.nodes):
            raise _RequestError("code_analysis_engine_failed")
        raw_ids = tuple(sorted(graph.nodes))
        if len(raw_ids) > MAX_SYMBOL_INDEX + 1:
            raise _RequestError("code_analysis_capacity_exceeded")
        symbols: list[dict[str, Any]] = []
        nodes: dict[str, dict[str, Any]] = {}
        name_index: dict[str, set[str]] = {}
        symbol_ids: set[str] = set()
        for index, raw_id in enumerate(raw_ids):
            node = graph.nodes[raw_id]
            symbol_id = encode_symbol_id(symbol_key, digest, index, raw_id)
            if symbol_id in symbol_ids:
                raise _RequestError("code_analysis_engine_failed")
            symbol_ids.add(symbol_id)
            projection = _project_node(node, mirror_root, symbol_id)
            nodes[raw_id] = projection
            symbols.append(projection)
            for key in _index_name_keys(raw_id, projection["name"]):
                name_index.setdefault(key, set()).add(raw_id)
        symbols.sort(
            key=lambda item: (
                item["path"],
                item["line"],
                item["column"],
                item["name"],
                item["kind"],
            )
        )

        incoming: dict[str, list[tuple[str, str]]] = {}
        outgoing: dict[str, list[tuple[str, str]]] = {}
        for edge in graph.edges:
            if _enum_value(edge.kind) != "calls":
                continue
            source = str(edge.source_id)
            target = str(edge.target_id)
            if source not in nodes or target not in nodes:
                raise _RequestError("code_analysis_engine_failed")
            confidence = _edge_confidence(edge.confidence)
            outgoing.setdefault(source, []).append((target, confidence))
            incoming.setdefault(target, []).append((source, confidence))

        self._graph = graph
        self._snapshot_digest = digest
        self._coverage = coverage
        self._languages = languages
        self._symbols = tuple(symbols)
        self._raw_ids = raw_ids
        self._nodes = nodes
        self._name_index = {key: tuple(sorted(values)) for key, values in name_index.items()}
        self._incoming = {key: tuple(sorted(values)) for key, values in incoming.items()}
        self._outgoing = {key: tuple(sorted(values)) for key, values in outgoing.items()}
        self._symbol_key = symbol_key
        return self.summary()

    def summary(self) -> dict[str, Any]:
        graph = self._require_graph()
        assert self._snapshot_digest is not None
        assert self._coverage is not None
        call_edges = sum(1 for edge in graph.edges if _enum_value(edge.kind) == "calls")
        return {
            "snapshotDigest": self._snapshot_digest,
            "coverage": self._coverage,
            "languages": list(self._languages),
            "nodeCount": len(graph.nodes),
            "edgeCount": len(graph.edges),
            "callEdgeCount": call_edges,
            "entrypointCount": len(graph.entrypoints),
            "dependencyCount": len(graph.dependencies),
            "rssKiB": max(0, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
        }

    def symbols(self, arguments: object) -> dict[str, Any]:
        self._require_graph()
        document = _object(arguments, {"limit"})
        limit = document["limit"]
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_SYMBOLS_RESPONSE
        ):
            raise _RequestError("invalid_request")
        assert self._snapshot_digest is not None
        return {
            "snapshotDigest": self._snapshot_digest,
            "items": list(self._symbols[:limit]),
            "observedTotal": len(self._symbols),
            "truncated": len(self._symbols) > limit,
        }

    def find_symbols(self, arguments: object) -> dict[str, Any]:
        self._require_graph()
        document = _object(arguments, {"query", "offset", "limit"})
        query = _query(document["query"])
        offset, limit = _page_arguments(document)
        matched: set[str] = set()
        for key in _query_name_keys(query):
            matched.update(self._name_index.get(key, ()))
        rows = [self._nodes[raw_id] for raw_id in matched]
        rows.sort(key=_projection_sort_key)
        return self._collection(rows, offset, limit)

    def relationships(self, operation: str, arguments: object) -> dict[str, Any]:
        self._require_graph()
        document = _object(arguments, {"symbolId", "offset", "limit"})
        symbol_id = document["symbolId"]
        if not isinstance(symbol_id, str):
            raise _RequestError("code_analysis_symbol_not_found")
        offset, limit = _page_arguments(document)
        raw_id = self._resolve_symbol_id(symbol_id)
        selected = self._incoming if operation == "find_callers" else self._outgoing
        rows: list[dict[str, Any]] = []
        for related_id, confidence in selected.get(raw_id, ()):
            row = dict(self._nodes[related_id])
            row["confidence"] = confidence
            rows.append(row)
        rows.sort(key=_relationship_sort_key)
        return self._collection(rows, offset, limit)

    def _resolve_symbol_id(self, symbol_id: str) -> str:
        assert self._symbol_key is not None
        assert self._snapshot_digest is not None
        from contractor_runtime.toolsets.code_analysis_ids import (
            decode_symbol_id,
            symbol_id_matches_upstream,
        )

        try:
            decoded = decode_symbol_id(self._symbol_key, symbol_id)
        except ValueError:
            raise _RequestError("code_analysis_symbol_not_found") from None
        if decoded.snapshot_digest != self._snapshot_digest:
            raise _RequestError("code_analysis_stale_symbol")
        try:
            raw_id = self._raw_ids[decoded.index]
        except IndexError:
            raise _RequestError("code_analysis_symbol_not_found") from None
        if not symbol_id_matches_upstream(self._symbol_key, decoded, raw_id):
            raise _RequestError("code_analysis_symbol_not_found")
        return raw_id

    def _collection(self, rows: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
        if offset > len(rows):
            raise _RequestError("code_analysis_input_invalid")
        assert self._snapshot_digest is not None
        page = rows[offset : offset + limit]
        return {
            "snapshotDigest": self._snapshot_digest,
            "items": page,
            "observedTotal": len(rows),
            "truncated": offset + len(page) < len(rows),
        }

    def _require_graph(self) -> Any:
        if self._graph is None:
            raise _RequestError("invalid_state")
        return self._graph


def main() -> int:
    protocol_fd = os.dup(sys.stdout.fileno())
    protocol = os.fdopen(protocol_fd, "wb", buffering=0)
    _silence_process_output()
    try:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        address_limit = (
            MAX_ADDRESS_SPACE_BYTES
            if hard_limit == resource.RLIM_INFINITY
            else min(MAX_ADDRESS_SPACE_BYTES, hard_limit)
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (address_limit, address_limit),
        )
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        return 70

    adapter = _TrailmarkAdapter()
    while True:
        try:
            payload = _read_frame(sys.stdin.buffer)
            if payload is None:
                return 0
            request = _request(payload)
            request_id = request["requestId"]
            try:
                result = _dispatch(adapter, request["operation"], request["arguments"])
                response = {
                    "schemaVersion": SCHEMA_VERSION,
                    "requestId": request_id,
                    "ok": True,
                    "result": result,
                }
            except _RequestError as error:
                response = {
                    "schemaVersion": SCHEMA_VERSION,
                    "requestId": request_id,
                    "ok": False,
                    "code": error.code,
                    "retryable": False,
                }
            _write_frame(protocol, response)
        except (BrokenPipeError, EOFError, _ProtocolError):
            return 65
        except MemoryError:
            return 71
        except BaseException:
            return 70


def _dispatch(adapter: _TrailmarkAdapter, operation: str, arguments: object) -> dict[str, Any]:
    if operation == "build":
        return adapter.build(arguments)
    if operation == "summary":
        _object(arguments, set())
        return adapter.summary()
    if operation == "symbols":
        return adapter.symbols(arguments)
    if operation == "find_symbol":
        return adapter.find_symbols(arguments)
    if operation in {"find_callers", "find_callees"}:
        return adapter.relationships(operation, arguments)
    raise _RequestError("unsupported_operation")


def _request(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, ValueError):
        raise _ProtocolError from None
    request = _object(document, {"schemaVersion", "requestId", "operation", "arguments"})
    if request["schemaVersion"] != SCHEMA_VERSION:
        raise _ProtocolError
    request_id = request["requestId"]
    if (
        not isinstance(request_id, str)
        or not 1 <= len(request_id) <= MAX_REQUEST_ID_CHARS
        or not all(
            character.isascii() and (character.isalnum() or character in "_-")
            for character in request_id
        )
    ):
        raise _ProtocolError
    if not isinstance(request["operation"], str):
        raise _ProtocolError
    return request


def _coverage(value: object) -> dict[str, Any]:
    document = _object(value, _COVERAGE_KEYS)
    for key in _COVERAGE_KEYS - {"incomplete", "reasons"}:
        count = document[key]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= 9_223_372_036_854_775_807
        ):
            raise _RequestError("invalid_request")
    if not isinstance(document["incomplete"], bool):
        raise _RequestError("invalid_request")
    reasons = document["reasons"]
    if (
        not isinstance(reasons, list)
        or len(reasons) > 16
        or any(not isinstance(reason, str) or reason not in _COVERAGE_REASONS for reason in reasons)
        or reasons != sorted(set(reasons))
        or document["incomplete"] != bool(reasons)
    ):
        raise _RequestError("invalid_request")
    return {
        **{key: document[key] for key in _COVERAGE_KEYS - {"reasons"}},
        "reasons": sorted(set(reasons)),
    }


def _object(value: object, keys: set[str]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or any(not isinstance(key, str) for key in value)
    ):
        raise _RequestError("invalid_request")
    return value


def _project_node(node: object, mirror_root: Path, symbol_id: str) -> dict[str, Any]:
    location = getattr(node, "location", None)
    if location is None:
        raise _RequestError("unsafe_graph_projection")
    path = _relative_path(getattr(location, "file_path", None), mirror_root)
    return {
        "symbolId": symbol_id,
        "name": _safe_text(getattr(node, "name", ""), MAX_NAME_CHARS),
        "kind": _safe_text(_enum_value(getattr(node, "kind", "")), 64),
        "path": path,
        "line": _positive_line(getattr(location, "start_line", 1)),
        "endLine": _positive_line(getattr(location, "end_line", 1)),
        "column": _nonnegative_column(getattr(location, "start_col", 0)),
    }


def _query(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_QUERY_CHARS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise _RequestError("code_analysis_input_invalid")
    return value


def _page_arguments(document: Mapping[str, Any]) -> tuple[int, int]:
    offset = document["offset"]
    limit = document["limit"]
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_SYMBOLS_RESPONSE
    ):
        raise _RequestError("code_analysis_input_invalid")
    return offset, limit


def _index_name_keys(identifier: str, name: str) -> set[str]:
    folded_name = name.casefold()
    folded_id = identifier.casefold()
    qualified = _qualified_name(folded_id)
    parts = qualified.split(".")
    return {
        folded_name,
        folded_id,
        *(".".join(parts[index:]) for index in range(len(parts))),
    }


def _query_name_keys(query: str) -> set[str]:
    folded = query.casefold()
    return {folded, _qualified_name(folded)}


def _qualified_name(value: str) -> str:
    return value.replace("::", ".").replace(":", ".").replace("#", ".")


def _projection_sort_key(item: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        item["path"],
        item["line"],
        item["column"],
        item["name"],
        item["kind"],
        item["symbolId"],
    )


def _relationship_sort_key(item: Mapping[str, Any]) -> tuple[object, ...]:
    return (*_projection_sort_key(item), item["confidence"])


def _edge_confidence(value: object) -> str:
    selected = _enum_value(value)
    if selected not in {"certain", "inferred", "uncertain"}:
        raise _RequestError("code_analysis_engine_failed")
    return selected


def _relative_path(value: object, mirror_root: Path) -> str:
    if not isinstance(value, str) or not value or len(value) > 16_384:
        raise _RequestError("unsafe_graph_projection")
    candidate = Path(value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (mirror_root / candidate).resolve()
    )
    try:
        relative = resolved.relative_to(mirror_root).as_posix()
    except ValueError:
        raise _RequestError("unsafe_graph_projection") from None
    if not relative or relative == "." or len(relative.encode("utf-8")) > MAX_PATH_CHARS:
        raise _RequestError("unsafe_graph_projection")
    return relative


def _safe_text(value: object, limit: int) -> str:
    text = str(value)
    cleaned = "".join(
        character if ord(character) >= 0x20 and ord(character) != 0x7F else "�"
        for character in text
    )
    return cleaned[:limit]


def _enum_value(value: object) -> str:
    selected = getattr(value, "value", value)
    return str(selected)


def _positive_line(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 1
    return max(1, min(value, 2_147_483_647))


def _nonnegative_column(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return max(0, min(value, 2_147_483_647))


def _valid_digest(value: str) -> bool:
    if not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _read_frame(stream: Any) -> bytes | None:
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise _ProtocolError
    length = struct.unpack(">I", header)[0]
    if not 0 < length <= MAX_REQUEST_BYTES:
        raise _ProtocolError
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise _ProtocolError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_frame(stream: Any, document: Mapping[str, Any]) -> None:
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise _ProtocolError from None
    if not 0 < len(payload) <= MAX_RESPONSE_BYTES:
        fallback = {
            "schemaVersion": SCHEMA_VERSION,
            "requestId": str(document.get("requestId", "unknown"))[:MAX_REQUEST_ID_CHARS],
            "ok": False,
            "code": "code_analysis_capacity_exceeded",
            "retryable": False,
        }
        payload = json.dumps(fallback, separators=(",", ":"), sort_keys=True).encode("utf-8")
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _silence_process_output() -> None:
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, sys.stdout.fileno())
        os.dup2(null_fd, sys.stderr.fileno())
    finally:
        os.close(null_fd)


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
