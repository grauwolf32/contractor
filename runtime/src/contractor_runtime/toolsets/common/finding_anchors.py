"""Bounded exact-source statement derivation; never consumes finding prose."""

from __future__ import annotations

import ast
import io
import json
import re
import stat
import time
import unicodedata
import zipfile
import zlib
from bisect import bisect_right
from importlib.metadata import version
from pathlib import PurePosixPath
from typing import Any

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from tree_sitter import Node
from tree_sitter_language_pack import Error as ParserLoadError
from tree_sitter_language_pack import downloaded_languages, get_parser

from contractor_runtime.core.lines import line_byte_starts
from contractor_runtime.digests import jcs_digest, sha256_digest

DERIVATION_SCHEMA = "contractor.finding.anchors-derived.v1"
GRAMMAR = "tree-sitter-language-pack/1.14.3;tree-sitter/0.25.2"
MAX_SOURCE_BYTES = 8 << 20
MAX_FILE_BYTES = 4 << 20
MAX_DERIVATION_BYTES = 16 << 10
MAX_ANCHORS = 32
LOCAL_RULE = "source.private-key-literal.ed25519/v1"
_PROFILES = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
}
_STATEMENTS = {
    "python": {
        "expression_statement",
        "assignment",
        "augmented_assignment",
        "call",
        "await",
        "yield",
        "return_statement",
        "raise_statement",
        "assert_statement",
        "delete_statement",
    },
    "javascript": {
        "lexical_declaration",
        "variable_declaration",
        "expression_statement",
        "return_statement",
        "throw_statement",
    },
    "typescript": {
        "lexical_declaration",
        "variable_declaration",
        "expression_statement",
        "return_statement",
        "throw_statement",
    },
}
_SCOPES = {
    "function_definition",
    "class_definition",
    "function_declaration",
    "function_expression",
    "generator_function_declaration",
    "generator_function",
    "arrow_function",
    "method_definition",
    "class_declaration",
    "class",
    "lambda",
    "decorated_definition",
}
_DIRECTIVE = re.compile(
    r"^(?:#!|//[#!]|/\*[!*]|(?:#|//|/\*)\s*(?:type:|coding[:=]|noqa|nosec|pragma|pyright:|mypy:|fmt:|pylint:|eslint|tslint|istanbul|c8\b|prettier|sourceMappingURL|sourceURL|@))"
)


class AnchorUnavailable(Exception):
    """Expected optional preparation outcome; carries no source text."""


def unavailable(reason: str) -> dict[str, Any]:
    return {"schema": DERIVATION_SCHEMA, "availability": "unavailable", "reason": reason}


class _Budget:
    def __init__(self) -> None:
        self.deadline = time.monotonic() + 10
        self.nodes = 0
        self.tokens = 0

    def visit(self, *, token_bytes: int = 0) -> None:
        self.nodes += 1
        self.tokens += token_bytes
        if self.nodes > 100_000 or self.tokens > (256 << 10) or time.monotonic() > self.deadline:
            raise AnchorUnavailable("input_limit")


def derive_zip(
    data: bytes, source_ref: dict[str, str], locations: list[dict[str, Any]], *, qualify_local: bool
) -> dict[str, Any]:
    """Parse selected exact files once, retaining compact components only."""
    if len(data) > MAX_SOURCE_BYTES or len(locations) > MAX_ANCHORS:
        return unavailable("input_limit")
    if not locations:
        return unavailable("not_provided")
    if version("tree-sitter") != "0.25.2" or version("tree-sitter-language-pack") != "1.14.3":
        raise RuntimeError("finding anchor parser does not match its pinned profile")
    budget = _Budget()
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 20_000:
                raise AnchorUnavailable("input_limit")
            by_name = {}
            declared = 0
            for info in entries:
                budget.visit()
                path = _safe_member(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (
                    path in by_name
                    or kind not in {0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG}
                    or info.flag_bits & 1
                ):
                    raise AnchorUnavailable("source_invalid")
                by_name[path] = info
                declared += info.file_size
                if declared > 128 << 20:
                    raise AnchorUnavailable("input_limit")
            for path in by_name:
                if any(
                    parent.as_posix() in by_name and not by_name[parent.as_posix()].is_dir()
                    for parent in PurePosixPath(path).parents
                    if parent.as_posix() != "."
                ):
                    raise AnchorUnavailable("source_invalid")
            grouped: dict[str, list[tuple[int, int, int, str]]] = {}
            for index, location in enumerate(locations):
                file = location.get("file")
                if not isinstance(file, str):
                    raise AnchorUnavailable("selector_unavailable")
                language = _PROFILES.get(PurePosixPath(file).suffix)
                if language is None:
                    raise AnchorUnavailable("anchor_language_unsupported")
                line_range = location.get("range")
                start = location.get("line") if line_range is None else line_range.get("start_line")
                end = start if line_range is None else line_range.get("end_line")
                if type(start) is not int or type(end) is not int or start <= 0 or end < start:
                    raise AnchorUnavailable("selector_unavailable")
                grouped.setdefault(file, []).append((index, start, end, language))
            total = 0
            anchors = [None] * len(locations)
            for file, selectors in grouped.items():
                info = by_name.get(file)
                if info is None or info.is_dir():
                    raise AnchorUnavailable("source_missing")
                if info.file_size > MAX_FILE_BYTES:
                    raise AnchorUnavailable("input_limit")
                with archive.open(info) as stream:
                    source = stream.read(MAX_FILE_BYTES + 1)
                if len(source) != info.file_size or len(source) > MAX_FILE_BYTES:
                    raise AnchorUnavailable("input_limit")
                total += len(source)
                if total > MAX_SOURCE_BYTES:
                    raise AnchorUnavailable("input_limit")
                source.decode("utf-8", errors="strict")
                if b"\0" in source:
                    raise AnchorUnavailable("source_invalid")
                language = selectors[0][3]
                # Publication never initiates a lazy grammar download. Runtime
                # installation/test preparation owns the versioned cache.
                if language not in downloaded_languages():
                    raise AnchorUnavailable("anchor_parser_unavailable")
                try:
                    parser = get_parser(language)
                except ParserLoadError:
                    raise AnchorUnavailable("anchor_parser_unavailable") from None

                def read_chunk(
                    offset: int, _position: object, chunk_source: bytes = source
                ) -> bytes:
                    time.sleep(0)
                    return chunk_source[offset : offset + (8 << 10)]

                tree = parser.parse(read_chunk)
                budget.visit()
                if tree.root_node.has_error:
                    raise AnchorUnavailable("parse_error")
                starts = line_byte_starts(source)
                for index, start, end, language in selectors:
                    anchors[index] = _derive(
                        tree.root_node,
                        source,
                        file,
                        start,
                        end,
                        language,
                        qualify_local,
                        budget,
                        starts,
                    )
                # No AST/source cache crosses files, invocations or publications.
                del tree, source, starts, read_chunk
        result = {
            "schema": DERIVATION_SCHEMA,
            "availability": "available",
            "sourceRef": source_ref,
            "sourceDigest": sha256_digest(data),
            "anchors": anchors,
        }
        if (
            len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
            > MAX_DERIVATION_BYTES
        ):
            raise AnchorUnavailable("input_limit")
        return result
    except AnchorUnavailable as error:
        return unavailable(str(error))
    except (
        zlib.error,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        UnicodeDecodeError,
        OSError,
        NotImplementedError,
    ):
        return unavailable("source_invalid")


def _safe_member(raw: str) -> str:
    path = raw[:-1] if raw.endswith("/") else raw
    if (
        not path
        or len(path.encode()) > 512
        or len(path.split("/")) > 32
        or "\\" in path
        or any(unicodedata.category(c) == "Cc" for c in path)
        or any(p in {"", ".", ".."} or ":" in p for p in path.split("/"))
    ):
        raise AnchorUnavailable("source_invalid")
    return path


def _children(node: Node, budget: _Budget):
    cursor = node.walk()
    if not cursor.goto_first_child():
        return
    while True:
        budget.visit()
        child = cursor.node
        if child.type != "comment" or _semantic_comment(child):
            yield cursor.field_name or "", child
        if not cursor.goto_next_sibling():
            return


def _semantic_comment(node: Node) -> bool:
    return _DIRECTIVE.match(node.text.decode("utf-8")) is not None


def _is_scope(node: Node) -> bool:
    if node.type != "variable_declarator":
        return node.type in _SCOPES
    value = node.child_by_field_name("value")
    for _ in range(8):
        if (
            value is None
            or value.type != "parenthesized_expression"
            or value.named_child_count != 1
        ):
            break
        value = value.named_child(0)
    return value is not None and value.type in {
        "arrow_function",
        "function_expression",
        "generator_function",
    }


def _directive_text(node: Node) -> str | None:
    if node.type == "comment":
        return node.text.decode("utf-8") if _semantic_comment(node) else None
    if node.type in {"hash_bang_line", "hash_bang"}:
        return node.text.decode("utf-8")
    parent = node.parent
    if parent is not None and parent.type in {"program", "module", "block", "statement_block"}:
        if node.type == "string":
            return node.text.decode("utf-8")
        if (
            node.type == "expression_statement"
            and node.named_child_count == 1
            and node.named_child(0).type == "string"
        ):
            return node.text.decode("utf-8")
    return None


def _canonical(node: Node, budget: _Budget, depth: int = 0, *, header: bool = False) -> Any:
    if depth > 64:
        raise AnchorUnavailable("input_limit")
    budget.visit()
    if node.child_count == 0:
        text = node.text.decode("utf-8")
        budget.visit(token_bytes=len(node.text))
        return [node.type, text]
    children = []
    for field, child in _children(node, budget):
        if header and (field == "body" or child.type in {"block", "statement_block", "class_body"}):
            continue
        if (
            header
            and node.type in {"variable_declarator", "decorated_definition"}
            and (field == "value" or _is_scope(child))
        ):
            continue
        children.append([field, _canonical(child, budget, depth + 1)])
    return [node.type, children]


def _derive(
    root: Node,
    source: bytes,
    file: str,
    start: int,
    end: int,
    language: str,
    qualify_local: bool,
    budget: _Budget,
    starts: list[int],
) -> dict[str, Any]:
    selected = []
    directives = []
    # Walk with a cursor: neither root.children nor an unbounded list of nodes
    # is materialized for a syntactically dense file.
    cursor = root.walk()
    depth = 0
    while True:
        node = cursor.node
        budget.visit()
        directive = _directive_text(node)
        if directive is not None:
            text = directive
            if len(text.encode()) > 1024 or len(directives) >= 32:
                raise AnchorUnavailable("input_limit")
            directives.append(text)
        if (
            _is_scope(node)
            and bisect_right(starts, node.start_byte) == start
            and bisect_right(starts, max(node.start_byte, node.end_byte - 1)) == end
        ):
            raise AnchorUnavailable("anchor_ambiguous")
        if (
            node.type in _STATEMENTS[language]
            and (
                language != "python"
                or (node.parent is not None and node.parent.type in {"module", "block"})
            )
            and bisect_right(starts, node.start_byte) == start
            and bisect_right(starts, max(node.start_byte, node.end_byte - 1)) == end
        ):
            selected.append(node)
            if len(selected) > 1:
                raise AnchorUnavailable("anchor_ambiguous")
        if depth < 128 and cursor.goto_first_child():
            depth += 1
            continue
        if depth >= 128 and node.child_count:
            raise AnchorUnavailable("input_limit")
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                break
            depth -= 1
        else:
            continue
        break
    if len(selected) != 1:
        raise AnchorUnavailable("anchor_ambiguous")
    node = selected[0]
    ancestors = []
    path = []
    current = node
    while current.parent is not None:
        parent = current.parent
        siblings = [child for _, child in _children(parent, budget) if child.is_named]
        path.append(next(i for i, sibling in enumerate(siblings) if sibling.id == current.id))
        if _is_scope(parent):
            ancestors.append(_canonical(parent, budget, header=True))
        current = parent
    signature = jcs_digest(["signature/v1", list(reversed(ancestors)), directives])
    node_digest = jcs_digest(_canonical(node, budget))
    result = {
        "file": file,
        "startLine": start,
        "endLine": end,
        "language": language,
        "profile": language + "/statement-v1",
        "grammar": GRAMMAR,
        "fileDigest": sha256_digest(source),
        "signatureDigest": signature,
        "path": list(reversed(path)),
        "nodeDigest": node_digest,
    }
    if qualify_local:
        literal = _private_key_literal(node, language)
        if literal is not None:
            literal_digest = sha256_digest(literal)
            result["context"] = {
                "scheme": "node-local/v1",
                "ruleId": LOCAL_RULE,
                "literalDigest": literal_digest,
                "contextDigest": jcs_digest(
                    ["node-local-private-key/v1", "ed25519", literal_digest]
                ),
            }
    return result


def _private_key_literal(node: Node, language: str) -> bytes | None:
    value = None
    if language == "python" and node.type in {"assignment", "expression_statement"}:
        assignment = node if node.type == "assignment" else node.named_child(0)
        if assignment.type == "assignment":
            left = assignment.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                value = assignment.child_by_field_name("right")
    elif (
        language != "python"
        and node.type in {"lexical_declaration", "variable_declaration"}
        and node.named_child_count == 1
    ):
        declaration = node.named_child(0)
        name = declaration.child_by_field_name("name")
        if (
            declaration.type == "variable_declarator"
            and name is not None
            and name.type == "identifier"
        ):
            value = declaration.child_by_field_name("value")
    if value is None or value.type != "string" or value.end_byte - value.start_byte > 4096:
        return None
    try:
        text = (
            ast.literal_eval(value.text.decode())
            if language == "python"
            else json.loads(value.text)
        )
        if not isinstance(text, str) or len(text) > 512:
            return None
        raw = text.encode("ascii")
        # The first local scheme supports small Ed25519 PKCS8 keys only. This
        # excludes expensive RSA validation and unknown/encrypted formats.
        if (
            re.fullmatch(
                rb"-----BEGIN PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{32,128}"
                rb"-----END PRIVATE KEY-----\r?\n?",
                raw,
            )
            is None
        ):
            return None
        key = load_pem_private_key(raw, password=None)
        return raw if isinstance(key, Ed25519PrivateKey) else None
    except (ValueError, TypeError, SyntaxError, UnicodeError, UnsupportedAlgorithm):
        return None
