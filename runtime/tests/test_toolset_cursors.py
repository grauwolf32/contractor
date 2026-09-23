"""Pin the wire format of signed toolset cursors and symbol IDs.

Cursors are opaque to models but are handed back verbatim across tool calls,
so every toolset must keep producing byte-identical values.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from contractor_runtime.projectfs.storage import WorkspaceSnapshot
from contractor_runtime.toolsets.code_analysis import ids
from contractor_runtime.toolsets.code_analysis.tools import CodeAnalysisError, _CodeAnalysisSession
from contractor_runtime.toolsets.filesystem.tools import FilesystemToolError, _FilesystemSession
from contractor_runtime.toolsets.workspace_changes.tools import _ChangesSession

KEY = bytes(range(32))
LS_QUERY = "sha256:f1ae7bfcbbe2ad4b62702b314c4f8068cad24b781b48c68b3bf30db6433333be"
DIFF_QUERY = "sha256:7db852e846d75817bbf5946e81b6d385deb9aff2dbbd9c7eb4584960f8017c5f"
SEARCH_QUERY = "sha256:deff26f290c0247a5235f161dc19f08fb30b64d1d112cf608333647c89bbf278"
FILESYSTEM_CURSOR = (
    "eyJsaW5lIjo3LCJvZmZzZXQiOjEwMCwicXVlcnkiOiJzaGEyNTY6ZjFhZTdiZmNiYmUyYWQ0YjYyNzAyYjMxNGM0Zjgw"
    "NjhjYWQyNGI3ODFiNDhjNjhiM2JmMzBkYjY0MzMzMzNiZSIsInNuYXBzaG90Ijoic2hhMjU2OjAzZDRkYjQ1ZjkxNWI0"
    "NWVhZDU1ODhhM2EyOWYyYjkyMzBkMmFlMjRkZTI4OWNmM2IyNzRmNDRlNGI3OGFkNmQifQ"
    ".y12T3WeETQG8MkDbW9bqHzRp29k7bTd3YBBNbDPolyk"
)
CHANGES_CURSOR = (
    "eyJmaW5nZXJwcmludCI6InNoYTI1NjpiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJi"
    "YmJiYmJiYmJiYmJiYmJiYmJiIiwib2Zmc2V0Ijo0MiwicXVlcnkiOiJzaGEyNTY6N2RiODUyZTg0NmQ3NTgxN2JiZjU5"
    "NDZlODFiNmQzODVkZWI5YWZmMmRiYmQ5YzdlYjQ1ODQ5NjBmODAxN2M1ZiJ9"
    ".0xD7AgQ5SQqgozYYGUJ5URCGtXXfNZuywsb8zsBgNbo"
)
CODE_ANALYSIS_CURSOR = (
    "eyJvZmZzZXQiOjIwMCwib3BlcmF0aW9uIjoic2VhcmNoX2RlZiIsInF1ZXJ5Ijoic2hhMjU2OmRlZmYyNmYyOTBjMDI0"
    "N2E1MjM1ZjE2MWRjMTlmMDhmYjMwYjY0ZDFkMTEyY2Y2MDgzMzM2NDdjODliYmYyNzgiLCJzbmFwc2hvdCI6InNoYTI1"
    "NjpjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjIn0"
    ".lruvBMGQWYFTpiS1pjXrNaSgaw2iqCUyNEWPXUjVFuU"
)
SYMBOL_ID = (
    "cas1.eyJiaW5kaW5nIjoiQkJLNEZ1ck9QVDB2aHlaaTQ5Ny1EZHVJLTM2SEdVN3V6bGhQWlVrcm1NZyIsImRpZ2VzdCI6"
    "InNoYTI1NjpkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRk"
    "IiwiaW5kZXgiOjN9.-hZLDTwt0ltXjsyIKCXSNkWLhqR-kLAaklQXfLdZbAk"
)
SNAPSHOT = WorkspaceSnapshot(
    directories=(), files=(), binary_paths=("bin/a.png",), digest="sha256:" + "a" * 64
)


def _tampered(cursor: str) -> list[str]:
    body, signature = cursor.split(".")
    return [
        "",
        cursor + "=",
        cursor + ".",
        body,
        f"{body}.{signature[:-1]}A",
        f"{body[:-1]}A.{signature}",
        f"{body}=.{signature}",
        "x" * 2049,
    ]


def test_filesystem_cursor_wire_format_is_stable() -> None:
    session = SimpleNamespace(_cursor_key=bytearray(KEY))
    cursor = _FilesystemSession._encode_cursor(session, SNAPSHOT, LS_QUERY, 100, 7)
    assert cursor == FILESYSTEM_CURSOR
    decoded = _FilesystemSession._decode_cursor(session, cursor)
    assert (decoded.query, decoded.offset, decoded.line) == (LS_QUERY, 100, 7)
    for invalid in _tampered(cursor):
        with pytest.raises(FilesystemToolError, match="workspace_cursor_invalid"):
            _FilesystemSession._decode_cursor(session, invalid)
    with pytest.raises(FilesystemToolError, match="workspace_cursor_invalid"):
        _FilesystemSession._decode_cursor(SimpleNamespace(_cursor_key=bytearray(32)), cursor)


def test_workspace_changes_cursor_wire_format_is_stable() -> None:
    session = SimpleNamespace(_key=bytearray(KEY))
    fingerprint = "sha256:" + "b" * 64
    cursor = _ChangesSession._encode_cursor(session, DIFF_QUERY, fingerprint, 42)
    assert cursor == CHANGES_CURSOR
    assert _ChangesSession._decode_cursor(session, cursor, DIFF_QUERY, fingerprint) == 42
    for invalid in _tampered(cursor):
        with pytest.raises(FilesystemToolError, match="workspace_cursor_invalid"):
            _ChangesSession._decode_cursor(session, invalid, DIFF_QUERY, fingerprint)
    for query, other in [(LS_QUERY, fingerprint), (DIFF_QUERY, "sha256:" + "c" * 64)]:
        with pytest.raises(FilesystemToolError, match="workspace_cursor_invalid"):
            _ChangesSession._decode_cursor(session, cursor, query, other)


def test_code_analysis_cursor_wire_format_is_stable() -> None:
    session = SimpleNamespace(_cursor_key=bytearray(KEY))
    snapshot = "sha256:" + "c" * 64
    cursor = _CodeAnalysisSession._encode_cursor(session, snapshot, "search_def", SEARCH_QUERY, 200)
    assert cursor == CODE_ANALYSIS_CURSOR
    decoded = _CodeAnalysisSession._decode_cursor(session, cursor)
    assert (decoded.snapshot, decoded.operation, decoded.query, decoded.offset) == (
        snapshot,
        "search_def",
        SEARCH_QUERY,
        200,
    )
    for invalid in _tampered(cursor):
        with pytest.raises(CodeAnalysisError, match="code_analysis_cursor_invalid"):
            _CodeAnalysisSession._decode_cursor(session, invalid)


def test_symbol_id_wire_format_is_stable() -> None:
    digest = "sha256:" + "d" * 64
    assert ids.encode_symbol_id(KEY, digest, 3, "module:fn") == SYMBOL_ID
    decoded = ids.decode_symbol_id(KEY, SYMBOL_ID)
    assert (decoded.snapshot_digest, decoded.index) == (digest, 3)
    assert ids.symbol_id_matches_upstream(KEY, decoded, "module:fn")
    assert ids.encode_symbol_key(KEY) == "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
    assert ids.decode_symbol_key("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8") == KEY
    for invalid in ["AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh9", "AAEC+w", "AAECé"]:
        with pytest.raises(ValueError):
            ids.decode_symbol_key(invalid)
