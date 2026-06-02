"""Local SQLite store for line-anchored review comments.

The explorer lets you leave GitHub-style comments on individual source lines of
a prompt / task / skill version. They are personal scratch notes, so they live
in a single SQLite file under the repo's ``.contractor/`` working dir (the same
place runs write artifacts) — not in git, not in the package.

A comment is anchored to ``(kind, target_id, version, line_start..line_end)``.
``version`` distinguishes prompt versions and skill index/reference files, so a
note pinned to line 40 of ``v7`` does not bleed onto ``v6`` where line 40 is
something else. Each operation opens its own connection (the HTTP server is
threaded) — cheap for a local single-user tool and side-steps cross-thread
sharing rules.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# analytics_ui/comments.py -> parents[1] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = _REPO_ROOT / ".contractor" / "explorer.db"

VALID_KINDS = {"agent", "task", "skill"}
_MAX_BODY = 20_000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            kind       TEXT NOT NULL,
            target_id  TEXT NOT NULL,
            version    TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end   INTEGER NOT NULL,
            body       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_target "
        "ON comments (kind, target_id, version)"
    )
    return conn


def _row(r: sqlite3.Row) -> dict[str, Any]:
    return {k: r[k] for k in r}


class CommentError(ValueError):
    """Raised on invalid comment input (→ HTTP 400)."""


def _validate(kind: str, line_start: Any, line_end: Any, body: Any) -> tuple[int, int, str]:
    if kind not in VALID_KINDS:
        raise CommentError(f"invalid kind: {kind!r}")
    try:
        ls, le = int(line_start), int(line_end)
    except (TypeError, ValueError):
        raise CommentError("line_start/line_end must be integers") from None
    if ls < 1 or le < ls:
        raise CommentError("require 1 <= line_start <= line_end")
    text = (body or "").strip()
    if not text:
        raise CommentError("comment body is empty")
    return ls, le, text[:_MAX_BODY]


def list_comments(
    kind: str | None = None,
    target_id: str | None = None,
    version: str | None = None,
) -> list[dict[str, Any]]:
    where, params = [], []
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if target_id:
        where.append("target_id = ?")
        params.append(target_id)
    if version:
        where.append("version = ?")
        params.append(version)
    sql = "SELECT * FROM comments"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY line_start, id"
    with _connect() as conn:
        return [_row(r) for r in conn.execute(sql, params).fetchall()]


def counts_by_target() -> dict[str, int]:
    """{"kind/target_id/version": count} — for nav/badge hints."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kind, target_id, version, COUNT(*) c FROM comments "
            "GROUP BY kind, target_id, version"
        ).fetchall()
    return {f"{r['kind']}/{r['target_id']}/{r['version']}": r["c"] for r in rows}


def add_comment(
    *, kind: str, target_id: str, version: str,
    line_start: Any, line_end: Any, body: str,
) -> dict[str, Any]:
    ls, le, text = _validate(kind, line_start, line_end, body)
    if not target_id or not version:
        raise CommentError("target_id and version are required")
    ts = _now()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO comments "
            "(kind, target_id, version, line_start, line_end, body, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (kind, target_id, version, ls, le, text, ts, ts),
        )
        new_id = cur.lastrowid
        row = conn.execute("SELECT * FROM comments WHERE id = ?", (new_id,)).fetchone()
    return _row(row)


def update_comment(comment_id: int, body: str) -> dict[str, Any] | None:
    text = (body or "").strip()
    if not text:
        raise CommentError("comment body is empty")
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE comments SET body = ?, updated_at = ? WHERE id = ?",
            (text[:_MAX_BODY], _now(), comment_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM comments WHERE id = ?", (comment_id,)).fetchone()
    return _row(row)


def delete_comment(comment_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
        return cur.rowcount > 0
