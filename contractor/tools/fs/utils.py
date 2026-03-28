import fnmatch

from typing import Optional, Any
from urllib.parse import quote as url_quote

from contractor.tools.fs.const import (
    _COMMENT_PREFIX_BY_EXT,
)

from contractor.utils.formatting import normalize_slashes


def _project_id_encoded(project_id: str) -> str:
    if project_id.isdigit():
        return project_id
    return url_quote(project_id, safe="")


def _is_ignored(path: str, patterns: list[str]) -> bool:
    normalized = normalize_slashes(path)
    basename = normalized.split("/")[-1]
    return any(
        fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern)
        for pattern in patterns
    )


def _ensure_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _split_lines_keepends(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines(keepends=True)


def _line_ending_for_text(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _comment_prefix_for_path(path: str, comment_style: Optional[str] = None) -> str:
    if comment_style:
        return comment_style

    normalized = normalize_slashes(path)
    basename = normalized.split("/")[-1].lower()

    if basename in {"dockerfile", "makefile"}:
        return "#"

    for ext, prefix in _COMMENT_PREFIX_BY_EXT.items():
        if basename.endswith(ext):
            return prefix

    return "#"


def _format_comment_line(
    *,
    comment: str,
    indent: str,
    prefix: str,
    newline: str,
) -> str:
    stripped = comment.strip()
    if prefix in {"<!--", "<!-- -->"}:
        return f"{indent}<!-- {stripped} -->{newline}"
    return f"{indent}{prefix} {stripped}{newline}"
