"""Unified-diff rendering for :class:`MemoryOverlayFileSystem`.

Extracted from ``overlayfs.py``: given the base-vs-overlay-visible entry maps
and two byte-read callbacks, produce ``diff -U``-style text. The function holds
no filesystem state of its own — the overlay computes the entry maps under its
lock and delegates here, so this stays a pure, independently-testable renderer.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from typing import Any


def _is_text(data: bytes) -> bool:
    """Heuristic: text if it decodes as UTF-8 and has no null bytes."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except (UnicodeDecodeError, ValueError):
        return False


def _lines(data: bytes) -> list[str]:
    """Split bytes into newline-terminated text lines for difflib."""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines(True)
    return [ln if ln.endswith("\n") else ln + "\n" for ln in lines]


def render_overlay_diff(
    *,
    base_entries: Mapping[str, Any],
    visible_entries: Mapping[str, Any],
    root_marker: str,
    read_base_bytes: Callable[[str], bytes],
    read_effective_bytes: Callable[[str], bytes],
    context_lines: int = 3,
    binary_marker: str = "Binary files differ",
) -> str:
    """Render a unified-diff-like text of base vs. overlay-visible state.

    ``base_entries`` / ``visible_entries`` map path → info dict (``type`` key).
    ``read_base_bytes`` / ``read_effective_bytes`` read a path's bytes from the
    base filesystem and the effective (overlay) view respectively. Returns the
    empty string when there are no differences.
    """
    base_paths = set(base_entries)
    visible_paths = set(visible_entries)
    output_lines: list[str] = []
    all_paths = sorted(base_paths | visible_paths)

    _base_cache: dict[str, bytes] = {}

    def _read_base_cached(p: str) -> bytes:
        if p not in _base_cache:
            _base_cache[p] = read_base_bytes(p)
        return _base_cache[p]

    def _emit_diff_header(path: str, status: str) -> None:
        output_lines.append(f"diff --overlay a{path} b{path}")
        output_lines.append(f"{status}")

    for path in all_paths:
        if path == root_marker:
            continue

        in_base = path in base_paths
        in_visible = path in visible_paths

        base_info = base_entries.get(path, {})
        visible_info = visible_entries.get(path, {})

        base_type = base_info.get("type", "file")
        visible_type = visible_info.get("type", "file")

        # --- Deleted paths ---------------------------------------------------
        if in_base and not in_visible:
            if base_type == "directory":
                _emit_diff_header(path, "deleted directory")
                output_lines.append("")
                continue

            _emit_diff_header(path, "deleted file")
            try:
                base_bytes = _read_base_cached(path)
            except FileNotFoundError:
                output_lines.append("")
                continue

            if not _is_text(base_bytes):
                output_lines.append(f"--- a{path}")
                output_lines.append("+++ /dev/null")
                output_lines.append(binary_marker)
                output_lines.append("")
                continue

            base_lines = _lines(base_bytes)
            diff_result = difflib.unified_diff(
                base_lines,
                [],
                fromfile=f"a{path}",
                tofile="/dev/null",
                n=context_lines,
            )
            output_lines.extend(line.rstrip("\n") for line in diff_result)
            output_lines.append("")
            continue

        # --- New paths -------------------------------------------------------
        if not in_base and in_visible:
            if visible_type == "directory":
                _emit_diff_header(path, "new directory")
                output_lines.append("")
                continue

            _emit_diff_header(path, "new file")
            try:
                current_bytes = read_effective_bytes(path)
            except FileNotFoundError:
                output_lines.append("")
                continue

            if not _is_text(current_bytes):
                output_lines.append("--- /dev/null")
                output_lines.append(f"+++ b{path}")
                output_lines.append(binary_marker)
                output_lines.append("")
                continue

            current_lines = _lines(current_bytes)
            diff_result = difflib.unified_diff(
                [],
                current_lines,
                fromfile="/dev/null",
                tofile=f"b{path}",
                n=context_lines,
            )
            output_lines.extend(line.rstrip("\n") for line in diff_result)
            output_lines.append("")
            continue

        # --- Both exist – check for modifications ----------------------------
        if base_type != visible_type:
            _emit_diff_header(path, f"type changed: {base_type} -> {visible_type}")
            output_lines.append("")
            continue

        if visible_type == "directory":
            continue

        try:
            base_bytes = _read_base_cached(path)
        except FileNotFoundError:
            base_bytes = b""

        try:
            current_bytes = read_effective_bytes(path)
        except FileNotFoundError:
            current_bytes = b""

        if base_bytes == current_bytes:
            continue

        _emit_diff_header(path, "modified file")

        if not _is_text(base_bytes) or not _is_text(current_bytes):
            output_lines.append(f"--- a{path}")
            output_lines.append(f"+++ b{path}")
            output_lines.append(binary_marker)
            output_lines.append("")
            continue

        base_lines = _lines(base_bytes)
        current_lines = _lines(current_bytes)
        diff_result = difflib.unified_diff(
            base_lines,
            current_lines,
            fromfile=f"a{path}",
            tofile=f"b{path}",
            n=context_lines,
        )
        output_lines.extend(line.rstrip("\n") for line in diff_result)
        output_lines.append("")

    # Strip trailing blank lines but keep one final newline.
    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    if not output_lines:
        return ""

    return "\n".join(output_lines) + "\n"
