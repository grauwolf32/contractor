"""Path-aware glob-to-regex translation shared by sandbox filesystems.

Semantics mirror Python's ``pathlib``-style globbing: ``*`` / ``?`` /
``[...]`` match within a single path segment (never crossing ``/``),
while ``**`` matches any number of segments, including zero.
"""

from __future__ import annotations

import re


def _translate_glob_segment(seg: str) -> str:
    """Translate one glob path segment to regex, never crossing ``/``."""
    out: list[str] = []
    i, n = 0, len(seg)
    while i < n:
        c = seg[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and seg[j] == "!":
                j += 1
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:  # no closing bracket: treat '[' literally
                out.append(re.escape(c))
            else:
                inner = seg[i + 1 : j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append("[" + inner + "]")
                i = j + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """
    Compile a glob pattern into a path-aware regex with Python-like semantics:
    ``*``/``?``/``[...]`` match within a single path segment, while ``**``
    matches any number of segments (including zero). Matches relative paths
    without a leading ``/``.
    """
    segments = pattern.split("/")
    parts: list[str] = []
    last = len(segments) - 1
    for idx, seg in enumerate(segments):
        if seg == "**":
            if idx == last:
                parts.append(".*")  # trailing ** matches anything, any depth
            else:
                parts.append("(?:[^/]*/)*")  # **/ matches zero or more segments
                continue  # the separator is baked into the group above
        else:
            parts.append(_translate_glob_segment(seg))
        if idx != last:
            parts.append("/")
    return re.compile("(?s:" + "".join(parts) + r")\Z")
