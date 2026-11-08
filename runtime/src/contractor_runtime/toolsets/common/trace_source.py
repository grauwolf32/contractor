"""Linear insertion-only annotation mapping over an atomic workspace capture."""

import re

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.lines import split_lines
from contractor_runtime.digests import sha256_digest
from contractor_runtime.projectfs.storage import WorkspaceSourceFile
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_BYTES as MAX_TRACE_BYTES,
)
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_LINES as MAX_TRACE_LINES,
)
from contractor_runtime.toolsets.common.line_window import (
    bounded_line_window,
)

_ANNOTATION = re.compile(r"^[ \t]*(?://|#|--)[ \t]+@(?:trace|validate|sink)[ \t]+[^\r\n]+$")


def trace_source(capture: WorkspaceSourceFile, start: int, end: int) -> dict:
    """Translate current coordinates; preserve the exact annotated evidence window."""
    original = split_lines(capture.original, keepends=True)
    current = split_lines(capture.current, keepends=True)
    if start < 1 or end < start or end > len(current) or end - start >= MAX_TRACE_LINES:
        raise ToolInputError("trace finding coordinates exceed the available source window")
    translated_start = translated_end = None
    base = 0
    for number, line in enumerate(current, start=1):
        if base < len(original) and line == original[base]:
            base += 1
            translated = base
        elif _ANNOTATION.fullmatch(line.rstrip("\r\n")):
            # Added annotation lines belong to the following original declaration.
            translated = base + 1
        else:
            raise ToolInputError(
                "trace finding requires unchanged source plus annotation insertions"
            )
        if number == start:
            translated_start = translated
        if number == end:
            translated_end = translated
    if base != len(original) or translated_end is None or translated_end > base:
        raise ToolInputError("trace finding source coordinates are unavailable")
    # Include nearby annotations/context while keeping one bounded exact window.
    window_start = max(1, start - min(8, MAX_TRACE_LINES - (end - start + 1)))
    text, window_end, partial = bounded_line_window(
        current,
        start_line=window_start,
        max_lines=end - window_start + 1,
        max_bytes=MAX_TRACE_BYTES,
    )
    if partial or window_end < end:
        raise ToolInputError("trace finding evidence exceeds the snippet byte limit")
    return {
        "schema": "contractor.trace-finding-evidence.v1",
        "source": {
            "artifact": capture.source_ref.model_dump(exclude_none=True),
            "file": capture.source_path,
            "startLine": translated_start,
            "endLine": translated_end,
            "digest": sha256_digest(capture.original.encode()),
        },
        "workspace": {
            "file": capture.path,
            "startLine": start,
            "endLine": end,
            "snippetStartLine": window_start,
            "snippetEndLine": window_end,
            "text": text,
            "digest": sha256_digest(capture.current.encode()),
        },
    }
