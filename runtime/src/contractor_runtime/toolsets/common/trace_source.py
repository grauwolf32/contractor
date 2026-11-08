"""Linear insertion-only annotation mapping over an atomic workspace capture."""

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.lines import split_lines
from contractor_runtime.digests import sha256_digest
from contractor_runtime.projectfs.storage import WorkspaceSourceFile
from contractor_runtime.toolsets.common.annotation_syntax import canonical_annotation
from contractor_runtime.toolsets.common.annotation_targets import parse_annotation_targets
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_BYTES as MAX_TRACE_BYTES,
)
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_LINES as MAX_TRACE_LINES,
)
from contractor_runtime.toolsets.common.line_window import (
    bounded_line_window,
)
from contractor_runtime.toolsets.common.source_languages import (
    Language,
    detect_language,
    is_header,
    load_parser,
)


def _validate_insertions(
    capture: WorkspaceSourceFile,
    original: list[str],
    current: list[str],
    inserted: dict[int, int],
) -> None:
    if not inserted:
        return
    language = detect_language(capture.path, header=Language.C)
    if language is None:
        raise ToolInputError("trace finding annotation language cannot be verified")
    # Resolve against the immutable source: an insertion cannot create its own
    # apparent target by changing a literal or preprocessor continuation.
    encoded = capture.original.encode()
    parsed = parse_annotation_targets(load_parser(language), encoded, language)
    # .h admits both C and C++ declarations. Try the other grammar only when
    # C cannot parse the immutable source; comment syntax is identical.
    if parsed.parse_error and is_header(capture.path):
        language = Language.CPP
        parsed = parse_annotation_targets(load_parser(language), encoded, language)
    if parsed.parse_error:
        raise ToolInputError("trace finding annotation syntax cannot be verified")
    slots: dict[int, str] = {}
    for target in parsed.targets:
        index = target.insertion_line - 1
        if not target.insertion_is_line_start or index in slots:
            continue
        declaration = original[index]
        indent = declaration[: len(declaration) - len(declaration.lstrip(" \t"))]
        slots[index] = indent
        while index > 0 and canonical_annotation(
            original[index - 1].removesuffix("\n").removesuffix("\r"), indent, language, index
        ):
            index -= 1
            slots[index] = indent
    for index, base in inserted.items():
        indent = slots.get(base)
        if indent is None or not canonical_annotation(
            current[index].removesuffix("\n").removesuffix("\r"), indent, language, index + 1
        ):
            raise ToolInputError("trace finding requires canonical annotations above declarations")


def trace_source(capture: WorkspaceSourceFile, start: int, end: int) -> dict:
    """Translate current coordinates; preserve the exact annotated evidence window."""
    original = split_lines(capture.original, keepends=True)
    current = split_lines(capture.current, keepends=True)
    if start < 1 or end < start or end > len(current) or end - start >= MAX_TRACE_LINES:
        raise ToolInputError("trace finding coordinates exceed the available source window")
    translated_start = translated_end = None
    base = 0
    inserted: dict[int, int] = {}
    for number, line in enumerate(current, start=1):
        if base < len(original) and line == original[base]:
            base += 1
            translated = base
        else:
            inserted[number - 1] = base
            # Added annotation lines belong to the following original declaration.
            translated = base + 1
        if number == start:
            translated_start = translated
        if number == end:
            translated_end = translated
    if base != len(original) or translated_end is None or translated_end > base:
        raise ToolInputError("trace finding source coordinates are unavailable")
    _validate_insertions(capture, original, current, inserted)
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
