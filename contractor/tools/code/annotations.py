"""Structured annotation tools for the trace worker.

Replaces the generic ``insert_line`` + raw ``# @trace target=...``
pattern with three specialised tool functions that validate
arguments, resolve the function definition via ``CodeTools``
(tree-sitter), and emit a properly-indented comment line above the
function in the right comment style for the file's language.

Tools exposed:
  - ``annotate_trace(file, function, target, args="", calls="")``
  - ``annotate_validate(file, function, arg, kind)``
  - ``annotate_sink(file, function, kind, arg="unknown")``

The on-disk format is identical to what trace_agent v5+ have always
produced, so the harness regex extractor keeps working and prompts
that mention "# @trace target=..." stay accurate. The win is that the
agent stops doing arithmetic with line numbers and can't emit a
malformed annotation.
"""

from __future__ import annotations

import logging
from typing import Any

from fsspec import AbstractFileSystem

from contractor.tools.code.tools import CodeTools, Language, detect_language
from contractor.tools.result import guard

logger = logging.getLogger(__name__)

# Languages where `#` is the line-comment marker. Anything not listed
# here gets the C-family ``//``.
_HASH_COMMENT_LANGS: frozenset[Language] = frozenset(
    {
        Language.PYTHON,
        Language.RUBY,
        Language.BASH,
        Language.ELIXIR,
        Language.PHP,  # PHP supports both `#` and `//`; pick `#` so the
                       # harness regex picks it up regardless.
    }
)

# Argument-state vocabulary lifted from the trace_agent prompt.
_ARG_STATES: frozenset[str] = frozenset(
    {"tainted", "validated", "clean", "derived"}
)


def _comment_prefix_for(file_path: str) -> str:
    lang = detect_language(file_path)
    if lang in _HASH_COMMENT_LANGS:
        return "#"
    return "//"


def _leading_ws(line: str) -> str:
    i = 0
    while i < len(line) and line[i] in (" ", "\t"):
        i += 1
    return line[:i]


def _parse_args_spec(args_spec: str) -> tuple[list[tuple[str, str]], str | None]:
    """Parse ``user:tainted,token:validated`` into pairs. Returns
    (pairs, error) where error is None on success.
    """
    if not args_spec:
        return [], None
    pairs: list[tuple[str, str]] = []
    for chunk in args_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            return [], (
                f"args entry {chunk!r} must use 'name:state' form "
                f"(states: {sorted(_ARG_STATES)})"
            )
        name, state = chunk.split(":", 1)
        name = name.strip()
        state = state.strip()
        if not name:
            return [], f"args entry {chunk!r} has empty name"
        if state not in _ARG_STATES:
            return [], (
                f"args entry {chunk!r}: unknown state {state!r} "
                f"(allowed: {sorted(_ARG_STATES)})"
            )
        pairs.append((name, state))
    return pairs, None


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _existing_annotation_above(lines: list[str], def_line: int) -> str | None:
    """If the line above `def_line` (1-based) is an annotation comment,
    return its trimmed text; else None.
    """
    if def_line < 2:
        return None
    candidate = lines[def_line - 2]
    stripped = candidate.strip()
    if not stripped:
        return None
    if stripped.startswith(("# @", "// @", "#@", "//@")):
        return stripped
    return None


def _build_function_locator(fs: AbstractFileSystem, root: str = "/"):
    """Return a function that resolves (file, function_name) → line.

    Uses ``CodeTools.search_definition`` so we get the same multi-language
    tree-sitter resolution the agent already uses for ``search_def``,
    with a grep fallback when tree-sitter has no spec for the language.
    """
    code_tools = CodeTools(fs=fs, root=root)

    def _locate(file: str, function: str) -> tuple[int | None, str]:
        bare = function.rsplit(".", 1)[-1]
        result = code_tools.search_definition(
            symbol=bare,
            path=file,
            max_results=10,
        )
        # Grep fallback is unsafe here: a call site looks just like a
        # definition to grep and would have us inserting an annotation
        # above an import or invocation. Require a real tree-sitter
        # definition.
        if not result.definitions:
            return None, (
                f"function {function!r} not defined in {file!r}. "
                f"Use `search_def` to find where it is actually "
                f"defined, then re-call with that file."
            )
        # Prefer a definition in the requested file.
        same = [d for d in result.definitions if d.file == file]
        if not same:
            chosen = result.definitions[0]
            return None, (
                f"closest definition for {function!r} is in "
                f"{chosen.file!r}, not {file!r}. Re-call with the right "
                f"file or use that one."
            )
        return same[0].line, ""

    return _locate


def annotation_tools(fs: AbstractFileSystem, *, root: str = "/") -> list:
    """Three annotation tools for the trace worker.

    All three resolve the target function via tree-sitter (same engine
    as ``search_def``) and insert a structured comment line directly
    above the function definition, preserving the def-line indentation
    and using the language-appropriate comment marker. Idempotent: if
    a ``# @trace`` / ``# @validate`` / ``# @sink`` line already sits
    immediately above the function, the call returns an error instead
    of stacking duplicates.
    """
    locate = _build_function_locator(fs, root=root)

    def _insert(
        file: str,
        function: str,
        kind: str,
        body: str,
    ) -> dict[str, Any]:
        line, err = locate(file, function)
        if err:
            return {"error": err, "kind": kind}
        if line is None:
            raise RuntimeError("locate() returned no line despite reporting no error")

        try:
            content = fs.read_text(file, encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            return {"error": f"file {file!r} not found", "kind": kind}
        except Exception as exc:
            return {"error": f"cannot read {file!r}: {exc}", "kind": kind}

        # Preserve original line-ending style.
        newline = "\r\n" if "\r\n" in content else "\n"
        lines = content.splitlines(keepends=False)

        if line < 1 or line > len(lines):
            return {
                "error": f"resolved line {line} is out of range "
                f"(1..{len(lines)})",
                "kind": kind,
            }

        existing = _existing_annotation_above(
            [ln + newline for ln in lines], line
        )
        if existing is not None and f"@{kind} " in existing:
            return {
                "error": f"function {function!r} already has a "
                f"@{kind} annotation",
                "existing": existing,
                "kind": kind,
            }

        def_line_text = lines[line - 1]
        indent = _leading_ws(def_line_text)
        prefix = _comment_prefix_for(file)
        annotation = f"{indent}{prefix} @{kind} {body}".rstrip()

        lines.insert(line - 1, annotation)
        new_content = newline.join(lines) + (newline if content.endswith(("\n", "\r")) else "")

        try:
            fs.write_text(file, new_content, encoding="utf-8")
        except Exception as exc:
            return {"error": f"cannot write {file!r}: {exc}", "kind": kind}

        return {
            "file": file,
            "function": function,
            "kind": kind,
            "annotation_line": line,
            "function_line": line + 1,
            "text": annotation.lstrip(),
        }

    def annotate_trace(
        file: str,
        function: str,
        target: str = "unknown",
        args: str = "",
        calls: str = "",
    ) -> dict[str, Any]:
        """Insert ``@trace target=... args=... calls=...`` above ``function``.

        Use this instead of ``insert_line`` for every trace annotation —
        it picks the comment marker, resolves the function line, checks
        for duplicates, and preserves indentation automatically.

        Arguments
          - file: virtual path of the source file (e.g. /routers/notes.py)
          - function: function or method name. Methods may be passed as
            ``ClassName.method`` or just ``method``.
          - target: target id from the assignment, or "unknown".
          - args: comma-separated ``name:state`` pairs. Allowed states:
            tainted, validated, clean, derived. May be empty.
          - calls: comma-separated callee symbols relevant to the trace.
        """

        def _impl() -> dict[str, Any]:
            if args:
                _, err = _parse_args_spec(args)
                if err:
                    return {"error": err, "kind": "trace"}
            parts = [f"target={target or 'unknown'}"]
            if args:
                parts.append(f"args={args}")
            if calls:
                parts.append(f"calls={','.join(_split_csv(calls))}")
            return _insert(file, function, "trace", " ".join(parts))

        return guard(_impl)

    def annotate_validate(
        file: str,
        function: str,
        arg: str,
        kind: str,
    ) -> dict[str, Any]:
        """Insert ``@validate arg=... kind=...`` above ``function``.

        Use when the function performs validation / sanitisation on a
        tainted argument. ``arg`` names the validated argument; ``kind``
        is a short label of the validation (e.g. ``regex``, ``schema``,
        ``length``, ``allowlist``).
        """

        def _impl() -> dict[str, Any]:
            if not arg:
                return {"error": "arg is required", "kind": "validate"}
            if not kind:
                return {"error": "kind is required", "kind": "validate"}
            body = f"arg={arg} kind={kind}"
            return _insert(file, function, "validate", body)

        return guard(_impl)

    def annotate_sink(
        file: str,
        function: str,
        kind: str,
        arg: str = "unknown",
    ) -> dict[str, Any]:
        """Insert ``@sink kind=... arg=...`` above ``function``.

        Use only when the function directly performs the sink or
        clearly wraps it. ``kind`` is the sink category (e.g. ``sql``,
        ``shell``, ``ssrf``, ``deserialize``, ``open-redirect``).
        """

        def _impl() -> dict[str, Any]:
            if not kind:
                return {"error": "kind is required", "kind": "sink"}
            body = f"kind={kind} arg={arg or 'unknown'}"
            return _insert(file, function, "sink", body)

        return guard(_impl)

    return [annotate_trace, annotate_validate, annotate_sink]


__all__ = ["annotation_tools"]
