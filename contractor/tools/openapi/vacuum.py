import subprocess
import json
import shutil
from typing import Any, Dict, Iterable, Optional


def ensure_vacuum() -> Optional[Dict[str, str]]:
    """
    Checks if the `vacuum` binary is available in PATH.

    Returns:
        None if available, otherwise a dict with an error message.
    """
    if shutil.which("vacuum") is None:
        return {"error": "vacuum binary not found in PATH"}
    return None


def extract_snippet(
    source_text: str,
    start_line: int,
    start_character: int,
    end_line: int,
    end_character: int,
) -> str:
    """
    Extracts a snippet from the original source text using Spectral/Vacuum range coordinates.

    Args:
        source_text: Original OpenAPI source text.
        start_line: 1-based start line number.
        start_character: 0-based start character offset.
        end_line: 1-based end line number.
        end_character: 0-based end character offset.

    Returns:
        Extracted snippet, or an empty string if the coordinates are invalid.
    """
    lines = source_text.splitlines()

    if (
        start_line < 1
        or end_line < 1
        or start_line > len(lines)
        or end_line > len(lines)
        or start_line > end_line
    ):
        return ""

    start_idx = start_line - 1
    end_idx = end_line - 1

    if start_idx == end_idx:
        line = lines[start_idx]
        return line[start_character:end_character]

    parts = [lines[start_idx][start_character:]]
    for idx in range(start_idx + 1, end_idx):
        parts.append(lines[idx])
    parts.append(lines[end_idx][:end_character])

    return "\n".join(parts)


def replace_range_with_snippet(
    issue: Dict[str, Any],
    source_text: str,
) -> Dict[str, Any]:
    """
    Returns a copy of the issue where `range` is replaced with `snippet`.
    All other fields are preserved unchanged.

    Args:
        issue: Original Vacuum/Spectral issue object.
        source_text: Original OpenAPI source text.

    Returns:
        A new issue dict with `snippet` instead of `range`.
    """
    result = dict(issue)
    issue_range = result.pop("range", None)

    snippet = ""

    if isinstance(issue_range, dict):
        start = issue_range.get("start")
        end = issue_range.get("end")

        if isinstance(start, dict) and isinstance(end, dict):
            start_line = start.get("line")
            start_character = start.get("character")
            end_line = end.get("line")
            end_character = end.get("character")

            if all(
                isinstance(value, int)
                for value in (start_line, start_character, end_line, end_character)
            ):
                snippet = extract_snippet(
                    source_text=source_text,
                    start_line=start_line,
                    start_character=start_character,
                    end_line=end_line,
                    end_character=end_character,
                )

    result["snippet"] = snippet
    return result


def process_issues(
    issues: list[Dict[str, Any]],
    source_text: str,
    include_severities: Iterable[int] = (1, 2),
    limit: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """
    Filters issues by severity, sorts them in descending severity order,
    limits the result size, and replaces `range` with `snippet`.

    Args:
        issues: Raw Vacuum/Spectral issue list.
        source_text: Original OpenAPI source text.
        include_severities: Severity levels to include. Defaults to (1, 2).
        limit: Maximum number of issues to return.

    Returns:
        Processed issue list.
    """
    allowed = set(include_severities)

    filtered = [
        issue
        for issue in issues
        if isinstance(issue.get("severity"), int) and issue["severity"] in allowed
    ]

    filtered.sort(key=lambda item: item["severity"], reverse=True)

    if limit is not None:
        filtered = filtered[:limit]

    return [
        replace_range_with_snippet(issue=issue, source_text=source_text)
        for issue in filtered
    ]


def lint_openapi(
    openapi_str: str,
    include_severities: Iterable[int] = (1, 2),
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs Vacuum spectral-report on an OpenAPI string via stdin.

    Args:
        openapi_str: OpenAPI specification as YAML or JSON string.
        include_severities: Severity levels to include in output.
            Defaults to (1, 2), excluding severity 0.
        limit: Maximum number of issues to return.

    Returns:
        A dict with processed issues, or an error dict.
    """
    err = ensure_vacuum()
    if err:
        return err

    try:
        process = subprocess.run(
            ["vacuum", "spectral-report", "-i", "-o"],
            input=openapi_str.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return {"error": f"failed to run vacuum: {exc}"}

    if process.returncode not in (0, 1):
        return {
            "error": "vacuum execution failed",
            "details": process.stderr.decode("utf-8", errors="ignore"),
        }

    try:
        parsed = json.loads(process.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "error": "failed to parse vacuum output",
            "details": str(exc),
            "raw_output": process.stdout.decode("utf-8", errors="ignore"),
        }

    if not isinstance(parsed, list):
        return {
            "error": "unexpected vacuum output format",
            "details": f"expected a list of issues, got {type(parsed).__name__}",
            "raw_output": parsed,
        }

    result = process_issues(
        issues=parsed,
        source_text=openapi_str,
        include_severities=include_severities,
        limit=limit,
    )

    return {"result": result}