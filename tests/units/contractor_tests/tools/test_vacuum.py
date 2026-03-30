import json
import pytest

from pathlib import Path
from unittest.mock import patch

from contractor.tools.openapi.vacuum import (
    OpenApiLinter,
    OpenApiLinterError,
    VacuumNotFoundError,
    VacuumExecutionError,
    VacuumOutputError,
    openapi_linter_tools,
)

TEST_BASE_DIR = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def mock_vacuum_in_path(monkeypatch: pytest.MonkeyPatch):
    """Patch shutil.which so OpenApiLinter believes vacuum is installed."""
    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.shutil.which",
        lambda name: "/usr/local/bin/vacuum" if name == "vacuum" else None,
    )


@pytest.fixture
def linter(mock_vacuum_in_path) -> OpenApiLinter:
    """Create an OpenApiLinter instance with vacuum availability mocked."""
    return OpenApiLinter(name="test")


def test_lint_returns_list_for_fakeproj_spec(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = TEST_BASE_DIR / "data" / "fakeproj" / "2" / "openapi.yaml"
    openapi_str = spec_path.read_text(encoding="utf-8")

    vacuum_output = [
        {
            "code": "oas3-parameter-description",
            "path": ["paths", "/example", "get", "parameters"],
            "message": "parameter does not contain a description",
            "severity": 1,
            "source": "stdin",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {
                    "line": 1,
                    "character": min(10, len(openapi_str.splitlines()[0])),
                },
            },
        }
    ]

    class CompletedProcessMock:
        returncode = 1
        stdout = json.dumps(vacuum_output).encode("utf-8")
        stderr = b""

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: CompletedProcessMock(),
    )

    result = linter.lint(openapi_str)

    assert isinstance(result, list)

    # Extra guard: result should be JSON-serializable
    json.dumps(result)


def test_extract_snippet_single_line(linter: OpenApiLinter) -> None:
    source = "first line\nsecond line\nthird line"
    snippet = linter.extract_snippet(
        source_text=source,
        start_line=2,
        start_character=0,
        end_line=2,
        end_character=6,
    )
    assert snippet == "second"


def test_extract_snippet_multi_line(linter: OpenApiLinter) -> None:
    source = "alpha\nbeta\ngamma"
    snippet = linter.extract_snippet(
        source_text=source,
        start_line=1,
        start_character=2,
        end_line=3,
        end_character=3,
    )
    assert snippet == "pha\nbeta\ngam"


def test_extract_snippet_returns_empty_string_for_invalid_range(
    linter: OpenApiLinter,
) -> None:
    source = "one\ntwo"
    snippet = linter.extract_snippet(
        source_text=source,
        start_line=3,
        start_character=0,
        end_line=3,
        end_character=1,
    )
    assert snippet == ""


def test_replace_range_with_snippet_preserves_all_other_fields(
    linter: OpenApiLinter,
) -> None:
    source = "openapi: 3.0.0\ninfo:\n  title: Demo\n"
    issue = {
        "code": "some-rule",
        "path": ["info", "title"],
        "message": "Example message",
        "severity": 1,
        "source": "stdin",
        "extra_field": {"nested": True},
        "range": {
            "start": {"line": 3, "character": 2},
            "end": {"line": 3, "character": 7},
        },
    }

    result = linter.replace_range_with_snippet(issue=issue, source_text=source)

    assert "range" not in result
    assert result["snippet"] == "title"
    assert result["code"] == issue["code"]
    assert result["path"] == issue["path"]
    assert result["message"] == issue["message"]
    assert result["severity"] == issue["severity"]
    assert result["source"] == issue["source"]
    assert result["extra_field"] == issue["extra_field"]


def test_replace_range_with_snippet_adds_empty_snippet_when_range_missing(
    linter: OpenApiLinter,
) -> None:
    issue = {
        "code": "some-rule",
        "path": ["paths", "/users"],
        "message": "Example message",
        "severity": 2,
        "source": "stdin",
    }

    result = linter.replace_range_with_snippet(issue=issue, source_text="anything")

    assert "range" not in result
    assert result["snippet"] == ""
    assert result["code"] == "some-rule"


def test_process_issues_filters_sorts_limits_and_replaces_range(
    linter: OpenApiLinter,
) -> None:
    source = "aaa\nbbb\nccc\nddd\n"
    issues = [
        {
            "code": "low",
            "path": ["a"],
            "message": "low",
            "severity": 0,
            "source": "stdin",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 3},
            },
        },
        {
            "code": "medium",
            "path": ["b"],
            "message": "medium",
            "severity": 1,
            "source": "stdin",
            "range": {
                "start": {"line": 2, "character": 0},
                "end": {"line": 2, "character": 3},
            },
        },
        {
            "code": "high",
            "path": ["c"],
            "message": "high",
            "severity": 2,
            "source": "stdin",
            "range": {
                "start": {"line": 3, "character": 0},
                "end": {"line": 3, "character": 3},
            },
        },
    ]

    result = linter.process_issues(
        issues=issues,
        source_text=source,
        include_severities=(1, 2),
        limit=1,
    )

    assert len(result) == 1
    assert result[0]["severity"] == 2
    assert result[0]["code"] == "high"
    assert result[0]["snippet"] == "ccc"
    assert "range" not in result[0]


def test_process_issues_keeps_only_selected_severities(
    linter: OpenApiLinter,
) -> None:
    issues = [
        {"code": "a", "severity": 0},
        {"code": "b", "severity": 1},
        {"code": "c", "severity": 2},
    ]

    result = linter.process_issues(
        issues=issues,
        source_text="x",
        include_severities=(2,),
        limit=None,
    )

    assert len(result) == 1
    assert result[0]["code"] == "c"
    assert result[0]["severity"] == 2


def test_lint_processes_vacuum_issues(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "line1\nline2\nline3\n"
    vacuum_output = [
        {
            "code": "low",
            "path": ["a"],
            "message": "low severity",
            "severity": 0,
            "source": "stdin",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 5},
            },
        },
        {
            "code": "high",
            "path": ["b"],
            "message": "high severity",
            "severity": 2,
            "source": "stdin",
            "custom": "keep me",
            "range": {
                "start": {"line": 2, "character": 0},
                "end": {"line": 2, "character": 5},
            },
        },
        {
            "code": "medium",
            "path": ["c"],
            "message": "medium severity",
            "severity": 1,
            "source": "stdin",
            "range": {
                "start": {"line": 3, "character": 0},
                "end": {"line": 3, "character": 5},
            },
        },
    ]

    class CompletedProcessMock:
        returncode = 1
        stdout = json.dumps(vacuum_output).encode("utf-8")
        stderr = b""

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: CompletedProcessMock(),
    )

    result = linter.lint(
        source,
        include_severities=(1, 2),
        limit=2,
    )

    assert isinstance(result, list)
    assert len(result) == 2

    first, second = result

    assert first["severity"] == 2
    assert first["code"] == "high"
    assert first["snippet"] == "line2"
    assert first["custom"] == "keep me"
    assert "range" not in first

    assert second["severity"] == 1
    assert second["code"] == "medium"
    assert second["snippet"] == "line3"
    assert "range" not in second


def test_lint_on_fakeproj_file_with_mocked_vacuum(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = TEST_BASE_DIR / "data" / "fakeproj" / "2" / "openapi.yaml"
    openapi_str = spec_path.read_text(encoding="utf-8")

    vacuum_output = [
        {
            "code": "oas3-parameter-description",
            "path": ["paths", "/example", "get", "parameters"],
            "message": "parameter does not contain a description",
            "severity": 1,
            "source": "stdin",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {
                    "line": 1,
                    "character": min(10, len(openapi_str.splitlines()[0])),
                },
            },
        }
    ]

    class CompletedProcessMock:
        returncode = 1
        stdout = json.dumps(vacuum_output).encode("utf-8")
        stderr = b""

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: CompletedProcessMock(),
    )

    result = linter.lint(openapi_str, include_severities=(1, 2), limit=10)

    assert isinstance(result, list)
    assert len(result) == 1
    assert "snippet" in result[0]
    assert "range" not in result[0]


def test_vacuum_not_found_raises_error() -> None:
    with pytest.raises(VacuumNotFoundError, match="vacuum binary not found"):
        OpenApiLinter(name="test")


def test_openapi_linter_tools_returns_error_when_vacuum_missing() -> None:
    tools = openapi_linter_tools(name="test")
    assert len(tools) == 1
    # The returned function should be an async function that returns an error
    assert callable(tools[0])


def test_lint_raises_on_vacuum_execution_failure(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcessMock:
        returncode = 2
        stdout = b""
        stderr = b"something went wrong"

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: FailedProcessMock(),
    )

    with pytest.raises(VacuumExecutionError, match="vacuum execution failed"):
        linter.lint("openapi: 3.0.0")


def test_lint_raises_on_invalid_json_output(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadOutputProcessMock:
        returncode = 0
        stdout = b"not valid json"
        stderr = b""

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: BadOutputProcessMock(),
    )

    with pytest.raises(VacuumOutputError, match="failed to parse vacuum output"):
        linter.lint("openapi: 3.0.0")


def test_lint_raises_on_non_list_output(
    linter: OpenApiLinter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DictOutputProcessMock:
        returncode = 0
        stdout = json.dumps({"unexpected": "format"}).encode("utf-8")
        stderr = b""

    monkeypatch.setattr(
        "contractor.tools.openapi.vacuum.subprocess.run",
        lambda *_args, **_kwargs: DictOutputProcessMock(),
    )

    with pytest.raises(VacuumOutputError, match="unexpected vacuum output format"):
        linter.lint("openapi: 3.0.0")