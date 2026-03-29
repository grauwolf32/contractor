import json
import pytest

from pathlib import Path

from contractor.tools.openapi.vacuum import (
    ensure_vacuum,
    extract_snippet,
    lint_openapi,
    process_issues,
    replace_range_with_snippet,
)

TEST_BASE_DIR = Path(__file__).parent.parent.parent.parent

def test_lint_openapi_returns_json_for_fakeproj_spec() -> None:
    spec_path = TEST_BASE_DIR / "data" / "fakeproj" / "2" / "openapi.yaml"
    openapi_str = spec_path.read_text(encoding="utf-8")

    result = lint_openapi(openapi_str)

    assert isinstance(result, dict)
    assert "error" not in result

    # Extra guard: result should be JSON-serializable
    json.dumps(result)

def test_extract_snippet_single_line() -> None:
    source = "first line\nsecond line\nthird line"
    snippet = extract_snippet(
        source_text=source,
        start_line=2,
        start_character=0,
        end_line=2,
        end_character=6,
    )
    assert snippet == "second"


def test_extract_snippet_multi_line() -> None:
    source = "alpha\nbeta\ngamma"
    snippet = extract_snippet(
        source_text=source,
        start_line=1,
        start_character=2,
        end_line=3,
        end_character=3,
    )
    assert snippet == "pha\nbeta\ngam"


def test_extract_snippet_returns_empty_string_for_invalid_range() -> None:
    source = "one\ntwo"
    snippet = extract_snippet(
        source_text=source,
        start_line=3,
        start_character=0,
        end_line=3,
        end_character=1,
    )
    assert snippet == ""


def test_replace_range_with_snippet_preserves_all_other_fields() -> None:
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

    result = replace_range_with_snippet(issue=issue, source_text=source)

    assert "range" not in result
    assert result["snippet"] == "title"
    assert result["code"] == issue["code"]
    assert result["path"] == issue["path"]
    assert result["message"] == issue["message"]
    assert result["severity"] == issue["severity"]
    assert result["source"] == issue["source"]
    assert result["extra_field"] == issue["extra_field"]


def test_replace_range_with_snippet_adds_empty_snippet_when_range_missing() -> None:
    issue = {
        "code": "some-rule",
        "path": ["paths", "/users"],
        "message": "Example message",
        "severity": 2,
        "source": "stdin",
    }

    result = replace_range_with_snippet(issue=issue, source_text="anything")

    assert "range" not in result
    assert result["snippet"] == ""
    assert result["code"] == "some-rule"


def test_process_issues_filters_sorts_limits_and_replaces_range() -> None:
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

    result = process_issues(
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


def test_process_issues_keeps_only_selected_severities() -> None:
    issues = [
        {"code": "a", "severity": 0},
        {"code": "b", "severity": 1},
        {"code": "c", "severity": 2},
    ]

    result = process_issues(
        issues=issues,
        source_text="x",
        include_severities=(2,),
        limit=None,
    )

    assert len(result) == 1
    assert result[0]["code"] == "c"
    assert result[0]["severity"] == 2

def test_lint_openapi_processes_vacuum_issues(monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr("contractor.tools.openapi.vacuum.ensure_vacuum", lambda: None)
    monkeypatch.setattr("contractor.tools.openapi.vacuum.subprocess.run", lambda *args, **kwargs: CompletedProcessMock())

    result = lint_openapi(
        source,
        include_severities=(1, 2),
        limit=2,
    )

    assert "result" in result
    assert len(result["result"]) == 2

    first, second = result["result"]

    assert first["severity"] == 2
    assert first["code"] == "high"
    assert first["snippet"] == "line2"
    assert first["custom"] == "keep me"
    assert "range" not in first

    assert second["severity"] == 1
    assert second["code"] == "medium"
    assert second["snippet"] == "line3"
    assert "range" not in second


def test_lint_openapi_on_fakeproj_file_with_mocked_vacuum(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "end": {"line": 1, "character": min(10, len(openapi_str.splitlines()[0]))},
            },
        }
    ]

    class CompletedProcessMock:
        returncode = 1
        stdout = json.dumps(vacuum_output).encode("utf-8")
        stderr = b""

    monkeypatch.setattr("contractor.tools.openapi.vacuum.ensure_vacuum", lambda: None)
    monkeypatch.setattr("contractor.tools.openapi.vacuum.subprocess.run", lambda *args, **kwargs: CompletedProcessMock())

    result = lint_openapi(openapi_str, include_severities=(1, 2), limit=10)

    assert "result" in result
    assert isinstance(result["result"], list)
    assert len(result["result"]) == 1
    assert "snippet" in result["result"][0]
    assert "range" not in result["result"][0]