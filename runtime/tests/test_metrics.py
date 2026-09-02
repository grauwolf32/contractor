from __future__ import annotations

import json
from types import SimpleNamespace

from contractor_runtime.metrics import (
    MAX_ARGUMENT_SUMMARY_BYTES,
    MAX_METRIC_COUNTER,
    MAX_METRIC_ERRORS,
    MAX_METRIC_TOOL_CALLS,
    MAX_REPORT_JSON_BYTES,
    MetricsState,
)

SECRET = "metrics-secret-that-must-not-survive"


class ToolFailure(RuntimeError):
    code = "synthetic_tool_failure"
    retryable = True


class SkillDisclosureFailure(RuntimeError):
    code = "SKILL_DISCLOSURE_LIMIT"
    retryable = False


def test_report_keeps_exact_aggregates_while_dropping_oldest_bounded_detail() -> None:
    state = MetricsState()
    for index in range(MAX_METRIC_TOOL_CALLS + 5):
        error = ToolFailure(f"provider leaked {SECRET}") if index % 5 == 0 else None
        state.record_tool_call(
            "probe",
            arguments={
                "index": index,
                "value": f"{index}-" + ("x" * 3500),
                "authorization": f"Bearer {SECRET}",
            },
            result={"body": SECRET, "count": index},
            error=error,
            secrets=(SECRET,),
            duration_ms=index,
        )
    state.record_model_call()
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            total_token_count=18,
            cached_content_token_count=3,
        )
    )

    assert len(state.tool_calls) == MAX_METRIC_TOOL_CALLS
    assert state.tool_calls[0].call_id == "tool-00000006"
    assert len(state.errors) == MAX_METRIC_ERRORS
    assert state.truncated

    report = state.build_report(report_id="worker-allocation-1", duration_ms=1234)
    encoded = report.model_dump_json(by_alias=True, exclude_none=True).encode()
    probe = report.metrics.tools["probe"]

    assert probe.calls == MAX_METRIC_TOOL_CALLS + 5
    assert probe.failed == 201
    assert probe.succeeded == 804
    assert report.metrics.model_calls == 1
    assert report.metrics.input_tokens == 11
    assert report.metrics.output_tokens == 7
    assert report.metrics.total_tokens == 18
    assert len(report.tool_calls) < MAX_METRIC_TOOL_CALLS
    assert report.tool_calls[-1].call_id == "tool-00001005"
    assert report.truncated
    assert len(encoded) <= MAX_REPORT_JSON_BYTES
    assert SECRET.encode() not in encoded
    assert all(
        len(json.dumps(call.arguments, separators=(",", ":")).encode())
        <= MAX_ARGUMENT_SUMMARY_BYTES
        for call in report.tool_calls
        if call.arguments is not None
    )


def test_argument_summary_redacts_nested_secrets_urls_and_artifact_bytes() -> None:
    state = MetricsState()
    state.record_tool_call(
        "write_artifact",
        arguments={
            "nested": {
                "token": SECRET,
                "accessToken": "opaque credential",
                "callback": "https://user:password@example.test/path?api_key=secret",
            },
            "data_base64": SECRET * 1000,
            "contentBase64": "encoded artifact bytes",
        },
        result={"dataBase64": SECRET, "size": 99},
        secrets=(SECRET,),
    )

    report = state.build_report(report_id="worker-allocation-2", duration_ms=1)
    encoded = report.model_dump_json(by_alias=True, exclude_none=True)
    arguments = report.tool_calls[0].arguments

    assert arguments is not None
    assert arguments["nested"] == {
        "token": {"redacted": True, "size": len(SECRET)},
        "accessToken": {"redacted": True, "size": len("opaque credential")},
        "callback": "[REDACTED_URL]",
    }
    assert arguments["data_base64"] == {
        "redacted": True,
        "size": len(SECRET) * 1000,
    }
    assert arguments["contentBase64"] == {
        "redacted": True,
        "size": len("encoded artifact bytes"),
    }
    assert "dataBase64" not in encoded
    assert SECRET not in encoded


def test_skill_metric_projects_only_validated_identity_outcome_and_result_size() -> None:
    state = MetricsState()
    state.record_tool_call(
        "load_skill_resource",
        arguments={"skill_name": "likec4", "file_path": "references/syntax.md"},
        result_size_bytes=123_456,
        duration_ms=7,
    )
    state.record_tool_call(
        "load_skill_resource",
        arguments={"arguments_valid": False},
        error=SkillDisclosureFailure("content must not be retained"),
        result_size_bytes=96,
        duration_ms=1,
    )

    report = state.build_report(report_id="worker-skills", duration_ms=8)
    encoded = report.model_dump_json(by_alias=True, exclude_none=True)
    first, second = report.tool_calls
    assert first.arguments == {
        "skill_name": "likec4",
        "file_path": "references/syntax.md",
    }
    assert first.result_size_bytes == 123_456
    assert second.arguments == {"arguments_valid": False}
    assert second.error is not None
    assert second.error.code == "SKILL_DISCLOSURE_LIMIT"
    assert second.result_size_bytes == 96
    assert "content must not be retained" not in encoded


def test_allocation_counters_saturate_at_the_state_wire_limit() -> None:
    state = MetricsState()
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=MAX_METRIC_COUNTER + 10,
            candidates_token_count=MAX_METRIC_COUNTER + 20,
            total_token_count=MAX_METRIC_COUNTER + 30,
            cached_content_token_count=MAX_METRIC_COUNTER + 40,
        )
    )
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=1,
            cached_content_token_count=1,
        )
    )

    assert state.counters == {
        "cached_input_tokens": MAX_METRIC_COUNTER,
        "input_tokens": MAX_METRIC_COUNTER,
        "output_tokens": MAX_METRIC_COUNTER,
        "total_tokens": MAX_METRIC_COUNTER,
    }
