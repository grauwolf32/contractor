from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes.private_codec import encode_private
from pydantic import ValidationError

from contractor_runtime.contracts import (
    MAX_EXACT_INTEGER,
    ExecutionReport,
    TerminationError,
    ToolCallRecord,
    WorkerCompletionDiagnostics,
)
from contractor_runtime.contracts.base import (
    ARGUMENT_NUMBER_MARKER,
    MAX_ARGUMENT_NUMBER_EXPONENT,
    MIN_ARGUMENT_NUMBER_EXPONENT,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.telemetry import metrics as metrics_module
from contractor_runtime.telemetry.metrics import (
    MAX_ARGUMENT_SUMMARY_BYTES,
    MAX_METRIC_COUNTER,
    MAX_METRIC_ERRORS,
    MAX_METRIC_TOOL_CALLS,
    MAX_REPORT_JSON_BYTES,
    SAFE_DERIVED_SIZE_METRIC_KEYS,
    MetricsState,
    trim_report_detail,
)
from contractor_runtime.toolsets.audit_results.arguments import AuditArgumentError
from contractor_runtime.toolsets.audit_results.collector import AuditCollectionError

SECRET = "metrics-secret-that-must-not-survive"


@pytest.mark.parametrize("tool_only", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_shared_zero_direct_model_calls_report_explicit_token_zeros(tool_only, complete) -> None:
    expected = json.loads(
        (
            Path(__file__).parents[2] / "api/testdata/v1alpha1/zero-direct-model-usage.json"
        ).read_text()
    )
    state = MetricsState()
    if tool_only:
        state.record_tool_call("read_artifact", arguments={}, result={"ok": True})
    metrics = state.build_report(report_id="zero-model", duration_ms=1, complete=complete).metrics
    actual = metrics.model_dump(by_alias=True, exclude_none=True)
    for name in ["modelCalls", "inputTokens", "outputTokens", "totalTokens"]:
        assert actual.get(name) == expected[name]


def test_shared_started_model_call_keeps_unknown_usage_including_provider_zero() -> None:
    state = MetricsState()
    state.record_model_call()
    unknown = state.build_report(report_id="unknown-model", duration_ms=1).metrics
    assert unknown.model_calls == 1
    assert unknown.input_tokens is None
    assert unknown.output_tokens is None
    assert unknown.total_tokens is None
    state.record_model_usage(
        SimpleNamespace(prompt_token_count=0, candidates_token_count=0, total_token_count=0)
    )
    known = state.build_report(report_id="known-model-zero", duration_ms=1).metrics
    # A provider zero total is the canonical missing-usage placeholder.
    assert (known.input_tokens, known.output_tokens, known.total_tokens) == (0, 0, None)
    state.record_model_usage(
        SimpleNamespace(prompt_token_count=1, candidates_token_count=0, total_token_count=1)
    )
    positive = state.build_report(report_id="known-model-positive", duration_ms=1).metrics
    assert (positive.input_tokens, positive.output_tokens, positive.total_tokens) == (1, 0, 1)


TELEMETRY_SIZE_CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/v1alpha1/telemetry-json-size-cases.json").read_text()
)
REPORT_TRIM_FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "api/testdata/v1alpha1/telemetry-report-trim-cases.json"
    ).read_text(encoding="utf-8")
)
REPORT_TRIM_CASES = REPORT_TRIM_FIXTURE["cases"]
# The Worker report bound before the allocation envelope is added.
WORKER_REPORT_BYTES = MAX_REPORT_JSON_BYTES - 4096


class ToolFailure(RuntimeError):
    code = "synthetic_tool_failure"
    retryable = True


class SkillDisclosureFailure(RuntimeError):
    code = "SKILL_DISCLOSURE_LIMIT"
    retryable = False


@pytest.mark.parametrize("case", TELEMETRY_SIZE_CASES, ids=lambda case: case["name"])
def test_shared_argument_summary_uses_shared_compact_utf8_limit(case):
    arguments = {"path": case["unit"] * case["repeat"]}
    assert (
        len(json.dumps(arguments, ensure_ascii=False, separators=(",", ":")).encode())
        == case["compact_bytes"]
    )
    state = MetricsState()
    state.record_tool_call("read_source_file", arguments=arguments)
    report = state.build_report(report_id="worker-size", duration_ms=1)
    call = report.tool_calls[0]
    if case["valid"]:
        assert call.arguments == arguments
        assert not call.arguments_truncated
        assert case["unit"].encode() in encode_private(report)
    else:
        assert call.arguments_truncated
        assert call.arguments is not None
        assert call.arguments["summary"] == "[TRUNCATED]"
        assert call.arguments["originalSizeBytes"] > MAX_ARGUMENT_SUMMARY_BYTES


@pytest.mark.parametrize("error_type", [AuditArgumentError, AuditCollectionError])
def test_shared_input_error_diagnostics_keep_repair_reason_and_redact_secrets(error_type):
    state = MetricsState()
    error = error_type("completed", f"Use the requested coverage. {SECRET}")
    error.details["invalidValue"] = "untrusted-input-must-not-be-logged"
    state.record_tool_call("submit_result", arguments={}, error=error, secrets=(SECRET,))
    report = state.build_report(report_id="worker-input-error", duration_ms=1)
    message = report.tool_calls[0].error.message
    assert message == "completed: Use the requested coverage. [REDACTED]"
    assert state.errors[0].message == message
    assert not report.tool_calls[0].error.retryable
    encoded = report.model_dump_json()
    assert SECRET not in encoded
    assert "untrusted-input-must-not-be-logged" not in encoded


def test_shared_short_secrets_never_garble_tool_names_codes_or_messages() -> None:
    state = MetricsState()
    state.record_tool_call(
        "read_artifact",
        arguments={"name": "a", "path": "data/alpha", "limit": "1", "ops": "operations"},
        error=ToolFailure("provider failed"),
        secrets=("a", "ops", "1"),
    )

    report = state.build_report(report_id="worker-short-secrets", duration_ms=1)
    call = report.tool_calls[0]
    assert call.tool == "read_artifact"
    assert call.error is not None
    assert call.error.code == "synthetic_tool_failure"
    assert call.error.message == "Tool read_artifact failed (ToolFailure)"
    # A short private value matches only a complete string.
    assert call.arguments == {
        "name": "[REDACTED]",
        "path": "data/alpha",
        "limit": "[REDACTED]",
        "[REDACTED]": "operations",
    }
    assert [error.code for error in report.errors] == ["tool_read_artifact_failed"]
    assert set(report.metrics.tools) == {"read_artifact"}


def test_shared_long_secrets_stay_redacted_in_arguments_messages_and_codes() -> None:
    private = "private-canary-value-0f3a9"

    class PrivateCode(RuntimeError):
        code = f"code-{private}"

    class TextCode(RuntimeError):
        code = "provider said no"

    state = MetricsState()
    state.record_tool_call(
        "probe",
        arguments={f"key-{private}": f"value {private} tail", "same": private},
        error=PrivateCode(),
        secrets=(private, "a"),
    )
    state.record_tool_call("probe", arguments={}, error=TextCode(), secrets=(private,))

    report = state.build_report(report_id="worker-long-secrets", duration_ms=1)
    first, second = report.tool_calls
    assert first.arguments == {"key-[REDACTED]": "value [REDACTED] tail", "same": "[REDACTED]"}
    assert first.error is not None and first.error.code == "tool_call_failed"
    assert second.error is not None and second.error.code == "tool_call_failed"
    assert private not in report.model_dump_json()


def test_shared_report_keeps_exact_aggregates_while_dropping_oldest_bounded_detail() -> None:
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
    # The State projection admitted with each record follows the same
    # retention, record for record.
    projection = state.snapshot()
    assert projection["toolCalls"] == [
        call.model_dump(mode="json", by_alias=True, exclude_none=True) for call in state.tool_calls
    ]
    assert projection["errors"] == [
        error.model_dump(mode="json", by_alias=True, exclude_none=True) for error in state.errors
    ]
    assert projection["truncated"] is True
    assert SECRET not in json.dumps(projection)

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


def test_shared_argument_summary_redacts_nested_secrets_urls_and_artifact_bytes() -> None:
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


def test_shared_argument_numbers_agree_with_the_server() -> None:
    # LIM-OBS-16 as the shared fixture pins it: a kept number stays as the
    # Runtime writes it; any other number, NaN and infinities become the one
    # marker in a summary the Runtime builds, and the record model rejects one
    # written anyway, as the Server does on receipt.
    numbers = REPORT_TRIM_FIXTURE["argumentNumbers"]
    assert numbers["minExponent"] == MIN_ARGUMENT_NUMBER_EXPONENT
    assert numbers["maxExponent"] == MAX_ARGUMENT_NUMBER_EXPONENT
    assert numbers["marker"] == ARGUMENT_NUMBER_MARKER
    record = (
        '{"callId":"tool-1","tool":"rank","arguments":{"value":%s},'
        '"argumentsTruncated":false,"outcome":"succeeded"}'
    )
    values = [(literal, True) for literal in numbers["kept"]]
    values += [(literal, False) for literal in numbers["replaced"]]
    for literal, kept in values:
        value = json.loads(literal)
        assert json.dumps(value) == literal, literal
        state = MetricsState()
        state.record_tool_call("rank", arguments={"value": value})
        call = state.build_report(report_id="worker-numbers", duration_ms=1).tool_calls[0]
        assert call.arguments == {"value": value if kept else ARGUMENT_NUMBER_MARKER}, literal
        assert call.arguments_truncated is not kept, literal
        if kept:
            ToolCallRecord.model_validate_json(record % literal)
        else:
            with pytest.raises(ValidationError):
                ToolCallRecord.model_validate_json(record % literal)
    for value in (float("nan"), float("inf"), float("-inf")):
        state = MetricsState()
        state.record_tool_call("rank", arguments={"value": value, "content_bytes": 10**20})
        call = state.build_report(report_id="worker-numbers", duration_ms=1).tool_calls[0]
        assert call.arguments == {
            "value": ARGUMENT_NUMBER_MARKER,
            "content_bytes": {"redacted": True, "size": None},
        }


def test_shared_argument_redaction_leaves_redacted_arguments_unchanged() -> None:
    state = MetricsState()
    state.record_tool_call(
        "write_artifact",
        arguments={
            "path": "notes/a.txt",
            "content": "plain text",
            "payload": b"raw bytes",
            "attachment": bytearray(b"abc"),
            "content_bytes": 120,
            "database_url": "postgres://db.example.test/app",
            "x" * 70 + "token": "v",
        },
    )
    state.record_tool_call("list_files", arguments={})
    first, empty = state.tool_calls
    assert first.arguments == {
        "path": "notes/a.txt",
        "content": {"redacted": True, "size": 10},
        "payload": {"redacted": True, "size": 9},
        "attachment": {"redacted": True, "size": 3},
        "content_bytes": 120,
        "database_url": {"redacted": True, "size": 30},
        "x" * 70 + "token": {"redacted": True, "size": 1},
    }
    assert first.arguments_truncated is False
    assert empty.arguments is None

    # Redacting the redacted summary again, as the Server does, changes nothing.
    state.record_tool_call("write_artifact", arguments=first.arguments)
    again = state.tool_calls[-1]
    assert _compact(again.arguments) == _compact(first.arguments)
    assert again.arguments_truncated is False
    # A placeholder whose size LIM-OBS-16 would reject is a value like any
    # other: it is redacted afresh instead of failing the record.
    state.record_tool_call(
        "http_session_set",
        arguments={"api_key": {"redacted": True, "size": 10**20}},
    )
    assert state.tool_calls[-1].arguments == {"api_key": {"redacted": True, "size": 2}}
    report = state.build_report(report_id="worker-redacted", duration_ms=1)
    encoded = _compact(report.model_dump(mode="json", by_alias=True, exclude_none=True))
    assert '"arguments":{}' not in encoded


def test_shared_skill_metric_projects_only_validated_identity_outcome_and_result_size() -> None:
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


def test_shared_allocation_counters_saturate_at_the_state_wire_limit() -> None:
    state = MetricsState()
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=MAX_METRIC_COUNTER + 10,
            candidates_token_count=MAX_METRIC_COUNTER + 20,
            total_token_count=2 * MAX_METRIC_COUNTER + 30,
            cached_content_token_count=MAX_METRIC_COUNTER + 5,
        )
    )
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
            cached_content_token_count=1,
        )
    )

    assert state.counters == {
        "cached_input_tokens": MAX_METRIC_COUNTER,
        "input_tokens": MAX_METRIC_COUNTER,
        "output_tokens": MAX_METRIC_COUNTER,
        "total_tokens": MAX_METRIC_COUNTER,
    }


def test_shared_report_counters_saturate_at_the_report_integer_bound() -> None:
    # Allocation counters keep the Worker State bound; the report projects
    # them onto the lower execution report bound instead of building a report
    # the Server rejects (OBS-R111).
    state = MetricsState()
    state.start_worker_budget(max_model_calls=1, max_tool_calls=1, max_total_tokens=100)
    state.record_model_call()
    state.record_model_usage(
        SimpleNamespace(
            prompt_token_count=MAX_METRIC_COUNTER,
            candidates_token_count=MAX_METRIC_COUNTER,
            total_token_count=2 * MAX_METRIC_COUNTER,
            cached_content_token_count=0,
        )
    )
    state.observe_worker_budget(
        model_calls=1,
        tool_calls=1,
        total_tokens=2 * MAX_METRIC_COUNTER,
        token_usage_unavailable=MAX_METRIC_COUNTER + 1,
    )
    state.record_summarizer_attempt(
        succeeded=True,
        model_calls=1,
        input_tokens=MAX_METRIC_COUNTER,
        output_tokens=MAX_METRIC_COUNTER,
        total_tokens=2 * MAX_METRIC_COUNTER,
        token_usage_unavailable=0,
    )
    for _ in range(2):
        state.record_tool_call("read_file", arguments={})
    state.record_tool_call("read_file", arguments={}, error=ToolFailure("synthetic failure"))
    state._tool_metrics["read_file"] = {
        "calls": MAX_METRIC_COUNTER,
        "succeeded": MAX_METRIC_COUNTER,
        "failed": MAX_METRIC_COUNTER,
    }

    report = state.build_report(report_id="worker-saturated", duration_ms=MAX_METRIC_COUNTER)

    metrics = report.metrics
    assert state.counters["total_tokens"] == MAX_METRIC_COUNTER
    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (
        MAX_EXACT_INTEGER,
        MAX_EXACT_INTEGER,
        MAX_EXACT_INTEGER,
    )
    assert (metrics.duration_ms, metrics.model_calls) == (MAX_EXACT_INTEGER, 1)
    tool = metrics.tools["read_file"]
    assert (tool.calls, tool.succeeded, tool.failed) == (MAX_EXACT_INTEGER,) * 3
    assert metrics.worker_budget is not None
    assert metrics.worker_budget.observed_total_tokens == MAX_EXACT_INTEGER
    assert metrics.worker_budget.token_usage_unavailable == MAX_EXACT_INTEGER
    assert metrics.summarizer is not None
    assert metrics.summarizer.total_tokens == MAX_EXACT_INTEGER
    # A fresh validation, which the budget's model_copy skips, accepts it.
    encoded = report.model_dump_json(by_alias=True, exclude_none=True)
    assert ExecutionReport.model_validate_json(encoded) == report
    above = json.loads(encoded)
    above["metrics"]["totalTokens"] = MAX_EXACT_INTEGER + 1
    with pytest.raises(ValidationError):
        ExecutionReport.model_validate_json(json.dumps(above))


class LongInputError(ToolInputError):
    """A tool input error whose diagnostic fills the error-message bound."""


def _report_bytes(report: ExecutionReport) -> int:
    return len(
        json.dumps(
            report.model_dump(mode="json", by_alias=True, exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )


def _projections(records) -> list[dict]:
    return [record.model_dump(mode="json", by_alias=True, exclude_none=True) for record in records]


def _trim(report: ExecutionReport, maximum: int) -> ExecutionReport:
    return trim_report_detail(
        report,
        maximum,
        tool_call_projections=_projections(report.tool_calls),
        error_projections=_projections(report.errors),
    )


def _previous_trim(report: ExecutionReport, maximum: int) -> ExecutionReport:
    """The loop trim_report_detail replaced: measure the whole report after each removal."""

    tool_calls = list(report.tool_calls)
    errors = list(report.errors)
    while _report_bytes(report) > maximum:
        if tool_calls:
            tool_calls.pop(0)
        elif errors:
            errors.pop(0)
        else:
            raise ValueError("aggregate execution report exceeds the 1 MiB limit")
        report = report.model_copy(
            update={"tool_calls": list(tool_calls), "errors": list(errors), "truncated": True}
        )
    return report


def _removal_sequence(report: ExecutionReport) -> list[ExecutionReport]:
    states = [report]
    tool_calls, errors = list(report.tool_calls), list(report.errors)
    while tool_calls or errors:
        if tool_calls:
            tool_calls.pop(0)
        else:
            errors.pop(0)
        states.append(
            report.model_copy(
                update={"tool_calls": list(tool_calls), "errors": list(errors), "truncated": True}
            )
        )
    return states


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _runtime_built_report(inputs: dict) -> ExecutionReport:
    """Build the report the fixture's runtimeInputs describe through MetricsState."""

    state = MetricsState()
    budget = inputs.get("workerBudget")
    if budget is not None:
        state.start_worker_budget(
            max_model_calls=budget["maxModelCalls"],
            max_tool_calls=budget["maxToolCalls"],
            max_total_tokens=budget["maxTotalTokens"],
        )
    for call in inputs["toolCalls"]:
        state.record_tool_call(
            call["tool"],
            arguments=call["arguments"],
            error=ToolFailure("synthetic failure") if call.get("failed") else None,
        )
    completion = inputs.get("completion")
    if completion is not None:
        state.record_completion(WorkerCompletionDiagnostics.model_validate(completion))
    return state.build_report(report_id=inputs["reportId"], duration_ms=inputs["durationMs"])


def _count_measurements(monkeypatch) -> dict[str, int]:
    """Count complete-report and single-record JSON measurements by the metrics module."""

    counts = {"reports": 0, "records": 0}
    measure = metrics_module._json_size

    def counting(value):
        counts["reports" if isinstance(value, dict) and "reportId" in value else "records"] += 1
        return measure(value)

    monkeypatch.setattr(metrics_module, "_json_size", counting)
    return counts


def test_shared_redaction_names_agree_with_the_server() -> None:
    # Every name the Server redacts is redacted here first, so the Server's
    # second pass never changes a member this producer kept (OBS-R27).
    redaction = REPORT_TRIM_FIXTURE["redaction"]
    unredacted = [
        name
        for name in redaction["sensitiveNames"]
        if not metrics_module._is_sensitive_metric_key(name)
    ]
    assert unredacted == []
    assert frozenset(redaction["safeDerivedSizeNames"]) == SAFE_DERIVED_SIZE_METRIC_KEYS


@pytest.mark.parametrize("case", REPORT_TRIM_CASES, ids=lambda case: case["name"])
def test_shared_report_trim_keeps_the_shared_suffix_at_every_boundary(case) -> None:
    if "runtimeInputs" in case:
        # The fixture holds what this producer builds; the Server checks that
        # its own redaction leaves those records unchanged.
        built = _runtime_built_report(case["runtimeInputs"])
        dumped = built.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert _compact(dumped) == _compact(case["report"])
    report = ExecutionReport.model_validate_json(json.dumps(case["report"]))
    # An empty arguments object counts as absent, as the Server encodes it.
    assert _report_bytes(report) == case["bytes"]
    for limit in case["limits"]:
        if limit.get("aggregateTooLarge"):
            with pytest.raises(ValueError, match="aggregate execution report"):
                _trim(report, limit["maximumBytes"])
            continue
        trimmed = _trim(report, limit["maximumBytes"])
        assert trimmed.tool_calls == report.tool_calls[limit["toolCallsFrom"] :]
        assert trimmed.errors == report.errors[limit["errorsFrom"] :]
        assert trimmed.truncated is limit["truncated"]
        assert trimmed.metrics == report.metrics
        assert _report_bytes(trimmed) == limit["bytes"] <= limit["maximumBytes"]


def test_shared_report_trim_matches_the_previous_loop_byte_for_byte() -> None:
    state = MetricsState()
    units = ("a", "ж", "界", "😀", "\u2028", "<>&", '"\\', "\n\t\u0001")
    for index in range(24):
        unit = units[index % len(units)]
        error = LongInputError("failed " + unit * (index * 11 + 1)) if index % 3 == 0 else None
        state.record_tool_call(
            "search_text",
            arguments={"pattern": unit * (index * 7 + 1), "limit": index},
            error=error,
            duration_ms=index,
        )
    state.record_model_error(RuntimeError("provider failed"))
    report = state.build_report(
        report_id="worker-differential",
        duration_ms=5,
        extra_errors=(
            TerminationError(code="run_cancelled", message="界 cancelled", retryable=False),
        ),
    )
    assert not report.truncated and len(report.tool_calls) == 24 and len(report.errors) == 10
    boundaries = sorted(
        {
            maximum
            for candidate in _removal_sequence(report)
            for size in (_report_bytes(candidate),)
            for maximum in (size - 1, size, size + 1)
        }
    )
    for maximum in boundaries:
        try:
            expected = _previous_trim(report, maximum)
        except ValueError:
            with pytest.raises(ValueError, match="aggregate execution report"):
                _trim(report, maximum)
            continue
        encoded = _trim(report, maximum).model_dump_json(by_alias=True, exclude_none=True)
        assert encoded == expected.model_dump_json(by_alias=True, exclude_none=True), maximum


def test_shared_report_trim_measures_the_complete_report_at_most_three_times(monkeypatch) -> None:
    state = MetricsState()
    message = "界" * 1365 + "a"  # 4096 UTF-8 bytes, the error-message bound
    for index in range(MAX_METRIC_TOOL_CALLS):
        failed = index >= MAX_METRIC_TOOL_CALLS - MAX_METRIC_ERRORS
        state.record_tool_call(
            "read_source_file",
            arguments={"path": f"{index:04d}" + "ж" * 1990},
            error=LongInputError(message) if failed else None,
        )
    assert len(state.tool_calls) == MAX_METRIC_TOOL_CALLS
    assert [len(error.message.encode()) for error in state.errors] == [4096] * MAX_METRIC_ERRORS
    projected = []
    projection = metrics_module._record_projection
    monkeypatch.setattr(
        metrics_module,
        "_record_projection",
        lambda record: projected.append(record) or projection(record),
    )
    counts = _count_measurements(monkeypatch)

    report = state.build_report(report_id="worker-maximum-detail", duration_ms=1)

    # The candidate, the envelope without detail and the retained result; each
    # admitted record is measured once and none is projected again.
    assert counts == {"reports": 3, "records": MAX_METRIC_TOOL_CALLS + MAX_METRIC_ERRORS}
    assert projected == []
    removed = MAX_METRIC_TOOL_CALLS - len(report.tool_calls)
    assert 0 < removed < MAX_METRIC_TOOL_CALLS
    assert report.tool_calls == list(state.tool_calls[removed:])
    assert report.errors == list(state.errors) and report.truncated
    assert report.metrics.tools["read_source_file"].calls == MAX_METRIC_TOOL_CALLS
    # It is the first report of the removal sequence that fits.
    assert _report_bytes(report) <= WORKER_REPORT_BYTES
    one_more = report.model_copy(update={"tool_calls": list(state.tool_calls[removed - 1 :])})
    assert _report_bytes(one_more) > WORKER_REPORT_BYTES


def test_shared_report_trim_rejects_an_aggregate_larger_than_the_bound() -> None:
    state = MetricsState()
    for index in range(20_000):
        state.record_tool_call(f"tool_{index:05d}_" + "x" * 50, arguments={"value": index})
    with pytest.raises(ValueError, match="aggregate execution report exceeds the 1 MiB limit"):
        state.build_report(report_id="worker-aggregate", duration_ms=1)


def test_shared_report_trim_measures_a_fitting_report_once(monkeypatch) -> None:
    state = MetricsState()
    state.record_tool_call("read_source_file", arguments={"path": "src/a.py"})
    counts = _count_measurements(monkeypatch)
    report = state.build_report(report_id="worker-small", duration_ms=1)
    assert counts == {"reports": 1, "records": 0}
    assert not report.truncated and len(report.tool_calls) == 1


def test_shared_report_trim_rejects_inconsistent_size_accounting(monkeypatch) -> None:
    report = ExecutionReport.model_validate_json(json.dumps(REPORT_TRIM_CASES[0]["report"]))
    measure = metrics_module._report_size

    def inflated(candidate: ExecutionReport) -> int:
        partial = 0 < len(candidate.tool_calls) < len(report.tool_calls)
        return measure(candidate) + (7 if partial else 0)

    monkeypatch.setattr(metrics_module, "_report_size", inflated)
    with pytest.raises(RuntimeError, match="not the computed"):
        _trim(report, _report_bytes(report) - 1)


def test_shared_model_phases_project_existing_summarization_without_a_second_owner() -> None:
    from typing import get_args

    from contractor_runtime.contracts.compaction import WorkerModelPhases
    from contractor_runtime.contracts.llm_cost import CostPhase

    assert set(WorkerModelPhases.model_fields) == set(get_args(CostPhase))
    metrics = MetricsState()
    metrics.record_summarizer_attempt(
        succeeded=True,
        model_calls=1,
        input_tokens=20,
        output_tokens=5,
        total_tokens=25,
        token_usage_unavailable=0,
    )
    phases = metrics.model_phases()
    assert phases.summarization.model_calls == 1
    assert phases.summarization.total_tokens == 25
    assert phases.summarization.duration_ms is None
    assert (
        phases.main.model_calls
        == phases.compaction.model_calls
        == phases.finalization.model_calls
        == 0
    )
    assert "summarization" not in metrics._model_phases
    report = metrics.build_report(report_id="summary", duration_ms=1)
    assert (
        report.metrics.model_phases.summarization.total_tokens
        == report.metrics.summarizer.total_tokens
    )
