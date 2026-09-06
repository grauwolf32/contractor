from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fakes.spec import allocation_spec
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from referencing import Registry, Resource

from contractor_runtime.contracts import (
    AgentRegistrationV2,
    AllocationFinalResponse,
    AllocationSpecV2,
    PerformanceMetricsRequest,
    PrivateProtocolDecodeError,
    RuntimeReport,
    RuntimeReportV2,
    RuntimeResources,
    decode_private_v2,
    encode_private_v2,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "testdata/contracts/private-v2"
CASES = json.loads((FIXTURES / "performance-cases.json").read_text())
FINAL = ROOT / "api/testdata/v1alpha1/valid/allocation-final-response.json"
SCHEMAS = {
    schema["$id"]: schema
    for path in (ROOT / "api").glob("**/*.schema.json")
    for schema in [json.loads(path.read_text())]
    if "$id" in schema
}
REGISTRY = Registry().with_resources(
    [(name, Resource.from_contents(schema)) for name, schema in SCHEMAS.items()]
)


def schema_validator(name: str, definition: str | None = None) -> Draft202012Validator:
    schema = SCHEMAS[f"https://contractor.local/schema/{name}.schema.json"]
    if definition is not None:
        schema = {**schema, "$ref": f"#/$defs/{definition}"}
    return Draft202012Validator(schema, registry=REGISTRY)


@pytest.mark.parametrize("case", CASES["validResources"], ids=lambda case: case["name"])
def test_performance_golden_valid_resources_round_trip(case: dict) -> None:
    value = decode_private_v2(RuntimeResources, json.dumps(case["value"]))
    assert json.loads(encode_private_v2(value)) == case["value"]
    schema_validator("v1alpha1/performance", "runtimeResources").validate(case["value"])
    report = json.loads(FINAL.read_text())
    report["report"]["runtime"]["resources"] = case["value"]
    final = decode_private_v2(AllocationFinalResponse, json.dumps(report))
    assert final.report.runtime.resources == value
    assert final.report.runtime.resources_error is None
    assert json.loads(encode_private_v2(final))["report"]["runtime"]["resources"] == case["value"]
    schema_validator("private-v2/runtime-report").validate(report["report"]["runtime"])
    schema_validator("v1alpha1/allocation").validate(report)


@pytest.mark.parametrize("case", CASES["invalidResources"], ids=lambda case: case["name"])
def test_performance_golden_resource_errors_are_isolated(case: dict) -> None:
    with pytest.raises(PrivateProtocolDecodeError) as caught:
        decode_private_v2(RuntimeResources, json.dumps(case["value"]))
    assert "secret-canary" not in str(caught.value)
    if case["schemaInvalid"]:
        assert not schema_validator("v1alpha1/performance", "runtimeResources").is_valid(
            case["value"]
        )
    report = json.loads(FINAL.read_text())
    original_worker = report["report"]["worker"].copy()
    report["report"]["runtime"]["resources"] = case["value"]
    final = decode_private_v2(AllocationFinalResponse, json.dumps(report))
    assert final.report.runtime.resources is None
    assert final.report.runtime.resources_error == "invalid_report"
    assert final.report.runtime.complete is True
    assert json.loads(encode_private_v2(final))["report"]["worker"] == original_worker
    assert b"secret-canary" not in encode_private_v2(final)
    assert b"resources_error" not in encode_private_v2(final)
    for model in (RuntimeReport, RuntimeReportV2):
        runtime = decode_private_v2(model, json.dumps(report["report"]["runtime"]))
        assert runtime.resources is None
        assert runtime.resources_error == "invalid_report"
        assert runtime.complete is True


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0, True])
def test_non_finite_negative_and_boolean_measurements_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        RuntimeResources.model_validate(
            {"version": 1, "scope": "runtime_process", "status": "partial", "cpuUserSeconds": value}
        )


@pytest.mark.parametrize("valid", [True, False])
def test_performance_golden_requests_and_capabilities(valid: bool) -> None:
    prefix = "valid" if valid else "invalid"
    for value in CASES[f"{prefix}Requests"]:
        validator = schema_validator("v1alpha1/performance", "performanceMetricsRequest")
        assert validator.is_valid(value) == valid
        if valid:
            assert (
                json.loads(
                    encode_private_v2(
                        decode_private_v2(PerformanceMetricsRequest, json.dumps(value))
                    )
                )
                == value
            )
        else:
            with pytest.raises(PrivateProtocolDecodeError):
                decode_private_v2(PerformanceMetricsRequest, json.dumps(value))
    for value in CASES[f"{prefix}Capabilities"]:
        registration = json.loads((FIXTURES / "valid/agent-registration.json").read_text())
        registration["supportedPerformanceMetricsVersions"] = value
        assert schema_validator("private-v2/agent-registration").is_valid(registration) == valid
        if valid:
            decoded = decode_private_v2(AgentRegistrationV2, json.dumps(registration))
            assert decoded.supported_performance_metrics_versions == value
        else:
            with pytest.raises(PrivateProtocolDecodeError):
                decode_private_v2(AgentRegistrationV2, json.dumps(registration))


def test_performance_old_new_allocation_and_registration_contracts() -> None:
    raw = (FIXTURES / "valid/agent-registration.json").read_bytes().strip()
    registration = decode_private_v2(AgentRegistrationV2, raw)
    assert registration.supported_performance_metrics_versions == []
    assert encode_private_v2(registration) == raw
    assert b"supportedPerformanceMetricsVersions" not in encode_private_v2(registration)
    for requested in (False, True):
        spec = allocation_spec()
        if requested:
            spec.performance_metrics = PerformanceMetricsRequest(version=1, interval_seconds=15)
        raw = encode_private_v2(spec)
        decoded = decode_private_v2(AllocationSpecV2, raw)
        assert (decoded.performance_metrics is not None) == requested
        assert (b"performanceMetrics" in raw) == requested
        schema_validator("private-v2/allocation").validate(json.loads(raw))


@pytest.mark.parametrize(
    "raw",
    [
        '{"complete":true,"adapters":{},"resources":{"version":1,"version":2}}',
        '{"complete":true,"adapters":{},"resources":{"cpuUserSeconds":NaN}}',
        '{"complete":true,"adapters":{},"resources":{"cpuUserSeconds":Infinity}}',
        '{"complete":true,"adapters":{},"resources":{},"unknown":"secret-canary"}',
        '{"complete":true,"adapters":{},"resources":{}} {}',
    ],
)
def test_resource_isolation_preserves_envelope_syntax(raw: str) -> None:
    with pytest.raises(PrivateProtocolDecodeError) as caught:
        decode_private_v2(RuntimeReportV2, raw)
    assert "secret-canary" not in str(caught.value)


def test_resource_isolation_preserves_original_final_report_size_limit() -> None:
    report = json.loads(FINAL.read_text())
    report["report"]["runtime"]["resources"] = {"unknown": "x" * 1024 * 1024}
    with pytest.raises(PrivateProtocolDecodeError):
        decode_private_v2(AllocationFinalResponse, json.dumps(report))
