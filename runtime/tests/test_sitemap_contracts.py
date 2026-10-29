"""Shared pinned sitemap authority and template identity."""

import json
from pathlib import Path

import pytest

from contractor_runtime.contracts import (
    PrivateProtocolDecodeError,
    ProjectSitemapPolicy,
    ResolvedAgentTemplate,
    decode_private,
)
from contractor_runtime.digests import _agent_template_digest


def test_shared_sitemap_access_policy_and_digest() -> None:
    cases = json.loads(
        (
            Path(__file__).resolve().parents[2] / "api/testdata/v1alpha1/sitemap-access-cases.json"
        ).read_text()
    )
    assert len(cases["valid"]) >= 2 and len(cases["invalid"]) >= 10
    for value in cases["valid"]:
        decode_private(ProjectSitemapPolicy, json.dumps(value))
    for value in [*cases["invalid"], *cases["invariantInvalid"]]:
        with pytest.raises(PrivateProtocolDecodeError):
            decode_private(ProjectSitemapPolicy, json.dumps(value))
    for raw in cases["strictInvalid"]:
        with pytest.raises(PrivateProtocolDecodeError):
            decode_private(ProjectSitemapPolicy, raw)
    template = ResolvedAgentTemplate.model_validate(cases["template"])
    assert _agent_template_digest(template) == template.ref.digest


def test_shared_sitemap_private_wire() -> None:
    from private_schemas import private_validator

    from contractor_runtime.contracts.sitemap import (
        SitemapMutationResponse,
        SitemapSnapshotResponse,
    )

    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/v1alpha1/sitemap-wire-cases.json").read_text()
    )
    models = {
        model.__name__: model
        for model in (
            SitemapMutationResponse,
            SitemapSnapshotResponse,
        )
    }
    from contractor_runtime.sitemap import MAX_SITEMAP_COMMAND_BYTES
    from contractor_runtime.sitemap_collection import (
        FLUSH_BUDGET_SECONDS,
        FLUSH_INTERVAL_SECONDS,
        MAX_ATTEMPTS,
        MAX_BYTES,
        MAX_ITEMS,
    )

    assert cases["collectorLimits"]["items"] == MAX_ITEMS
    assert cases["collectorLimits"]["batchBytes"] == MAX_BYTES == MAX_SITEMAP_COMMAND_BYTES
    assert "secrets" not in cases["collectorLimits"]
    assert cases["collectorLimits"]["attempts"] == MAX_ATTEMPTS
    assert cases["collectorLimits"]["flushBudgetSeconds"] == FLUSH_BUDGET_SECONDS
    assert cases["collectorLimits"]["flushIntervalSeconds"] == FLUSH_INTERVAL_SECONDS
    assert {case["type"] for case in cases["valid"]} == {
        "SitemapMutationRequest",
        "SitemapMutationResponse",
        "SitemapSnapshotRequest",
        "SitemapSnapshotResponse",
        "SitemapCollectionRequest",
    }
    assert len(cases["invalid"]) >= 15
    for valid, group in ((True, cases["valid"]), (False, cases["invalid"])):
        for case in group:
            validator = private_validator("project-sitemap.schema.json#/$defs/" + case["type"])
            assert validator.is_valid(case["value"]) == valid
            if case["type"] in models:
                if valid:
                    decode_private(models[case["type"]], json.dumps(case["value"]))
                else:
                    with pytest.raises(PrivateProtocolDecodeError):
                        decode_private(models[case["type"]], json.dumps(case["value"]))

    collection = next(
        case for case in cases["valid"] if "observations" in case["value"].get("input", {})
    )
    validator = private_validator("project-sitemap.schema.json#/$defs/" + collection["type"])
    obsolete = {**collection["value"], "input": {**collection["value"]["input"], "secrets": []}}
    assert not validator.is_valid(obsolete)
