"""Common Task provenance obeys the same resource and revision bounds in both readers."""

import json

import pytest
from test_findings_reader_toolset import FIXTURE, _fixture, _json, _package

from contractor_runtime.toolsets.security_findings.collection import decode_collection


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["audit_origin_cases"], ids=lambda case: case["name"]
)
def test_shared_collection_task_origin_bounds(case):
    collection, contents = _fixture()
    collection["entries"][1]["audit_origin"] = json.loads(case["origin"])
    payload = _package(collection, contents, metadata=_json(collection))
    if case["valid"]:
        assert decode_collection(payload).metadata == collection
    else:
        with pytest.raises(ValueError):
            decode_collection(payload)
