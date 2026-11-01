"""A Run-scoped reader follows the same complete exact snapshot fixture as Go."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contractor_runtime.contracts import ArtifactRef


def test_shared_round_results_exact_pages_and_provenance() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "api/testdata/audit-composition/round-results-cases.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["limits"] == {
        "documentBytes": 256 << 10,
        "pages": 256,
        "totalBytes": 64 << 20,
        "pageItems": 128,
        "sliceDescriptors": 512,
        "sliceBytes": 1 << 20,
    }
    page, manifest, consumer = (fixture[key] for key in ("page", "manifest", "consumer"))
    canonical = json.dumps(page, separators=(",", ":")).encode()
    descriptor = manifest["pages"][0]["artifact"]
    assert descriptor["digest"] == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert descriptor["sizeBytes"] == len(canonical) == manifest["metadata_bytes"]
    assert (
        consumer["source_generation"] == manifest["source_generation"] == page["source_generation"]
    )
    assert len(consumer["pages"]) == len(consumer["mappings"]) == 1
    source = ArtifactRef.model_validate(consumer["mappings"][0]["source"]["ref"])
    local = ArtifactRef.model_validate(consumer["pages"][0]["artifact"]["ref"])
    assert source.revision is not None and local.revision is not None
    assert local.namespace == "audit-context"
    assert consumer["mappings"][0]["consumer"]["ref"] == local.model_dump(exclude_none=True)
    item = page["items"][0]
    assert item["selected_attempt"]["attempt"] == 2
    assert item["findings"][0]["project_id"] == "project-fixture"
    assert set(item["findings"][0]) == {"finding_id", "project_id"}
    assert item["attachments"][0]["classification"] == "accepted-attachment"
