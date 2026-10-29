"""Actual scanner normalizers produce the reports consumed by the Go importer."""

import json
from pathlib import Path

from contractor_runtime.toolsets.scan.ffuf import ffuf_observation
from contractor_runtime.toolsets.scan.katana import katana_observation
from contractor_runtime.toolsets.scan.process import ProcessResult


def test_shared_sitemap_scanner_report_producers() -> None:
    fixtures = Path(__file__).parents[2] / "api/testdata/sitemap-reports"
    ffuf = json.loads((fixtures / "ffuf-report.json").read_text())
    raw = {
        "input": {"FUZZ": "dXNlcnM="},
        "position": 1,
        "status": 200,
        "length": 10,
        "words": 1,
        "lines": 1,
        "duration": 1000,
        "url": "https://example.test/users?expand=profile",
        "content-type": "application/json",
        "redirectlocation": "",
    }
    actual = ffuf_observation(
        ProcessResult(
            0,
            stdout=json.dumps(raw).encode(),
            stderr=b":: Progress: [1/1] :: Job [1/1] :: 1 req/sec :: Errors: 0 ::",
        ),
        1,
    )
    actual["wordlistArtifact"] = {
        "namespace": "inputs",
        "name": "wordlist",
        "revision": "wordlist-one",
    }
    assert actual == ffuf["observation"]
    assert ffuf["schemaVersion"] == 1 and ffuf["tool"] == "scan_ffuf"

    katana = json.loads((fixtures / "katana-report.json").read_text())
    actual, targets = katana_observation(
        ProcessResult(
            0,
            stdout=json.dumps(
                {
                    "request": {
                        "method": "GET",
                        "endpoint": "https://example.test/users?expand=profile",
                        "source": "https://example.test/",
                    },
                    "response": {"status_code": 200},
                }
            ).encode(),
        ),
        seed="https://example.test/",
        max_depth=2,
        max_pages=100,
        rate_limit=10,
        timeout_seconds=60,
    )
    published = {"namespace": "scan", "name": "targets", "revision": "targets-one"}
    actual.update(targetsArtifact=published, artifacts={"targets": published})
    assert targets == b"https://example.test/users?expand=profile\n"
    assert actual == katana["observation"]
    assert katana["schemaVersion"] == 1 and katana["tool"] == "scan_katana"
