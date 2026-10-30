"""The shared finding collection as an input of the findings reader tests.

api/testdata/findings/finding-collection-v1.fixture.json holds the Go Server
and the Runtime to one collection package; the test_shared_* test of
test_findings_reader_toolset.py checks it. The other reader tests build their
packages from it here and do not check it (HELPER_READS in shared_fixtures.py).
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_BYTES = (
    Path(__file__).resolve().parents[3]
    / "api"
    / "testdata"
    / "findings"
    / "finding-collection-v1.fixture.json"
).read_bytes()


def collection() -> tuple[dict, dict[str, bytes]]:
    """A fresh copy of the collection metadata and its document contents."""

    value = json.loads(FIXTURE_BYTES)
    return value["collection"], {key: text.encode() for key, text in value["contents"].items()}
