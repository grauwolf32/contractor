"""Provider-neutral metadata normalization stays within the strict common wire."""

import json
from pathlib import Path

import pytest
from private_schemas import private_validator

from contractor_runtime.toolsets.common.observation_metadata import (
    normalize_http_observation,
    normalize_http_observations,
)


def observation(**changes):
    return {
        "url": "https://target.example/path?token=opaque&password=value",
        "method": "GET",
        "outcome": "response",
        "status": 200,
        "at": "2026-09-01T00:00:00Z",
        **changes,
    }


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("2026-09-01T03:00:00+03:00", "2026-09-01T00:00:00Z"),
        ("2026-08-31T23:00:00-01:00", "2026-09-01T00:00:00Z"),
        ("2026-09-01T00:00:00.000Z", "2026-09-01T00:00:00Z"),
        ("2026-09-01T03:00:00.123456789+03:00", "2026-09-01T00:00:00.123456789Z"),
        ("2026-09-01T00:00:00.123456700Z", "2026-09-01T00:00:00.1234567Z"),
    ],
)
def test_normalized_timestamps_preserve_nanoseconds_and_strict_wire(timestamp, expected):
    result = normalize_http_observation(observation(at=timestamp))
    assert result == observation(at=expected)
    assert private_validator("project-sitemap.schema.json#/$defs/SitemapHTTPObservation").is_valid(
        result
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": 700},
        {"status": True},
        {"status": "200"},
        {"outcome": "attempt", "status": 200},
        {"outcome": "response", "status": 0},
        {"method": "get"},
        {"method": "A" * 17},
        {"at": "provider-invalid"},
        {"at": "2026-09-01T00:00:00"},
        {"at": "2026-09-01T00:00:00+24:00"},
        {"at": "2026-02-30T00:00:00Z"},
        {"url": "https:///missing-host"},
        {"url": "https://target.example:0/path"},
        {"url": "https://[::1]wrong/path"},
        {"url": "https://target.example/%ZZ"},
        {"url": "https://target.example/?q=%ZZ"},
        {"url": "https://target.example/?q=a;b"},
        {"url": "https://target.example/?=value"},
        {"url": "https://target.example/?%00=value"},
        {"url": "https://target.example/?%FF=value"},
        {"url": "https://target.example/?" + "x" * 65 + "=value"},
        {"url": "https://target.example/" + "x" * 8192},
        {"parameters": [""]},
        {"parameters": ["x\n"]},
        {"parameters": ["x\u0085"]},
        {"parameters": ["é" * 33]},
        {"parameters": None},
        {"raw": "not metadata"},
    ],
)
def test_unrepresentable_rows_never_poison_other_observations(changes):
    good = observation()
    invalid = observation(**changes)
    assert normalize_http_observation(invalid) is None
    accepted, gaps = normalize_http_observations([good, invalid, good])
    assert accepted == [good, good]
    assert gaps == [{"reason": "projection", "count": 1}]


def test_projection_preserves_selected_path_and_query_without_secret_filtering():
    row = observation(
        url="https://user:excluded@target.example/Session/short-password/%2F?token=opaque#excluded",
        parameters=["authorization", "password", "authorization"],
    )
    result = normalize_http_observation(row)
    assert result == {
        **row,
        "url": "https://target.example/Session/short-password/%2F?token=opaque",
        "parameters": ["authorization", "password"],
    }
    assert normalize_http_observation(observation(url="https://target.example/é space"))["url"] == (
        "https://target.example/%C3%A9%20space"
    )


def test_parameter_union_is_bounded_without_counting_duplicate_query_values_as_names():
    query = "&".join(f"key{i}=value" for i in range(100))
    row = observation(url="https://target.example/?" + query, parameters=["key0"])
    assert normalize_http_observation(row) == row
    assert normalize_http_observation({**row, "parameters": ["extra"]}) is None
    repeated = observation(url="https://target.example/?" + "&".join(["q=value"] * 101))
    assert normalize_http_observation(repeated) == repeated
    formatted = observation(parameters=["\u200dtoken"])
    assert normalize_http_observation(formatted) == formatted


def test_url_bound_includes_unicode_authority_encoding():
    prefix = "https://bücher.example/"
    row = observation(url=prefix + "a" * (8192 - len(prefix.encode())))
    assert normalize_http_observation(row) is None


def test_page_work_and_projection_gaps_remain_bounded():
    accepted, gaps = normalize_http_observations([observation()] * 101)
    assert len(accepted) == 100 and gaps == [{"reason": "projection", "count": 1}]
    accepted, gaps = normalize_http_observations([observation(status=700)] * 100)
    assert accepted == [] and gaps == [{"reason": "projection", "count": 100}]


def test_shared_host_grammar_and_sitemap_projection_examples():
    root = Path(__file__).resolve().parents[2]
    hosts = json.loads((root / "api/testdata/scan-host-syntax.json").read_text())
    for case in hosts:
        result = normalize_http_observation(observation(url=f"http://{case['authority']}/path"))
        assert (result is not None) == case["accepted"], case["host"]
    cases = json.loads((root / "api/testdata/project-sitemap-cases.json").read_text())
    for case in cases["projection"]:
        result = normalize_http_observation(observation(url=case["url"]))
        assert result is not None
        assert private_validator(
            "project-sitemap.schema.json#/$defs/SitemapHTTPObservation"
        ).is_valid(result)
