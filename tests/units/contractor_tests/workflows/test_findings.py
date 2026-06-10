"""Tests for the shared YAML findings-artifact loaders."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from contractor.workflows.findings import (
    load_findings_artifact,
    load_yaml_dict_artifact,
)


def _service(text: str | None) -> AsyncMock:
    part = None if text is None else SimpleNamespace(text=text)
    service = AsyncMock()
    service.load_artifact = AsyncMock(return_value=part)
    return service


async def _load_dict(text: str | None) -> dict:
    return await load_yaml_dict_artifact(
        _service(text), app_name="app", user_id="u", filename="f"
    )


async def _load_findings(text: str | None) -> list[dict]:
    return await load_findings_artifact(
        _service(text), app_name="app", user_id="u", filename="f"
    )


class TestLoadYamlDictArtifact:
    @pytest.mark.asyncio
    async def test_missing_artifact_returns_empty(self):
        assert await _load_dict(None) == {}

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        assert await _load_dict("") == {}

    @pytest.mark.asyncio
    async def test_invalid_yaml_returns_empty(self):
        assert await _load_dict("{ not: [valid") == {}

    @pytest.mark.asyncio
    async def test_non_mapping_returns_empty(self):
        assert await _load_dict("- a\n- b\n") == {}

    @pytest.mark.asyncio
    async def test_mapping_round_trips(self):
        assert await _load_dict("a: 1\nb: 2\n") == {"a": 1, "b": 2}


class TestLoadFindingsArtifact:
    @pytest.mark.asyncio
    async def test_name_backfilled_from_key(self):
        findings = await _load_findings("sqli:\n  severity: high\n")
        assert findings == [{"name": "sqli", "severity": "high"}]

    @pytest.mark.asyncio
    async def test_explicit_name_field_wins(self):
        findings = await _load_findings("key:\n  name: explicit\n")
        assert findings == [{"name": "explicit"}]

    @pytest.mark.asyncio
    async def test_non_mapping_entries_dropped(self):
        findings = await _load_findings("good:\n  severity: low\nbad: just-a-string\n")
        assert findings == [{"name": "good", "severity": "low"}]

    @pytest.mark.asyncio
    async def test_missing_artifact_returns_empty_list(self):
        assert await _load_findings(None) == []
