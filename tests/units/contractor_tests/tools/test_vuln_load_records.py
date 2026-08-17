"""Regression tests for ``_load_artifact_records`` failure handling.

Every vuln/verification CRUD op is load→mutate→save-of-parsed-records, so a
single malformed record (hand-edited artifact or schema drift) must (a) not raise
and turn every subsequent read/write/delete into an error, and (b) NOT be
silently dropped from the artifact on the next unrelated write. The loader keeps
such rows in an ``unparsed`` map that ``dump()`` re-emits verbatim.
"""
from __future__ import annotations

import pytest
import yaml
from google.genai import types

from contractor.tools.vuln import VulnerabilityReportTools, _load_artifact_records


class InMemoryArtifactCtx:
    def __init__(self) -> None:
        self.store: dict[str, types.Part] = {}

    async def save_artifact(self, *, filename: str, artifact: types.Part) -> None:
        self.store[filename] = artifact

    async def load_artifact(self, *, filename: str) -> types.Part | None:
        return self.store.get(filename)


class _Rec:
    def __init__(self, name: str) -> None:
        self.name = name


def _normalize_strict(name, item, index):
    """Stand-in for a frozen pydantic model that rejects out-of-vocab values."""
    if item.get("severity") == "BOGUS":
        raise ValueError(f"invalid severity for {name}")
    return _Rec(name)


def _normalize_internal_name(name, item, index):
    if item.get("severity") == "BOGUS":
        raise ValueError(f"invalid severity for {name}")
    return _Rec(item.get("name", name))


def _seed(ctx: InMemoryArtifactCtx, key: str, payload) -> None:
    ctx.store[key] = types.Part.from_text(text=yaml.safe_dump(payload))


@pytest.mark.asyncio
async def test_bad_record_is_preserved_not_dropped():
    ctx = InMemoryArtifactCtx()
    _seed(
        ctx,
        "k",
        {
            "good-1": {"severity": "high"},
            "poison": {"severity": "BOGUS"},  # normalize() raises on this row
            "good-2": {"severity": "low"},
            "not-a-dict": ["unexpected"],  # fails the per-item isinstance guard
        },
    )

    records, unparsed = await _load_artifact_records(
        ctx, artifact_key="k", normalize=_normalize_strict
    )

    # Valid rows parse; the poison + non-dict rows are preserved, not fatal and
    # not dropped — the caller re-dumps `unparsed` so they survive a save.
    assert set(records) == {"good-1", "good-2"}
    assert unparsed == {"poison": {"severity": "BOGUS"}, "not-a-dict": ["unexpected"]}


@pytest.mark.asyncio
async def test_non_mapping_top_level_raises():
    # A list/scalar top level has no named rows to preserve; returning {} would
    # let the next save() overwrite the whole artifact, so loading must fail loud.
    ctx = InMemoryArtifactCtx()
    _seed(ctx, "k", ["a", "b"])  # a list, not a mapping

    with pytest.raises(ValueError, match="not a top-level mapping"):
        await _load_artifact_records(ctx, artifact_key="k", normalize=_normalize_strict)


@pytest.mark.asyncio
async def test_missing_artifact_returns_empty():
    ctx = InMemoryArtifactCtx()
    records, unparsed = await _load_artifact_records(
        ctx, artifact_key="absent", normalize=_normalize_strict
    )
    assert records == {}
    assert unparsed == {}


@pytest.mark.asyncio
async def test_internal_name_collision_with_malformed_key_fails_without_data_loss():
    ctx = InMemoryArtifactCtx()
    original = {
        "valid-storage-key": {"name": "legacy-bad", "severity": "high"},
        "legacy-bad": {"severity": "BOGUS"},
    }
    _seed(ctx, "k", original)

    with pytest.raises(ValueError, match="record key collision"):
        await _load_artifact_records(
            ctx,
            artifact_key="k",
            normalize=_normalize_internal_name,
        )

    assert yaml.safe_load(ctx.store["k"].text) == original


@pytest.mark.asyncio
async def test_write_report_preserves_a_malformed_sibling_row():
    """End-to-end: writing a NEW report must not erase an existing malformed one.

    Reproduces the load→mutate→save data-loss path: an out-of-vocab ``severity``
    (schema drift / hand edit) fails ``VulnerabilityReport`` validation on load,
    and previously vanished from the artifact when any other report was written.
    """
    tools = VulnerabilityReportTools(name="ns")
    key = tools.artifact_key
    ctx = InMemoryArtifactCtx()
    _seed(
        ctx,
        key,
        {
            "legacy-bad": {"title": "old", "severity": "moderate"},  # not in vocab
            "good": {"title": "ok", "severity": "high"},
        },
    )

    await tools.write_report(
        name="fresh",
        place_type="file",
        place="app.py",
        title="Fresh finding",
        summary="",
        severity="low",
        confidence="medium",
        details="",
        ctx=ctx,
    )

    saved = yaml.safe_load(ctx.store[key].text)
    # The new report landed AND the malformed legacy row survived verbatim.
    assert "fresh" in saved
    assert "good" in saved
    assert saved["legacy-bad"] == {"title": "old", "severity": "moderate"}
