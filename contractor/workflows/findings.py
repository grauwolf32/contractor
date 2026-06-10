"""Shared loaders for YAML findings artifacts.

Several workflows persist findings as a ``name -> fields`` YAML mapping
(``user:vulnerability-reports/...`` and friends) and re-load them for a
downstream stage. The load + parse + reshape steps live here so the read
side cannot drift between workflows.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from google.adk.artifacts import BaseArtifactService

logger = logging.getLogger(__name__)


async def load_yaml_dict_artifact(
    artifact_service: BaseArtifactService,
    *,
    app_name: str,
    user_id: str,
    filename: str,
) -> dict[str, Any]:
    """Load an artifact's text as a YAML mapping.

    Returns ``{}`` when the artifact is missing, empty, unparseable
    (logged as a warning), or parses to something other than a mapping.
    """
    part = await artifact_service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
    )
    if part is None or not getattr(part, "text", None):
        return {}
    try:
        raw = yaml.safe_load(part.text or "") or {}
    except yaml.YAMLError as exc:
        logger.warning("could not parse %s as YAML: %s — skipping", filename, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


async def load_findings_artifact(
    artifact_service: BaseArtifactService,
    *,
    app_name: str,
    user_id: str,
    filename: str,
) -> list[dict[str, Any]]:
    """Load a ``name -> fields`` findings artifact as a list of dicts.

    Each mapping value becomes one finding dict with its key backfilled
    under ``"name"`` (an explicit ``name`` field wins). Non-mapping
    values are dropped. Returns ``[]`` on any load/parse failure.
    """
    raw = await load_yaml_dict_artifact(
        artifact_service,
        app_name=app_name,
        user_id=user_id,
        filename=filename,
    )
    findings: list[dict[str, Any]] = []
    for name, item in raw.items():
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        entry.setdefault("name", name)
        findings.append(entry)
    return findings
