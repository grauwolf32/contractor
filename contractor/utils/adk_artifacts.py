"""Compatibility helpers for Google ADK's file artifact storage."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_legacy_artifact_layout(root: Path, *, app_name: str) -> bool:
    """Copy pre-ADK-2.6 artifacts into the app-scoped storage layout.

    Google ADK 2.7 only reads ``apps/<app>/users`` and no longer falls back to
    the legacy top-level ``users`` directory. Contractor's stores have one
    known app name, so callers can safely attribute the old data. The source is
    deliberately retained as a recovery copy.

    Returns ``True`` when a copy was made. If the app-scoped destination
    already exists, it is left untouched to avoid merging version histories.
    """
    root = Path(root)
    legacy_users = root / "users"
    scoped_users = root / "apps" / app_name / "users"
    if not legacy_users.is_dir():
        return False
    if scoped_users.exists():
        logger.warning(
            "Legacy artifacts remain at %s; skipped migration because %s "
            "already exists",
            legacy_users,
            scoped_users,
        )
        return False

    shutil.copytree(legacy_users, scoped_users)
    logger.info(
        "Copied legacy artifacts from %s to ADK's app-scoped layout at %s",
        legacy_users,
        scoped_users,
    )
    return True
