"""ADK plugin that removes a run's code-execution sandbox at agent-run end.

The exploit agents' ``run_python`` / ``execute_bash`` tools start a podman
container lazily, keyed by the ADK ``invocation_id`` (stable across the
invocation tree). This plugin's ``after_run_callback`` fires once when the
top-level run finishes and removes that run's container. ``atexit`` + the
container TTL in :mod:`contractor.tools.podman` are backstops.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from contractor.tools.podman import teardown_sandbox

logger = logging.getLogger(__name__)


class SandboxCleanupPlugin(BasePlugin):
    """Tear down the per-agent-run code-exec container when the run ends."""

    def __init__(self, name: str = "sandbox_cleanup") -> None:
        super().__init__(name=name)

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if not invocation_id:
            return
        try:
            teardown_sandbox(invocation_id)
        except Exception:  # cleanup must never break the run
            logger.exception("sandbox teardown failed for %s", invocation_id)
