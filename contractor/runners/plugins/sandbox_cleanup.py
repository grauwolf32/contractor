"""ADK plugin that removes code-execution sandboxes at the end of a task run.

The exploit agents' ``run_python`` / ``execute_bash`` tools start a podman
container keyed by the worker ``namespace`` (per case/finding). Tearing it down
reliably is tricky: the worker runs inside an ``AgentTool`` sub-``Runner`` with
its own ``invocation_id``, so matching teardown to the worker's id across that
boundary is fragile.

Instead this plugin keys off the **outer** run. A fresh plugin instance is built
per task iteration (``TaskRunner._build_plugins``) and propagated into the inner
``AgentTool`` runners, so ``before_run_callback`` fires first for the outer run:
we record that ``invocation_id`` as the root, and only when ``after_run_callback``
fires for that same root (the outer run finishing — the hook that reliably fires,
as the metrics/trace plugins demonstrate and the probe in
``tests/units/.../plugins/test_run_callbacks.py`` verifies) do we sweep every
sandbox. Safe because code-exec runs are sequential. ADK only awaits the hook
after the run's event stream is consumed to completion, so a run that raises or
is cancelled mid-stream skips it — ``TaskRunner.run``'s finally-sweep, ``atexit``
and the container TTL remain backstops for those paths.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from contractor.tools.podman import teardown_all

logger = logging.getLogger(__name__)


class SandboxCleanupPlugin(BasePlugin):
    """Sweep code-exec sandboxes when the outer task run finishes."""

    def __init__(self, name: str = "sandbox_cleanup") -> None:
        super().__init__(name=name)
        self._root: str | None = None

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        # First run seen (the outer task run) is the root we tear down on.
        if self._root is None:
            self._root = getattr(invocation_context, "invocation_id", None)

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        # Only act when the outer run ends; inner AgentTool sub-runs are skipped
        # so a mid-task sub-run completion doesn't kill the live sandbox.
        if self._root is not None and invocation_id != self._root:
            return
        try:
            teardown_all()
        except Exception:  # cleanup must never break the run
            logger.exception("sandbox teardown failed")
