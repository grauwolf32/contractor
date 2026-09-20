"""Suppress SDK/HTTP diagnostics only inside Contractor-owned MCP tasks.

The SDK logs session IDs and protocol bodies at several levels. Those records
are not a safe observation surface. Contractor emits its own bounded codes.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

mcp_private_task: ContextVar[bool] = ContextVar("contractor_mcp_private_task", default=False)


class _PrivateMCPFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not mcp_private_task.get()


def install_private_log_filter() -> None:
    # Apply to origin loggers: ancestor logger filters do not filter propagated
    # records, and changing root handlers would affect unrelated application IO.
    for name in (
        "",  # The pinned SDK also uses logging.warning/error directly.
        "client",  # ClientSession's logger name in mcp 1.30.0.
        "mcp.client.session",
        "mcp.client.streamable_http",
        "mcp.shared.session",
        "mcp.shared._httpx_utils",
        "httpx",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(item, _PrivateMCPFilter) for item in logger.filters):
            logger.addFilter(_PrivateMCPFilter())
