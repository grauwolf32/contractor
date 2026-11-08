"""Classified transport outcomes shared by bridge consumers and implementations."""

from contractor_runtime.core.tool_errors import ToolExecutionError


class BridgeInfrastructureError(ToolExecutionError):
    def __init__(self):
        super().__init__("http_bridge_failed")


class BridgeFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
