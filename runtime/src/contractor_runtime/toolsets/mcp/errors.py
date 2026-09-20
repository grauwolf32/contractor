"""Content-free MCP boundary errors; never preserve a remote exception."""

from __future__ import annotations


class MCPError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(f"MCP operation failed ({code})")


class MCPCleanupError(MCPError):
    def __init__(self) -> None:
        super().__init__("cleanup_unconfirmed")
