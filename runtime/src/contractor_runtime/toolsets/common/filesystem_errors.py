"""The model-facing error type of the workspace file tools.

The filesystem, edit_files and workspace_changes toolsets raise the same
stable codes (ERR-R10 in docs/spec/platform/errors.md), so they share one error
type here instead of one toolset importing another.
"""

from __future__ import annotations


class FilesystemToolError(RuntimeError):
    """Stable model-facing failure without content or physical backend detail."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Filesystem operation failed ({code})")
