"""Invocation-owned exact artifact read positions; no payloads or grants."""

import secrets

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.cursors import decode_cursor, encode_cursor

_FIELDS = {
    "namespace": str,
    "name": str,
    "revision": str,
    "start_line": int,
    "max_lines": int,
    "line_offset": int,
}


class ArtifactReadCursors:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._key = secrets.token_bytes(32)

    def issue(
        self, ref: ArtifactRef, start_line: int = 1, max_lines: int = 200, line_offset: int = 0
    ) -> str:
        exact = ref.require_exact()
        return encode_cursor(
            self._key,
            {
                "namespace": exact.namespace,
                "name": exact.name,
                "revision": exact.revision,
                "start_line": start_line,
                "max_lines": max_lines,
                "line_offset": line_offset,
            },
        )

    def decode(self, value: str) -> dict:
        try:
            result = decode_cursor(self._key, value, _FIELDS)
            ArtifactRef(
                namespace=result["namespace"], name=result["name"], revision=result["revision"]
            ).require_exact()
            return result
        except (ValueError, TypeError):
            raise ToolInputError(
                "Invalid or expired artifact cursor; start a fresh read "
                "or use the current task input cursor"
            ) from None
