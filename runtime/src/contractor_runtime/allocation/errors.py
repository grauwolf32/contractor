"""Bounded private API failures for allocation lifecycle operations."""

from __future__ import annotations

from typing import Any


class AllocationError(Exception):
    """Stable private-API error which never includes allocation secrets."""

    def __init__(self, code: str, message: str, *, retryable: bool, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


def _conflict(message: str) -> AllocationError:
    return AllocationError("allocation_conflict", message, retryable=False, status_code=409)


def _not_found() -> AllocationError:
    return AllocationError(
        "allocation_not_found",
        "allocation is not active on this Runtime Agent",
        retryable=False,
        status_code=404,
    )
