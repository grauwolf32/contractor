"""Tool-owned equivalence and repair text; no framework execution state."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepeatRule:
    key: tuple[str | int, ...]
    limit: int
    message: str
