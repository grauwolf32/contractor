from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class EventView:
    event_type: str
    title: str
    body: Optional[str] = None
    tone: str = "default"  # info | success | warning | error | muted


__all__ = ["EventView"]
