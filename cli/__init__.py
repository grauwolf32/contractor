from dataclasses import dataclass


@dataclass(slots=True)
class EventView:
    event_type: str
    title: str
    body: str | None = None
    tone: str = "default"  # info | success | warning | error | muted


__all__ = ["EventView"]
