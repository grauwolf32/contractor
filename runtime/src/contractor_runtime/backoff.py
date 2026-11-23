"""Bounded exponential backoff shared by Control Plane reconnect loops."""

from __future__ import annotations

from collections.abc import Callable

MAX_BACKOFF_DOUBLINGS = 10


def bounded_backoff(failures: int, ceiling: float, jitter: Callable[[], float]) -> float:
    """Return the delay before retrying after ``failures`` consecutive failures.

    Doubles from 0.5s, stops at ``ceiling`` and spreads each delay over
    75%-125% so runtimes recovering from one Server restart do not reconnect
    in lockstep. ``jitter`` returns a value in [0, 1).
    """
    if ceiling <= 0:
        raise ValueError("backoff ceiling must be positive")
    base = min(ceiling, 0.5 * (2 ** min(max(failures, 1) - 1, MAX_BACKOFF_DOUBLINGS)))
    return min(ceiling, base * (0.75 + 0.5 * jitter()))
